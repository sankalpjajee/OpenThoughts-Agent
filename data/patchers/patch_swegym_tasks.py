"""
patch_swegym_tasks.py
---------------------
Patches DCAgent2/swegym-tasks to fix MissingDependency failures and reduce
the number of unique Docker images to ≤ 11 (one per repo).

Key insight: Daytona has a 600s Dockerfile BUILD timeout. Heavy pip installs
(moto[all], torch, pandas) must NOT go in the Dockerfile — they must be
installed at runtime in test.sh/solve.sh.

Strategy:
  1. Dockerfile: Minimal ubuntu:24.04 with only apt packages (no pip installs)
     - For heavy compiled repos (pandas, MONAI, modin): use pre-built custom
       image from ghcr.io that already has all compiled deps installed
  2. test.sh: Clone repo at runtime, install deps, run tests
     - For moto: pip install -e .[all] to get all extras
     - For compiled packages (pandas, MONAI): pip install --no-build-isolation -e .
     - For pure Python (mypy, dvc): pip install -e .
  3. solve.sh: Same as test.sh + also applies the code fix patch from config.json

Usage:
    python3 data/patchers/patch_swegym_tasks.py \\
        --output-dir /tmp/swegym_patched_v9 \\
        [--limit 10] [--dry-run]
"""

import argparse
import gzip
import io
import json
import logging
import pathlib
import tarfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dockerfile templates
# ---------------------------------------------------------------------------

# Minimal Dockerfile for pure-Python repos
_DOCKERFILE_TEMPLATE = """\
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /testbed
RUN apt-get update && \\
    apt-get install -y --no-install-recommends \\
        git \\
        python3 \\
        python3-pip \\
        python3-dev \\
        python3-venv \\
        build-essential \\
        pkg-config \\
        ca-certificates \\
        curl \\
        jq \\
        {extra_apt} \\
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:$PATH"
"""

# Custom pre-built images for heavy compiled repos
# Image already has compiled deps (pandas, torch, etc.) installed
_CUSTOM_IMAGE_DOCKERFILE = """\
FROM {image}
WORKDIR /testbed
RUN apt-get update && \\
    apt-get install -y --no-install-recommends git jq curl wget \\
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:$PATH"
"""

# Map repo -> custom image (version-aware for pandas)
REGISTRY = "ghcr.io/sankalpjajee"

REPO_CUSTOM_IMAGES = {
    "Project-MONAI/MONAI": f"{REGISTRY}/swegym-monai:latest",
    "modin-project/modin": f"{REGISTRY}/swegym-modin:latest",
    # pandas: version-specific images resolved at runtime based on config.json version field
}

# Per-repo extra apt packages (lightweight, fast to install)
REPO_APT_DEPS = {
    "conan-io/conan": ["cmake"],
    "bokeh/bokeh": ["nodejs", "npm"],
}

# Per-repo: pip packages to install at RUNTIME (in test.sh/solve.sh)
# For custom-image repos (pandas, MONAI, modin), deps are already in the image.
REPO_RUNTIME_DEPS = {
    "pandas-dev/pandas": [
        "cython versioneer",
    ],
    "Project-MONAI/MONAI": [
        "nibabel pillow scipy einops parameterized",
    ],
    "modin-project/modin": [
        "dask[complete]",
    ],
    "getmoto/moto": [
        "boto3 botocore",
        "responses werkzeug flask",
    ],
    "python/mypy": [
        "mypy-extensions typing-extensions tomli",
    ],
    "iterative/dvc": [
        "gitpython shtab voluptuous funcy pathspec shortuuid diskcache rich",
    ],
    "dask/dask": [
        "numpy pandas toolz fsspec",
    ],
    "pydantic/pydantic": [
        "pydantic-core annotated-types typing-extensions",
    ],
    "facebookresearch/hydra": [
        "omegaconf antlr4-python3-runtime==4.9.3",
    ],
    "bokeh/bokeh": [],
    "conan-io/conan": [],
}

# For compiled packages, use --no-build-isolation at editable install time
REPO_INSTALL_FLAGS = {
    "pandas-dev/pandas": "--no-build-isolation",
    "Project-MONAI/MONAI": "--no-build-isolation",
    "python/mypy": "--no-build-isolation",
    "iterative/dvc": "--no-build-isolation",
    "dask/dask": "--no-build-isolation",
    "modin-project/modin": "--no-build-isolation",
    "pydantic/pydantic": "--no-build-isolation",
    "conan-io/conan": "--no-build-isolation",
    "facebookresearch/hydra": "--no-build-isolation",
    "bokeh/bokeh": "--no-build-isolation",
}

# Repos that need extras installed (e.g. moto[all])
REPO_INSTALL_EXTRAS = {
    "getmoto/moto": "[all]",
}


def build_dockerfile(repo: str, version: str = "") -> str:
    """Build a Dockerfile for the given repo."""
    # pandas: version-specific custom images
    if repo == "pandas-dev/pandas":
        major_minor = ".".join(version.lstrip("v").split(".")[:2])
        image = f"{REGISTRY}/swegym-pandas-{major_minor}:latest"
        return _CUSTOM_IMAGE_DOCKERFILE.format(image=image)

    # Other heavy repos with single custom image
    if repo in REPO_CUSTOM_IMAGES:
        image = REPO_CUSTOM_IMAGES[repo]
        return _CUSTOM_IMAGE_DOCKERFILE.format(image=image)

    # Pure-Python repos: minimal ubuntu:24.04
    extra_apt = " ".join(REPO_APT_DEPS.get(repo, []))
    if not extra_apt:
        extra_apt = "wget"
    return _DOCKERFILE_TEMPLATE.format(extra_apt=extra_apt)


def build_runtime_install_snippet(repo: str) -> str:
    """Build shell snippet to install runtime deps."""
    pkgs = REPO_RUNTIME_DEPS.get(repo, [])
    if not pkgs:
        return ""
    lines = ["# --- Install runtime dependencies ---"]
    for pkg_line in pkgs:
        lines.append(
            "python3 -m pip install --break-system-packages " + pkg_line + " || true"
        )
    lines.append("# --- End runtime dependencies ---")
    return "\n".join(lines) + "\n"


def build_ensure_deps(repo: str) -> str:
    """Build the ensure_dependencies function for the given repo."""
    install_flags = REPO_INSTALL_FLAGS.get(repo, "")
    extras = REPO_INSTALL_EXTRAS.get(repo, "")
    runtime_install = build_runtime_install_snippet(repo)

    # Build the pip install line carefully (no Python f-string backslash issues)
    pip_install_line = (
        "python3 -m pip install --break-system-packages "
        + install_flags
        + " -e ."
        + extras
        + " || \\\n"
        + "        python3 -m pip install --break-system-packages -e ."
        + extras
        + " || true"
    )

    lines = [
        "ensure_dependencies() {",
        '    log "Installing project dependencies"',
        "",
    ]
    if runtime_install:
        lines.append("    " + runtime_install.replace("\n", "\n    ").rstrip())
        lines.append("")
    lines += [
        "    if [ -f requirements-dev.txt ]; then",
        '        log "Installing requirements-dev.txt"',
        "        python3 -m pip install --break-system-packages -r requirements-dev.txt || true",
        "    fi",
        "",
        "    if [ -f requirements.txt ]; then",
        '        log "Installing requirements.txt"',
        "        python3 -m pip install --break-system-packages -r requirements.txt || true",
        "    fi",
        "",
        "    if [ -f pyproject.toml ] || [ -f setup.py ] || [ -f setup.cfg ]; then",
        '        log "Installing project in editable mode"',
        "        " + pip_install_line,
        "    fi",
        "",
        '    python3 -m pip install --break-system-packages "pytest>=8.0.0" "pytest-xdist>=3.5.0" || true',
        "}",
    ]
    return "\n".join(lines)


def build_clone_preamble(repo: str, commit: str) -> str:
    """Shell snippet to clone the repo at the specific commit at runtime."""
    lines = [
        "# --- Runtime repo setup ---",
        "if [ ! -d /testbed/repo/.git ]; then",
        "    git clone https://github.com/" + repo + ".git /testbed/repo 2>/dev/null || \\",
        "    git clone --depth=100 https://github.com/" + repo + ".git /testbed/repo",
        "fi",
        "cd /testbed/repo",
        "git fetch origin " + commit + " 2>/dev/null || git fetch --depth=100 origin " + commit + " 2>/dev/null || true",
        "git checkout " + commit + " 2>/dev/null || git checkout -b work_" + commit[:8] + " " + commit + " 2>/dev/null || true",
        "# --- End runtime repo setup ---",
        "",
    ]
    return "\n".join(lines)


def patch_test_sh(content: str, repo: str, commit: str) -> str:
    """Patch test.sh to:
    1. Clone the repo at runtime (before cd $REPO_DIR)
    2. Fix ensure_dependencies to use --break-system-packages
    """
    # Step 1: Replace ensure_dependencies function
    start = content.find("ensure_dependencies()")
    if start != -1:
        brace_start = content.find("{", start)
        if brace_start != -1:
            depth = 0
            i = brace_start
            while i < len(content):
                if content[i] == "{":
                    depth += 1
                elif content[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
                i += 1
            else:
                end = len(content)
            content = content[:start] + build_ensure_deps(repo) + content[end:]

    # Step 2: Add git clone before "cd $REPO_DIR"
    clone_preamble = build_clone_preamble(repo, commit)
    cd_marker = 'cd "$REPO_DIR"'
    if cd_marker in content and "git clone" not in content:
        content = content.replace(cd_marker, clone_preamble + cd_marker, 1)

    return content


def patch_solve_sh(content: str, repo: str, commit: str, code_patch: str = "") -> str:
    """Patch solve.sh to clone the repo at runtime and apply the code fix."""
    clone_preamble = build_clone_preamble(repo, commit)

    # Build the code fix application snippet
    code_fix_snippet = ""
    if code_patch:
        code_fix_lines = [
            "",
            "# --- Apply code fix (oracle: restore fixed code) ---",
            'code_fix_file="$(mktemp /tmp/swegym-code-fix-XXXX.diff)"',
            "cat <<'CODE_FIX_EOF' > \"$code_fix_file\"",
            code_patch,
            "CODE_FIX_EOF",
            "git apply --whitespace=nowarn --apply \"$code_fix_file\" || \\",
            "git apply --whitespace=fix --apply \"$code_fix_file\" || \\",
            "patch -p1 --forward < \"$code_fix_file\" || true",
            "rm -f \"$code_fix_file\"",
            "# --- End code fix ---",
            "",
        ]
        code_fix_snippet = "\n".join(code_fix_lines)

    # Runtime dep install snippet for solve.sh
    runtime_install = build_runtime_install_snippet(repo)
    install_flags = REPO_INSTALL_FLAGS.get(repo, "")
    extras = REPO_INSTALL_EXTRAS.get(repo, "")

    pip_install_line = (
        "python3 -m pip install --break-system-packages "
        + install_flags
        + " -e ."
        + extras
        + " || \\\n"
        + "python3 -m pip install --break-system-packages -e ."
        + extras
        + " || true"
    )

    install_lines = [
        "",
        "# --- Install dependencies ---",
    ]
    if runtime_install:
        install_lines.append(runtime_install.rstrip())
    install_lines += [
        "if [ -f requirements.txt ]; then",
        "    python3 -m pip install --break-system-packages -r requirements.txt || true",
        "fi",
        pip_install_line,
        'python3 -m pip install --break-system-packages "pytest>=8.0.0" "pytest-xdist>=3.5.0" || true',
        "# --- End install ---",
        "",
    ]
    install_snippet = "\n".join(install_lines)

    # Insert clone + code fix + install before the first cd /testbed/repo
    cd_marker = "cd /testbed/repo"
    if cd_marker in content and "git clone" not in content:
        content = content.replace(
            cd_marker,
            clone_preamble + code_fix_snippet + install_snippet + cd_marker,
            1,
        )

    # Fix any remaining pip calls
    content = content.replace(
        "python3 -m pip install -e .",
        "python3 -m pip install --break-system-packages -e .",
    )
    content = content.replace(
        "python3 -m pip install -r ",
        "python3 -m pip install --break-system-packages -r ",
    )

    return content


# ---------------------------------------------------------------------------
# Task patching
# ---------------------------------------------------------------------------

def dedent_sh(content: str) -> str:
    """Strip exactly 8 leading spaces from lines that start with 8 spaces."""
    result = []
    for line in content.split("\n"):
        if line.startswith("        "):  # 8 spaces
            result.append(line[8:])
        else:
            result.append(line)
    return "\n".join(result)


def patch_task(task_dir: pathlib.Path, repo: str, commit: str, dry_run: bool = False) -> dict:
    """Patch a single extracted swegym task directory."""
    changes = {}

    # Read code_patch from tests/config.json
    code_patch = ""
    version = ""
    config_path = task_dir / "tests" / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            code_patch = cfg.get("patch", "")
            version = cfg.get("version", "")
        except Exception:
            pass

    # 1. Dockerfile - custom image for heavy repos, minimal for pure-Python
    dockerfile_path = task_dir / "environment" / "Dockerfile"
    new_dockerfile = build_dockerfile(repo, version=version)
    if not dry_run:
        dockerfile_path.parent.mkdir(parents=True, exist_ok=True)
        dockerfile_path.write_text(new_dockerfile)
    changes["Dockerfile"] = True

    # 2. test.sh - dedent + add git clone + fix ensure_dependencies
    test_sh_path = task_dir / "tests" / "test.sh"
    if test_sh_path.exists():
        original = test_sh_path.read_text()
        dedented = dedent_sh(original)
        patched = patch_test_sh(dedented, repo, commit)
        if not dry_run:
            test_sh_path.write_text(patched)
        changes["test.sh"] = True
    else:
        changes["test.sh"] = False

    # 3. solve.sh - dedent + add git clone + embed code fix
    solve_sh_path = task_dir / "solution" / "solve.sh"
    if solve_sh_path.exists():
        original = solve_sh_path.read_text()
        dedented = dedent_sh(original)
        patched = patch_solve_sh(dedented, repo, commit, code_patch=code_patch)
        if not dry_run:
            solve_sh_path.write_text(patched)
        changes["solve.sh"] = True
    else:
        changes["solve.sh"] = False

    return changes


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def extract_task(task_binary: bytes, task_name: str, output_dir: pathlib.Path) -> pathlib.Path:
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(io.BytesIO(task_binary)) as gz:
        with tarfile.open(fileobj=gz) as tar:
            tar.extractall(str(task_dir))
    return task_dir


def main():
    parser = argparse.ArgumentParser(description="Patch DCAgent2/swegym-tasks")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/swegym_patched_v9"),
        help="Output directory for patched tasks",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no writes)")
    parser.add_argument(
        "--repo-id",
        default="DCAgent2/swegym-tasks",
        help="HuggingFace dataset repo ID",
    )
    args = parser.parse_args()

    from datasets import load_dataset
    log.info(f"Loading dataset {args.repo_id}...")
    ds = load_dataset(args.repo_id, split="train")
    total = len(ds) if args.limit is None else min(args.limit, len(ds))
    log.info(f"Processing {total} tasks...")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"patched": 0, "errors": 0}
    repo_counts = {}

    for i in range(total):
        row = ds[i]
        task_name = row["path"]
        task_binary = bytes(row["task_binary"])
        try:
            task_dir = extract_task(task_binary, task_name, args.output_dir)
            config_path = task_dir / "tests" / "config.json"
            if config_path.exists():
                cfg = json.loads(config_path.read_text())
                repo = cfg.get("repo", "unknown")
                commit = cfg.get("base_commit", "main")
            else:
                repo = "unknown"
                commit = "main"
            repo_counts[repo] = repo_counts.get(repo, 0) + 1
            changes = patch_task(task_dir, repo, commit, dry_run=args.dry_run)
            stats["patched"] += 1
            if i % 100 == 0:
                log.info(f"  [{i}/{total}] {task_name} repo={repo} commit={commit[:12]} changes={changes}")
        except Exception as e:
            log.error(f"  [{i}] {task_name} ERROR: {e}")
            stats["errors"] += 1

    log.info(f"\nDone. Patched: {stats['patched']}, Errors: {stats['errors']}")
    log.info(f"Output: {args.output_dir}")
    log.info("\nTasks per repo:")
    for repo, count in sorted(repo_counts.items(), key=lambda x: -x[1]):
        log.info(f"  {count:4d}  {repo}")

    dockerfiles = set()
    for p in args.output_dir.rglob("environment/Dockerfile"):
        dockerfiles.add(p.read_text())
    log.info(f"\nUnique Dockerfiles after patching: {len(dockerfiles)} (target: ≤11)")


if __name__ == "__main__":
    main()
