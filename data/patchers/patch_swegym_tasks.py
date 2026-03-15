"""
patch_swegym_tasks.py
---------------------
Patches DCAgent2/swegym-tasks to fix MissingDependency failures and reduce
the number of unique Docker images to ≤ 11 (one per repo).

Strategy:
  1. Dockerfile: Pre-install the PyPI release of each repo's package
     (pre-built wheels = no compilation needed at runtime)
  2. test.sh: Clone repo at runtime, apply code fix, run pip install
     --no-build-isolation -e . (reuses pre-installed compiled extensions)
  3. solve.sh: Same as test.sh + also applies the code fix patch from config.json

Usage:
    python3 data/patchers/patch_swegym_tasks.py \
        --output-dir /tmp/swegym_patched_v5 \
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
# Per-repo Dockerfile templates
# Pre-install PyPI packages (pre-built wheels) to avoid compilation at runtime
# ---------------------------------------------------------------------------

_DOCKERFILE_BASE = """\
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
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:$PATH"
"""

# Per-repo: extra apt packages + pip packages to pre-install in Dockerfile
# These are installed from PyPI (pre-built wheels) to avoid compilation
REPO_DEPS = {
    "pandas-dev/pandas": {
        "apt": [],
        # Install pandas + all its build deps as pre-built wheels
        # --no-build-isolation at runtime will reuse these
        "pip": [
            "pandas",
            "numpy",
            "python-dateutil",
            "pytz",
            "cython",
            "versioneer",
            "pytest",
            "pytest-xdist",
        ],
        # Runtime: use --no-build-isolation to reuse pre-installed compiled extensions
        "install_flags": "--no-build-isolation",
    },
    "Project-MONAI/MONAI": {
        "apt": [],
        "pip": [
            "torch --index-url https://download.pytorch.org/whl/cpu",
            "monai",
            "numpy",
            "nibabel",
            "pillow",
            "scipy",
            "pytest",
            "pytest-xdist",
        ],
        "install_flags": "--no-build-isolation",
    },
    "getmoto/moto": {
        "apt": [],
        "pip": [
            "boto3",
            "botocore",
            "moto[all]",
            "pytest",
            "pytest-xdist",
        ],
        "install_flags": "",
    },
    "python/mypy": {
        "apt": [],
        "pip": [
            "mypy",
            "mypy-extensions>=1.0",
            "typing-extensions>=4.1",
            "tomli",
            "pytest",
            "pytest-xdist",
        ],
        "install_flags": "--no-build-isolation",
    },
    "iterative/dvc": {
        "apt": [],
        "pip": [
            "dvc",
            "gitpython",
            "shtab>=1.3.4",
            "voluptuous",
            "funcy>=1.14",
            "pathspec>=0.9.0",
            "shortuuid>=0.5",
            "diskcache>=5.2.1",
            "rich>=12",
            "pytest",
            "pytest-xdist",
        ],
        "install_flags": "--no-build-isolation",
    },
    "dask/dask": {
        "apt": [],
        "pip": [
            "dask[complete]",
            "numpy",
            "pandas",
            "toolz",
            "fsspec",
            "pytest",
            "pytest-xdist",
        ],
        "install_flags": "--no-build-isolation",
    },
    "modin-project/modin": {
        "apt": [],
        "pip": [
            "modin[all]",
            "pandas",
            "numpy",
            "boto3",
            "pytest",
            "pytest-xdist",
        ],
        "install_flags": "--no-build-isolation",
    },
    "pydantic/pydantic": {
        "apt": [],
        # pydantic v2 has pre-built wheels (pydantic-core), no Rust compilation needed
        "pip": [
            "pydantic",
            "pydantic-core",
            "annotated-types",
            "typing-extensions",
            "pytest",
            "pytest-xdist",
        ],
        "install_flags": "--no-build-isolation",
    },
    "conan-io/conan": {
        "apt": ["cmake"],
        "pip": [
            "conan",
            "pytest",
            "pytest-xdist",
        ],
        "install_flags": "--no-build-isolation",
    },
    "facebookresearch/hydra": {
        "apt": [],
        "pip": [
            "hydra-core",
            "omegaconf>=2.2",
            "antlr4-python3-runtime==4.9.3",
            "pytest",
            "pytest-xdist",
        ],
        "install_flags": "--no-build-isolation",
    },
    "bokeh/bokeh": {
        "apt": ["nodejs", "npm"],
        "pip": [
            "bokeh",
            "pytest",
            "pytest-xdist",
        ],
        "install_flags": "--no-build-isolation",
    },
}


def build_dockerfile(repo: str) -> str:
    """Build a Dockerfile for the given repo with pre-installed PyPI packages."""
    deps = REPO_DEPS.get(repo, {"apt": [], "pip": [], "install_flags": ""})
    apt_pkgs = deps.get("apt", [])
    pip_pkgs = deps.get("pip", [])

    lines = [_DOCKERFILE_BASE]

    # Extra apt packages (if any)
    if apt_pkgs:
        lines.append("RUN apt-get update && apt-get install -y --no-install-recommends \\\n")
        for pkg in apt_pkgs:
            lines.append(f"        {pkg} \\\n")
        lines.append("    && rm -rf /var/lib/apt/lists/*\n")

    # Pre-install pip packages (pre-built wheels from PyPI)
    if pip_pkgs:
        for pkg in pip_pkgs:
            lines.append(
                f"RUN python3 -m pip install --break-system-packages {pkg} || true\n"
            )

    return "".join(lines)


# ---------------------------------------------------------------------------
# Runtime setup preamble (injected into test.sh and solve.sh)
# ---------------------------------------------------------------------------

def build_clone_preamble(repo: str, commit: str) -> str:
    """Shell snippet to clone the repo at the specific commit at runtime."""
    return f"""\
# --- Runtime repo setup ---
if [ ! -d /testbed/repo/.git ]; then
    git clone https://github.com/{repo}.git /testbed/repo 2>/dev/null || \\
    git clone --depth=50 https://github.com/{repo}.git /testbed/repo
    cd /testbed/repo && git fetch origin {commit} 2>/dev/null || git fetch --depth=50 origin {commit}
    git checkout {commit}
fi
# --- End runtime repo setup ---
"""


def build_ensure_deps(repo: str) -> str:
    """Build the ensure_dependencies function for the given repo."""
    deps = REPO_DEPS.get(repo, {"install_flags": ""})
    install_flags = deps.get("install_flags", "")

    return f"""\
ensure_dependencies() {{
    log "Installing project dependencies"

    if [ -f requirements-dev.txt ]; then
        log "Installing requirements-dev.txt"
        python3 -m pip install --break-system-packages -r requirements-dev.txt || true
    fi

    if [ -f requirements.txt ]; then
        log "Installing requirements.txt"
        python3 -m pip install --break-system-packages -r requirements.txt || true
    fi

    if [ -f pyproject.toml ] || [ -f setup.py ] || [ -f setup.cfg ]; then
        log "Installing project in editable mode"
        python3 -m pip install --break-system-packages {install_flags} -e . || \\
        python3 -m pip install --break-system-packages -e . || true
    fi

    python3 -m pip install --break-system-packages "pytest>=8.0.0" "pytest-xdist>=3.5.0" || true
}}"""


def patch_test_sh(content: str, repo: str, commit: str) -> str:
    """Patch test.sh to:
    1. Clone the repo at runtime (before cd $REPO_DIR)
    2. Fix ensure_dependencies to use --break-system-packages + --no-build-isolation
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
    if cd_marker in content:
        content = content.replace(cd_marker, clone_preamble + cd_marker, 1)

    return content


def patch_solve_sh(content: str, repo: str, commit: str, code_patch: str = "") -> str:
    """Patch solve.sh to clone the repo at runtime and apply the code fix.

    The oracle needs to:
    1. Clone at base_commit (buggy code)
    2. Apply the code fix (from config.json["patch"])
    3. Apply the test patch (already in solve.sh)
    4. Run tests -> should pass
    """
    clone_preamble = build_clone_preamble(repo, commit)

    # Build the code fix application snippet
    code_fix_snippet = ""
    if code_patch:
        code_fix_snippet = f"""
# --- Apply code fix (oracle: restore fixed code) ---
code_fix_file="$(mktemp /tmp/swegym-code-fix-XXXX.diff)"
cat <<'CODE_FIX_EOF' > "$code_fix_file"
{code_patch}
CODE_FIX_EOF
git apply --whitespace=nowarn --apply "$code_fix_file" || git apply --whitespace=fix --apply "$code_fix_file" || true
rm -f "$code_fix_file"
# --- End code fix ---
"""

    # Insert clone + code fix before the first cd /testbed/repo
    cd_marker = "cd /testbed/repo"
    if cd_marker in content and "git clone" not in content:
        content = content.replace(cd_marker, clone_preamble + cd_marker + code_fix_snippet, 1)

    # Fix pip calls to use --break-system-packages
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
    """Strip exactly 8 leading spaces from lines that start with 8 spaces.

    The original swegym solve.sh and test.sh are indented by 8 spaces on all
    script lines, but heredoc content (diff lines) may start at column 0.
    We strip 8 spaces only from lines that have them.
    """
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
    config_path = task_dir / "tests" / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            code_patch = cfg.get("patch", "")
        except Exception:
            pass

    # 1. Dockerfile - repo-specific, no git clone
    dockerfile_path = task_dir / "environment" / "Dockerfile"
    new_dockerfile = build_dockerfile(repo)
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
        default=pathlib.Path("/tmp/swegym_patched_v5"),
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
            # Extract
            task_dir = extract_task(task_binary, task_name, args.output_dir)

            # Get repo and commit from config.json
            config_path = task_dir / "tests" / "config.json"
            if config_path.exists():
                cfg = json.loads(config_path.read_text())
                repo = cfg.get("repo", "unknown")
                commit = cfg.get("base_commit", "main")
            else:
                repo = "unknown"
                commit = "main"

            repo_counts[repo] = repo_counts.get(repo, 0) + 1

            # Patch
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

    # Count unique Dockerfiles
    dockerfiles = set()
    for p in args.output_dir.rglob("environment/Dockerfile"):
        dockerfiles.add(p.read_text())
    log.info(f"\nUnique Dockerfiles after patching: {len(dockerfiles)} (target: ≤11)")


if __name__ == "__main__":
    main()
