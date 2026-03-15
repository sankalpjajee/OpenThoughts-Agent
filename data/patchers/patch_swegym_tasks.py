#!/usr/bin/env python3
"""
patch_swegym_tasks.py
---------------------
Patches DCAgent2/swegym-tasks to fix MissingDependency failures and reduce
the number of unique Docker images to ≤ 11 (one per repo).

Problem:
  - Original Dockerfile: ubuntu:24.04 + git clone at specific commit
    → Each task has a unique Dockerfile (2438 unique images)
    → Missing heavy build deps (pandas needs Cython, MONAI needs torch, etc.)

Fix strategy:
  1. Dockerfile: Remove git clone (make it repo-specific, not commit-specific)
     → Pre-install heavy deps for each repo
     → 11 unique images instead of 2438
  2. test.sh: Add git clone at runtime (clone repo + checkout specific commit)
     → Fix ensure_dependencies to use --break-system-packages
  3. solve.sh: Add git clone at runtime before applying patches

Usage:
    python3 data/patchers/patch_swegym_tasks.py \
        --output-dir /tmp/swegym_patched_v1 \
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
# Per-repo Dockerfile templates (no git clone - that happens at runtime)
# ---------------------------------------------------------------------------

_DOCKERFILE_HEADER = """\
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
"""

_DOCKERFILE_FOOTER = """\
ENV PATH="/root/.local/bin:$PATH"
"""

# Per-repo extra apt packages and pip pre-installs
REPO_DEPS = {
    "pandas-dev/pandas": {
        "apt": ["cython3", "python3-numpy", "libhdf5-dev", "libopenblas-dev"],
        "pip": ["cython", "numpy", "python-dateutil", "pytz"],
    },
    "Project-MONAI/MONAI": {
        "apt": [],
        # CPU-only torch to avoid 2GB GPU wheel
        "pip": [
            "torch --index-url https://download.pytorch.org/whl/cpu",
            "numpy",
            "nibabel",
            "pillow",
            "scipy",
        ],
    },
    "getmoto/moto": {
        "apt": [],
        "pip": ["boto3", "botocore"],
    },
    "python/mypy": {
        "apt": [],
        "pip": [
            "mypy-extensions>=1.0",
            "typing-extensions>=4.1",
            "tomli",
        ],
    },
    "iterative/dvc": {
        "apt": [],
        "pip": [
            "gitpython",
            "shtab>=1.3.4",
            "voluptuous",
            "funcy>=1.14",
            "pathspec>=0.9.0",
            "shortuuid>=0.5",
            "diskcache>=5.2.1",
            "rich>=12",
        ],
    },
    "dask/dask": {
        "apt": [],
        "pip": ["numpy", "pandas", "toolz", "fsspec"],
    },
    "modin-project/modin": {
        "apt": [],
        "pip": ["pandas", "numpy"],
    },
    "pydantic/pydantic": {
        "apt": ["cargo", "rustc"],
        "pip": ["maturin"],
    },
    "conan-io/conan": {
        "apt": ["cmake"],
        "pip": [],
    },
    "facebookresearch/hydra": {
        "apt": [],
        "pip": ["omegaconf>=2.2", "antlr4-python3-runtime==4.9.3"],
    },
    "bokeh/bokeh": {
        "apt": ["nodejs", "npm"],
        "pip": [],
    },
}


def build_dockerfile(repo: str) -> str:
    """Build a Dockerfile for the given repo with pre-installed deps."""
    deps = REPO_DEPS.get(repo, {"apt": [], "pip": []})
    apt_pkgs = deps.get("apt", [])
    pip_pkgs = deps.get("pip", [])

    lines = [_DOCKERFILE_HEADER]

    # Extra apt packages
    if apt_pkgs:
        for pkg in apt_pkgs:
            lines.append(f"        {pkg} \\\n")

    lines.append("    && rm -rf /var/lib/apt/lists/*\n")

    # Pre-install pip packages
    if pip_pkgs:
        for pkg in pip_pkgs:
            lines.append(
                f"RUN python3 -m pip install --break-system-packages {pkg} || true\n"
            )

    lines.append(_DOCKERFILE_FOOTER)
    return "".join(lines)


# ---------------------------------------------------------------------------
# Runtime setup preamble (injected into test.sh and solve.sh)
# ---------------------------------------------------------------------------

def build_clone_preamble(repo: str, commit: str) -> str:
    """Shell snippet to clone the repo at the specific commit at runtime."""
    return f"""\
# --- Runtime repo setup ---
if [ ! -d /testbed/repo/.git ]; then
    git clone --depth=1 https://github.com/{repo}.git /testbed/repo || \\
    git clone https://github.com/{repo}.git /testbed/repo
    cd /testbed/repo && git fetch --depth=1 origin {commit} && git checkout {commit}
fi
# --- End runtime repo setup ---
"""


ENSURE_DEPS_FUNCTION = """\
ensure_dependencies() {
    log "Installing base Python tooling"
    python3 -m pip install --break-system-packages --upgrade pip setuptools wheel

    if [ -f requirements-dev.txt ]; then
        log "Installing requirements-dev.txt"
        python3 -m pip install --break-system-packages -r requirements-dev.txt || true
    fi

    if [ -f requirements.txt ]; then
        log "Installing requirements.txt"
        python3 -m pip install --break-system-packages -r requirements.txt || true
    fi

    if [ -f pyproject.toml ] || [ -f setup.py ]; then
        log "Installing project in editable mode"
        python3 -m pip install --break-system-packages -e . || true
        # Auto-discover and install all optional extras
        cat > /tmp/_discover_extras.py << 'PYEOF'
import ast, configparser, pathlib, re, sys
try:
    pp = pathlib.Path('pyproject.toml')
    if pp.exists():
        txt = pp.read_text()
        m = re.search(r'\\[project\\.optional-dependencies\\](.+?)(?:\\n\\[|\\Z)', txt, re.S)
        if m:
            keys = re.findall(r'^(\\w[\\w-]*)\\s*=', m.group(1), re.M)
            if keys: print(','.join(keys)); sys.exit(0)
    cfg = pathlib.Path('setup.cfg')
    if cfg.exists():
        cp = configparser.ConfigParser(); cp.read(str(cfg))
        if 'options.extras_require' in cp:
            print(','.join(cp['options.extras_require'].keys())); sys.exit(0)
    sp = pathlib.Path('setup.py')
    if sp.exists():
        tree = ast.parse(sp.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == 'extras_require':
                if isinstance(node.value, ast.Dict):
                    keys = [k.value for k in node.value.keys if isinstance(k, (ast.Constant, ast.Str))]
                    if keys: print(','.join(keys)); sys.exit(0)
except Exception: pass
PYEOF
        EXTRAS=$(python3 /tmp/_discover_extras.py 2>/dev/null)
        if [ -n "$EXTRAS" ]; then
            log "Installing extras: $EXTRAS"
            python3 -m pip install --break-system-packages -e ".[$EXTRAS]" 2>/dev/null || true
        fi
    fi

    python3 -m pip install --break-system-packages "pytest>=8.0.0" "pytest-xdist>=3.5.0" || true
}"""


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
            content = content[:start] + ENSURE_DEPS_FUNCTION + content[end:]

    # Step 2: Add git clone before "cd $REPO_DIR"
    clone_preamble = build_clone_preamble(repo, commit)
    cd_marker = 'cd "$REPO_DIR"'
    if cd_marker in content:
        content = content.replace(cd_marker, clone_preamble + cd_marker, 1)

    return content


def patch_solve_sh(content: str, repo: str, commit: str) -> str:
    """Patch solve.sh to clone the repo at runtime if not already present."""
    clone_preamble = build_clone_preamble(repo, commit)

    # The solve.sh starts with shebang and set -Eeuo pipefail, then cd /testbed/repo
    # Insert clone before the first cd /testbed/repo
    cd_marker = "cd /testbed/repo"
    if cd_marker in content and "git clone" not in content:
        content = content.replace(cd_marker, clone_preamble + cd_marker, 1)

    # Also fix pip calls to use --break-system-packages
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

def patch_task(task_dir: pathlib.Path, repo: str, commit: str, dry_run: bool = False) -> dict:
    """Patch a single extracted swegym task directory."""
    changes = {}

    # 1. Dockerfile - repo-specific, no git clone
    dockerfile_path = task_dir / "environment" / "Dockerfile"
    new_dockerfile = build_dockerfile(repo)
    if not dry_run:
        dockerfile_path.parent.mkdir(parents=True, exist_ok=True)
        dockerfile_path.write_text(new_dockerfile)
    changes["Dockerfile"] = True

    # 2. test.sh - add git clone + fix ensure_dependencies
    test_sh_path = task_dir / "tests" / "test.sh"
    if test_sh_path.exists():
        original = test_sh_path.read_text()
        patched = patch_test_sh(original, repo, commit)
        if patched != original:
            if not dry_run:
                test_sh_path.write_text(patched)
            changes["test.sh"] = True
        else:
            changes["test.sh"] = False
    else:
        changes["test.sh"] = False

    # 3. solve.sh - add git clone
    solve_sh_path = task_dir / "solution" / "solve.sh"
    if solve_sh_path.exists():
        original = solve_sh_path.read_text()
        patched = patch_solve_sh(original, repo, commit)
        if patched != original:
            if not dry_run:
                solve_sh_path.write_text(patched)
            changes["solve.sh"] = True
        else:
            changes["solve.sh"] = False
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
        default=pathlib.Path("/tmp/swegym_patched_v1"),
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
