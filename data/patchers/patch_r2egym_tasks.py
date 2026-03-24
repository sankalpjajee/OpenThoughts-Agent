#!/usr/bin/env python3
"""
Patch R2E-Gym tasks from 4,578 unique docker images → 10 images (one per repo).

R2E-Gym/R2E-Gym-Subset contains 4,578 tasks across exactly 10 GitHub repos:
  pandas (1444), numpy (781), pillow (620), orange3 (482), aiohttp (299),
  tornado (261), scrapy (215), pyramid (189), datalad (179), coveragepy (108)

Each task currently uses a unique pre-built image (repo @ specific buggy commit).
After patching, each task uses one of 10 shared images where the repo is
pre-cloned at HEAD with all deps installed. The agent's only setup step is:

    git checkout {commit_hash}   # fast: seconds, not minutes

Key improvements over prior version:
  - pandas/numpy/pillow/aiohttp: install pre-built wheel (fast) instead of
    building from source (which times out in Daytona's 600s build limit)
  - orange3: add xvfb + libgl1 for headless Qt display (fixes 0% pass rate)
  - test.sh: uses Harbor-mounted /tests/test_*.py directly (no setup_files needed)
  - solve.sh: simple git checkout oracle

Usage
-----
    python patch_r2egym_tasks.py \\
        --sandbox-repo DCAgent2/r2egym_sandboxes \\
        --output-dir /path/to/patched \\
        [--limit N] \\
        [--upload-to SankalpKJ/r2egym-patched]
"""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import argparse

# ---------------------------------------------------------------------------
# Per-repo configuration
# ---------------------------------------------------------------------------

# Python version per repo — matched to the original docker images
_REPO_PYTHON_VERSION: dict[str, str] = {
    "pandas":     "3.11",
    "numpy":      "3.11",
    "pillow":     "3.11",
    "orange3":    "3.10",
    "aiohttp":    "3.11",
    "tornado":    "3.11",
    "scrapy":     "3.11",
    "pyramid":    "3.8",
    "datalad":    "3.11",
    "coveragepy": "3.7",
}

# GitHub clone URLs
_REPO_GITHUB_URL: dict[str, str] = {
    "pandas":     "https://github.com/pandas-dev/pandas.git",
    "numpy":      "https://github.com/numpy/numpy.git",
    "pillow":     "https://github.com/python-pillow/Pillow.git",
    "orange3":    "https://github.com/biolab/orange3.git",
    "aiohttp":    "https://github.com/aio-libs/aiohttp.git",
    "tornado":    "https://github.com/tornadoweb/tornado.git",
    "scrapy":     "https://github.com/scrapy/scrapy.git",
    "pyramid":    "https://github.com/Pylons/pyramid.git",
    "datalad":    "https://github.com/datalad/datalad.git",
    "coveragepy": "https://github.com/nedbat/coveragepy.git",
}

# Native apt packages needed beyond the common baseline
_REPO_EXTRA_APT: dict[str, str] = {
    "pandas":     "gfortran libopenblas-dev liblapack-dev pkg-config",
    "numpy":      "gfortran libopenblas-dev liblapack-dev pkg-config",
    "pillow":     "libjpeg-dev zlib1g-dev libpng-dev libtiff-dev libwebp-dev",
    # orange3: xvfb + libgl1 for headless Qt display (fixes 0% pass rate)
    "orange3":    "libxml2-dev libxslt1-dev libgl1-mesa-glx libglib2.0-0 xvfb libxkbcommon-x11-0 libdbus-1-3",
    "aiohttp":    "",
    "tornado":    "",
    "scrapy":     "libxml2-dev libxslt1-dev",
    "pyramid":    "",
    "datalad":    "git-annex",
    "coveragepy": "",
}

# Install commands run inside the cloned repo during Docker build.
#
# KEY STRATEGY for compiled repos (pandas, numpy, pillow, aiohttp):
#   1. Install pre-built wheel from PyPI first (fast, no compilation needed)
#   2. Then install the cloned source in editable mode WITHOUT rebuilding
#      C extensions (--no-build-isolation --no-deps skips the heavy build step)
# This avoids Daytona's 600s build timeout while still having the source
# available for git checkout at runtime.
_REPO_INSTALL_CMD: dict[str, str] = {
    # pandas: install wheel first for C extensions, then editable source
    "pandas": (
        "pip install 'pandas>=1.0,<3.0' numpy pyarrow pytest pytest-xdist hypothesis "
        "python-dateutil pytz xlrd openpyxl xlsxwriter odfpy tables "
        "beautifulsoup4 lxml html5lib scipy bottleneck numexpr "
        "2>/dev/null || true && "
        "pip install -e . --no-build-isolation --no-deps 2>/dev/null || true"
    ),
    # numpy: install wheel first for C extensions, then editable source
    "numpy": (
        "pip install 'numpy>=1.20,<2.0' pytest hypothesis cython "
        "2>/dev/null || true && "
        "pip install -e . --no-build-isolation --no-deps 2>/dev/null || true"
    ),
    # pillow: install wheel first for C extensions, then editable source
    "pillow": (
        "pip install 'Pillow>=8.0,<11.0' pytest pytest-timeout "
        "2>/dev/null || true && "
        "pip install -e . --no-build-isolation --no-deps 2>/dev/null || true"
    ),
    # orange3: PyQt5 headless (xvfb handles display at test time)
    "orange3": (
        "pip install PyQt5 2>/dev/null || true && "
        "pip install -e '.[test]' 2>/dev/null || pip install -e . 2>/dev/null || pip install orange3 2>/dev/null || true"
    ),
    # aiohttp: install wheel first to avoid C extension build timeout
    "aiohttp": (
        "pip install 'aiohttp>=3.0,<4.0' pytest pytest-asyncio aiohttp-cors "
        "2>/dev/null || true && "
        "pip install -e '.[dev]' --no-build-isolation --no-deps 2>/dev/null || "
        "pip install -e . --no-build-isolation --no-deps 2>/dev/null || true"
    ),
    "tornado":    "pip install -e . 2>/dev/null || pip install tornado; pip install pytest",
    "scrapy":     "pip install -e '.[tests]' 2>/dev/null || pip install -e . 2>/dev/null || pip install Scrapy pytest",
    "pyramid":    "pip install -e '.[testing]' 2>/dev/null || pip install -e . 2>/dev/null || pip install pyramid pytest",
    "datalad":    "pip install -e '.[devel]' 2>/dev/null || pip install -e . 2>/dev/null || pip install datalad pytest",
    "coveragepy": "pip install -e '.[dev]' 2>/dev/null || pip install -e . 2>/dev/null || pip install coverage pytest; pip install unittest-mixins mock 2>/dev/null || true",
}


def _build_dockerfile(repo_name: str) -> str:
    """Build the shared Dockerfile for a specific repo (one per repo, 10 total)."""
    python_version = _REPO_PYTHON_VERSION.get(repo_name, "3.11")
    github_url = _REPO_GITHUB_URL[repo_name]
    install_cmd = _REPO_INSTALL_CMD[repo_name]
    extra_apt = _REPO_EXTRA_APT.get(repo_name, "").strip()
    apt_extra_line = f"    {extra_apt} \\\n" if extra_apt else ""

    return f"""\
FROM python:{python_version}-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \\
    git curl wget jq \\
    build-essential \\
    libffi-dev libssl-dev \\
    locales locales-all tzdata \\
    tmux \\
{apt_extra_line}\
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

ENV PYTHONPATH=/testbed

# Pre-clone the repo and install all dependencies at HEAD.
# Agent only needs: git checkout {{commit_hash}}
RUN git clone {github_url} /testbed
WORKDIR /testbed
RUN mkdir -p /output && chmod 777 /output
RUN {install_cmd}

RUN mkdir -p /logs /r2e_tests
"""


# ---------------------------------------------------------------------------
# test.sh template
# ---------------------------------------------------------------------------
# Runs test_*.py files from /tests/ (Harbor mounts them there).
# Uses xvfb-run if available (for orange3 Qt headless).
# Writes reward 0 or 1 to /logs/verifier/reward.txt.
_TEST_SH = """\
#!/bin/bash
set -e
mkdir -p /logs/verifier

# Prefer the repo venv if it exists, fall back to system Python
if [ -d /testbed/.venv/bin ]; then
    export PATH=/testbed/.venv/bin:$PATH
    PYTHON=/testbed/.venv/bin/python
elif command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

# Ensure pytest is available
$PYTHON -m pytest --version &>/dev/null || $PYTHON -m pip install pytest -q

# Harbor mounts the task's tests/ directory at /tests/ in the container.
TEST_FILES=()
for f in /tests/test_*.py; do
    [ -f "$f" ] && TEST_FILES+=("$f")
done
if [ ${#TEST_FILES[@]} -eq 0 ]; then
    echo "ERROR: no test_*.py files found in /tests/" >&2
    echo 0 > /logs/verifier/reward.txt
    exit 1
fi

# Clean up stale bytecode
find /testbed -name '*.pyc' -delete 2>/dev/null || true
find /testbed -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Run pytest from / (neutral dir) so that python -m pytest does NOT add /testbed
# to sys.path[0] via the empty-string '' entry, which would shadow installed
# binary packages (e.g. numpy C-extensions) with uncompiled source.
cd /
set +e
if command -v xvfb-run &>/dev/null; then
    xvfb-run -a $PYTHON -m pytest "${TEST_FILES[@]}" -x -rA 2>&1
else
    $PYTHON -m pytest "${TEST_FILES[@]}" -x -rA 2>&1
fi
EXIT_CODE=$?
set -e

if [ $EXIT_CODE -eq 0 ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
"""

# ---------------------------------------------------------------------------
# tests/test_state.py — Harbor reward reader
# ---------------------------------------------------------------------------
_TEST_STATE_PY = """\
from pathlib import Path

def get_reward() -> float:
    reward_file = Path("/logs/verifier/reward.txt")
    if not reward_file.exists():
        return 0.0
    try:
        return float(reward_file.read_text().strip())
    except Exception:
        return 0.0
"""

# ---------------------------------------------------------------------------
# solution/solve.sh — oracle: just checkout the fixed commit
# ---------------------------------------------------------------------------
_SOLVE_SH_TEMPLATE = """\
#!/bin/bash
set -euo pipefail
# Oracle: checkout the commit that contains the fix.
cd /testbed && git checkout {base_commit}
"""

# instruction preamble
_SETUP_PREAMBLE = """\
## Environment Setup (complete this step first)

```bash
cd /testbed && git checkout {base_commit}
```

---

"""


# ---------------------------------------------------------------------------
# Repo name extraction
# ---------------------------------------------------------------------------

def _repo_name_from_metadata(metadata: dict) -> str | None:
    """Extract short repo name (e.g. 'pandas') from metadata."""
    repo_name = metadata.get("repo_name", "")
    short = repo_name.split("/")[-1].lower() if repo_name else ""
    if short in _REPO_GITHUB_URL:
        return short
    # fuzzy match
    return next(
        (k for k in _REPO_GITHUB_URL if short.startswith(k) or k.startswith(short)),
        None,
    )


# ---------------------------------------------------------------------------
# Tarball repack
# ---------------------------------------------------------------------------

def repack_task(task_binary: bytes) -> bytes:
    """
    Repack a task tarball with:
      - environment/Dockerfile  → shared per-repo Dockerfile (10 unique total)
      - tests/test.sh           → new test runner (xvfb-aware, reward 0/1)
      - tests/test_state.py     → Harbor reward reader
      - solution/solve.sh       → oracle (git checkout base_commit)
      - instruction.md          → prepend setup preamble
    """
    # Read all existing members
    existing: dict[str, bytes] = {}
    existing_dirs: set[str] = set()
    existing_modes: dict[str, int] = {}

    with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.isdir():
                existing_dirs.add(m.name)
            elif m.isfile():
                f = tf.extractfile(m)
                existing[m.name] = f.read() if f else b""
                existing_modes[m.name] = m.mode

    # Extract metadata
    metadata = None
    for candidate in [
        "environment/workspace/metadata.json",
        "setup_files/metadata.json",
    ]:
        if candidate in existing:
            metadata = json.loads(existing[candidate])
            break
    if metadata is None:
        raise ValueError("No metadata.json found in tarball")

    repo_name = _repo_name_from_metadata(metadata)
    if repo_name is None:
        raise ValueError(f"Unknown repo: {metadata.get('repo_name')!r}")

    base_commit = metadata.get("base_commit") or metadata.get("new_commit_hash")
    if not base_commit:
        raise ValueError("No base_commit in metadata")

    # Build replacement files
    replacements = {
        "environment/Dockerfile": _build_dockerfile(repo_name).encode(),
        "tests/test.sh":          _TEST_SH.encode(),
        "tests/test_state.py":    _TEST_STATE_PY.encode(),
        "solution/solve.sh":      _SOLVE_SH_TEMPLATE.format(base_commit=base_commit).encode(),
    }

    # instruction.md: prepend preamble if not already patched
    orig_instruction = existing.get("instruction.md", b"")
    preamble = _SETUP_PREAMBLE.format(base_commit=base_commit).encode()
    if b"## Environment Setup" not in orig_instruction:
        replacements["instruction.md"] = preamble + orig_instruction
    else:
        replacements["instruction.md"] = orig_instruction

    # setup_files/metadata.json: copy from environment/workspace/ if needed
    if "setup_files/metadata.json" not in existing and "environment/workspace/metadata.json" in existing:
        replacements["setup_files/metadata.json"] = existing["environment/workspace/metadata.json"]

    # Merge: start with existing files, apply replacements
    new_files: dict[str, bytes] = {}
    for name, data in existing.items():
        new_files[name] = replacements.get(name, data)
    for name, data in replacements.items():
        new_files[name] = data

    # Write new tarball
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf_out:
        # Write directory entries first
        dirs_written: set[str] = set()
        for path in sorted(new_files.keys()):
            parts = path.split("/")
            for i in range(1, len(parts)):
                dir_path = "/".join(parts[:i])
                if dir_path not in dirs_written:
                    dir_info = tarfile.TarInfo(name=dir_path)
                    dir_info.type = tarfile.DIRTYPE
                    dir_info.mode = 0o755
                    tf_out.addfile(dir_info)
                    dirs_written.add(dir_path)

        # Write files
        for path, data in sorted(new_files.items()):
            info = tarfile.TarInfo(name=path)
            info.size = len(data)
            info.mode = existing_modes.get(path, 0o755 if path.endswith(".sh") else 0o644)
            # Ensure shell scripts are executable
            if path.endswith(".sh"):
                info.mode = 0o755
            tf_out.addfile(info, io.BytesIO(data))

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch R2E-Gym tasks: 4,578 unique images → 10 (one per repo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sandbox-repo",
        default="DCAgent2/r2egym_sandboxes",
        help="HuggingFace dataset repo (e.g. DCAgent2/r2egym_sandboxes)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write patched tasks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N tasks (for testing)",
    )
    parser.add_argument(
        "--upload-to",
        default=None,
        help="HuggingFace repo to upload patched dataset",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token (default: HF_TOKEN env var)",
    )
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")

    try:
        from datasets import load_dataset, Dataset, Features, Value
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print(f"Loading dataset: {args.sandbox_repo}")
    ds = load_dataset(args.sandbox_repo, split="train", token=token)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Total tasks: {len(ds)}")
    print(f"Unique Dockerfiles will be: {len(_REPO_GITHUB_URL)} (one per repo)")

    stats = {"total": 0, "patched": 0, "error": 0}
    patched_rows = []

    for i, row in enumerate(ds):
        stats["total"] += 1
        path = row.get("path") or row.get("task_id") or f"r2egym-{i:04d}"
        task_binary = row.get("task_binary") or row.get("content")
        if task_binary is None:
            stats["error"] += 1
            print(f"  [{path}] No task_binary/content column")
            continue
        if isinstance(task_binary, list):
            task_binary = bytes(task_binary)

        try:
            new_binary = repack_task(task_binary)
        except Exception as e:
            stats["error"] += 1
            print(f"  [{path}] Error: {e}")
            continue

        patched_rows.append({"path": path, "task_binary": new_binary})
        stats["patched"] += 1

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(ds)} (patched={stats['patched']})")

    print(f"\nStats: {stats}")

    if not patched_rows:
        print("No tasks patched; exiting")
        return

    # Count unique Dockerfiles
    unique_dfs: set[str] = set()
    for row in patched_rows:
        try:
            with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:gz") as tf:
                try:
                    m = tf.getmember("environment/Dockerfile")
                    unique_dfs.add(tf.extractfile(m).read().decode())
                except KeyError:
                    pass
        except Exception:
            pass
    print(f"Unique Dockerfiles: {len(unique_dfs)} (expected: {len(_REPO_GITHUB_URL)})")

    features = Features({"path": Value("string"), "task_binary": Value("binary")})
    out_ds = Dataset.from_list(patched_rows, features=features)

    os.makedirs(args.output_dir, exist_ok=True)
    out_ds.save_to_disk(args.output_dir)
    print(f"Saved {len(patched_rows)} patched tasks to {args.output_dir}")

    if args.upload_to:
        print(f"Uploading to {args.upload_to}...")
        out_ds.push_to_hub(args.upload_to, token=token, private=False)
        print(f"Uploaded to https://huggingface.co/datasets/{args.upload_to}")


if __name__ == "__main__":
    main()
