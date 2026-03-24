#!/usr/bin/env python3
"""
Patch R2E-Gym tasks: 4,578 unique docker images → 10 (one per repo).

Merges two HuggingFace datasets:
  - DCAgent2/r2egym_sandboxes    → task tarballs (instruction.md, Dockerfile, metadata, test.sh)
  - R2E-Gym/R2E-Gym-Lite         → test file content (test_file_codes, test_file_names)

For each task:
  1. Matches sandbox tarball to R2E-Gym-Lite row by commit hash
  2. Injects test files (test_0.py, conftest.py, etc.) from R2E-Gym-Lite
  3. Replaces Dockerfile with one of 10 shared per-repo images
  4. Adds solution/solve.sh (oracle: git checkout base_commit)
  5. Adds tests/test_state.py (Harbor reward reader)
  6. Rewrites tests/test.sh to run injected test files + calculate reward

Compiled repos (pandas, numpy, pillow, aiohttp, orange3) use custom pre-built
ghcr.io/open-thoughts/r2egym-<repo>:latest images to avoid C extension build
timeouts. Pure-Python repos use python:X.Y-bookworm directly.

Usage (on cluster):
    # First build and push the 5 compiled-repo images:
    bash data/patchers/r2egym_base_images/build_and_push.sh

    # Then run the patcher (test with 10 tasks first):
    python data/patchers/patch_r2egym_tasks.py \\
        --output-dir /mnt/sda4T/home/jajee/r2egym_patched \\
        --limit 10

    # Full run + upload:
    python data/patchers/patch_r2egym_tasks.py \\
        --output-dir /mnt/sda4T/home/jajee/r2egym_patched \\
        --upload-to SankalpKJ/r2egym-patched
"""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Per-repo configuration
# ---------------------------------------------------------------------------

# Repos that need custom pre-built ghcr.io images (compiled C extensions)
_COMPILED_REPOS = {"pandas", "numpy", "pillow", "aiohttp", "orange3"}

# Repos that can use python:X.Y-bookworm directly (pure Python, fast install)
_PURE_PYTHON_REPOS = {"tornado", "scrapy", "pyramid", "datalad", "coveragepy"}

_GHCR_REGISTRY = "ghcr.io/open-thoughts"

_REPO_PYTHON_VERSION: dict[str, str] = {
    "pandas":     "3.11",
    "numpy":      "3.11",
    "pillow":     "3.11",
    "orange3":    "3.10",
    "aiohttp":    "3.11",
    "tornado":    "3.11",
    "scrapy":     "3.11",
    "pyramid":    "3.11",
    "datalad":    "3.11",
    "coveragepy": "3.11",
}

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

_REPO_EXTRA_APT: dict[str, str] = {
    "pandas":     "gfortran libopenblas-dev liblapack-dev pkg-config",
    "numpy":      "gfortran libopenblas-dev liblapack-dev pkg-config",
    "pillow":     "libjpeg-dev zlib1g-dev libpng-dev libtiff-dev libwebp-dev libfreetype6-dev liblcms2-dev libopenjp2-7-dev",
    "orange3":    "libxml2-dev libxslt1-dev libgl1-mesa-glx libglib2.0-0 xvfb libxkbcommon-x11-0 libdbus-1-3 libegl1 libxcb-xinerama0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-cursor0",
    "aiohttp":    "",
    "tornado":    "",
    "scrapy":     "libxml2-dev libxslt1-dev",
    "pyramid":    "",
    "datalad":    "git-annex",
    "coveragepy": "",
}

_REPO_INSTALL_CMD: dict[str, str] = {
    "tornado":    "pip install -e . 2>/dev/null || pip install tornado; pip install pytest",
    "scrapy":     "pip install -e '.[tests]' 2>/dev/null || pip install -e . 2>/dev/null || pip install Scrapy pytest",
    "pyramid":    "pip install -e '.[testing]' 2>/dev/null || pip install -e . 2>/dev/null || pip install pyramid pytest",
    "datalad":    "pip install -e '.[devel]' 2>/dev/null || pip install -e . 2>/dev/null || pip install datalad pytest",
    "coveragepy": "pip install -e '.[dev]' 2>/dev/null || pip install -e . 2>/dev/null || pip install coverage pytest; pip install unittest-mixins mock 2>/dev/null || true",
}


# ---------------------------------------------------------------------------
# Dockerfile builders
# ---------------------------------------------------------------------------

def _build_dockerfile_compiled(repo_name: str) -> str:
    """Dockerfile for compiled repos: use pre-built ghcr.io image."""
    return f"""\
FROM {_GHCR_REGISTRY}/r2egym-{repo_name}:latest

# Pre-built image already has:
#   - repo cloned at /testbed (HEAD)
#   - all deps + C extensions compiled
#   - pytest installed
# Agent only needs: cd /testbed && git checkout <commit>

RUN mkdir -p /logs /r2e_tests /setup_files
"""


def _build_dockerfile_pure(repo_name: str) -> str:
    """Dockerfile for pure-Python repos: build from python:X.Y-bookworm."""
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

RUN git clone {github_url} /testbed
WORKDIR /testbed
RUN {install_cmd}

RUN mkdir -p /logs /r2e_tests /setup_files
"""


def _build_dockerfile(repo_name: str) -> str:
    if repo_name in _COMPILED_REPOS:
        return _build_dockerfile_compiled(repo_name)
    return _build_dockerfile_pure(repo_name)


# ---------------------------------------------------------------------------
# test.sh template
# ---------------------------------------------------------------------------
# This test.sh:
#   1. Finds test_*.py and conftest.py files in /tests/ (Harbor mounts them)
#   2. Runs pytest on them
#   3. Parses output and compares to expected_output_json from metadata
#   4. Writes reward (0.0 or 1.0) to /logs/verifier/reward.txt

_TEST_SH = """\
#!/bin/bash
set -e
mkdir -p /logs/verifier

# Use system Python (no .venv in shared images)
PYTHON=python3

# Ensure pytest is available
$PYTHON -m pytest --version &>/dev/null || $PYTHON -m pip install pytest -q

# Harbor mounts the task's tests/ directory at /tests/ in the container.
TEST_FILES=()
for f in /tests/test_*.py; do
    [ -f "$f" ] && TEST_FILES+=("$f")
done
if [ ${#TEST_FILES[@]} -eq 0 ]; then
    echo "ERROR: no test_*.py files found in /tests/" >&2
    echo "0" > /logs/verifier/reward.txt
    exit 0
fi

# Copy conftest.py if present (pytest needs it next to test files or in /testbed)
if [ -f /tests/conftest.py ]; then
    cp /tests/conftest.py /testbed/conftest_r2e.py 2>/dev/null || true
fi

# Clean up stale bytecode
find /testbed -name '*.pyc' -delete 2>/dev/null || true
find /testbed -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# Run pytest from /testbed so imports work correctly.
# Use xvfb-run if available (for Qt-dependent repos like orange3).
cd /testbed
TEST_OUTPUT_FILE="/tmp/test_output.txt"

set +e
if command -v xvfb-run &>/dev/null; then
    xvfb-run -a $PYTHON -m pytest "${TEST_FILES[@]}" -rA 2>&1 | tee "$TEST_OUTPUT_FILE"
else
    $PYTHON -m pytest "${TEST_FILES[@]}" -rA 2>&1 | tee "$TEST_OUTPUT_FILE"
fi
PYTEST_EXIT=$?
set -e

# ---- Calculate reward by comparing test results to expected_output_json ----
cat > /tmp/calculate_reward.py << 'REWARD_SCRIPT_EOF'
#!/usr/bin/env python3
import re, json, sys
from pathlib import Path

def parse_log_pytest(log):
    if not log or "short test summary info" not in log:
        return {}
    status_map = {}
    summary = log.split("short test summary info")[1].strip()
    for line in summary.split("\\n"):
        line = line.strip()
        if "PASSED" in line:
            parts = line.split("::")
            if len(parts) > 1:
                test_name = ".".join(parts[1:])
                status_map[test_name] = "PASSED"
        elif "FAILED" in line:
            parts = line.split("::")
            if len(parts) > 1:
                test_name = ".".join(parts[1:]).split(" - ")[0]
                status_map[test_name] = "FAILED"
        elif "ERROR" in line:
            parts = line.split("::")
            if len(parts) > 1:
                test_name = ".".join(parts[1:]).split(" - ")[0]
            else:
                test_name = line
            status_map[test_name] = "ERROR"
    return status_map

def decolor(d):
    strip = lambda k: re.sub(r"\\u001b\\[\\d+m", "", k)
    return {strip(k): v for k, v in d.items()}

def get_reward(parsed, expected_json):
    p = {k.split(" - ")[0]: v for k, v in decolor(parsed).items()}
    e = {k.split(" - ")[0]: v for k, v in decolor(json.loads(expected_json)).items()}
    p = dict(sorted(p.items()))
    e = dict(sorted(e.items()))
    if len(p) != len(e):
        return 0.0
    for k in p:
        if not k:
            continue
        if k not in e or p[k] != e[k]:
            return 0.0
    return 1.0

test_output = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else ""
meta = None
for candidate in ["/setup_files/metadata.json", "/workspace/metadata.json",
                   "/tests/metadata.json"]:
    p = Path(candidate)
    if p.exists():
        meta = json.loads(p.read_text())
        break
if meta is None:
    Path("/logs/verifier/reward.txt").write_text("0")
    print("Reward: 0 (no metadata found)")
    sys.exit(0)

expected = meta.get("expected_output_json", "{}")
parsed = parse_log_pytest(test_output)
reward = get_reward(parsed, expected)
Path("/logs/verifier").mkdir(parents=True, exist_ok=True)
Path("/logs/verifier/reward.txt").write_text(str(reward))
print(f"Reward: {reward}")
REWARD_SCRIPT_EOF

$PYTHON /tmp/calculate_reward.py "$TEST_OUTPUT_FILE"

if [ ! -f /logs/verifier/reward.txt ]; then
    echo "0" > /logs/verifier/reward.txt
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
# solution/solve.sh — oracle: git checkout the fixed commit
# ---------------------------------------------------------------------------

_SOLVE_SH_TEMPLATE = """\
#!/bin/bash
set -euo pipefail
# Oracle: checkout the commit that contains the fix.
cd /testbed && git checkout {base_commit}
"""


# ---------------------------------------------------------------------------
# instruction.md preamble
# ---------------------------------------------------------------------------

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

_REPO_ALIASES = {
    "orange3": "orange3",
    "pillow": "pillow",
    "pil": "pillow",
    "coverage": "coveragepy",
    "coveragepy": "coveragepy",
    "coverage.py": "coveragepy",
}

ALL_REPOS = set(_REPO_GITHUB_URL.keys())


def _repo_name_from_string(raw: str) -> str | None:
    """Normalize a repo name string to one of our 10 canonical names."""
    short = raw.split("/")[-1].lower().strip() if raw else ""
    if short in ALL_REPOS:
        return short
    if short in _REPO_ALIASES:
        return _REPO_ALIASES[short]
    # fuzzy
    return next(
        (k for k in ALL_REPOS if short.startswith(k) or k.startswith(short)),
        None,
    )


# ---------------------------------------------------------------------------
# Tarball repack
# ---------------------------------------------------------------------------

def repack_task(
    task_binary: bytes,
    test_file_names: list[str],
    test_file_codes: list[str],
) -> bytes:
    """
    Repack a task tarball with:
      - environment/Dockerfile  → shared per-repo Dockerfile (10 unique total)
      - tests/test.sh           → new test runner with reward calculation
      - tests/test_state.py     → Harbor reward reader
      - tests/test_N.py         → injected test files from R2E-Gym-Lite
      - tests/conftest.py       → injected conftest if present
      - setup_files/metadata.json → copy of metadata for reward calculation
      - solution/solve.sh       → oracle (git checkout base_commit)
      - instruction.md          → prepend setup preamble
    """
    # Read existing tarball
    existing: dict[str, bytes] = {}
    existing_modes: dict[str, int] = {}

    with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile():
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

    repo_name = _repo_name_from_string(metadata.get("repo_name", ""))
    if repo_name is None:
        raise ValueError(f"Unknown repo: {metadata.get('repo_name')!r}")

    base_commit = metadata.get("base_commit") or metadata.get("new_commit_hash")
    if not base_commit:
        raise ValueError("No base_commit in metadata")

    # Build replacement/new files
    replacements: dict[str, bytes] = {}

    # 1. Dockerfile
    replacements["environment/Dockerfile"] = _build_dockerfile(repo_name).encode()

    # 2. test.sh
    replacements["tests/test.sh"] = _TEST_SH.encode()

    # 3. test_state.py
    replacements["tests/test_state.py"] = _TEST_STATE_PY.encode()

    # 4. Inject test files from R2E-Gym-Lite
    for fname, code in zip(test_file_names, test_file_codes):
        # Ensure test files are in tests/ directory
        if not fname.startswith("tests/"):
            fname = f"tests/{fname}"
        replacements[fname] = code.encode()

    # 5. setup_files/metadata.json (for reward calculation inside container)
    meta_bytes = existing.get(
        "environment/workspace/metadata.json",
        existing.get("setup_files/metadata.json", b"{}"),
    )
    replacements["setup_files/metadata.json"] = meta_bytes

    # 6. solution/solve.sh
    replacements["solution/solve.sh"] = _SOLVE_SH_TEMPLATE.format(
        base_commit=base_commit
    ).encode()

    # 7. instruction.md — prepend setup preamble
    orig_instruction = existing.get("instruction.md", b"")
    if b"## Environment Setup" not in orig_instruction:
        preamble = _SETUP_PREAMBLE.format(base_commit=base_commit).encode()
        replacements["instruction.md"] = preamble + orig_instruction

    # Merge: existing + replacements
    new_files: dict[str, bytes] = {}
    for name, data in existing.items():
        new_files[name] = replacements.pop(name, data)
    for name, data in replacements.items():
        new_files[name] = data

    # Write new tarball
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf_out:
        # Directory entries
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

        # File entries
        for path, data in sorted(new_files.items()):
            info = tarfile.TarInfo(name=path)
            info.size = len(data)
            info.mode = 0o755 if path.endswith(".sh") else 0o644
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
        help="HF dataset with task tarballs (default: DCAgent2/r2egym_sandboxes)",
    )
    parser.add_argument(
        "--lite-repo",
        default="R2E-Gym/R2E-Gym-Lite",
        help="HF dataset with test file content (default: R2E-Gym/R2E-Gym-Lite)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save patched dataset",
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
        help="HF repo to upload patched dataset (e.g. SankalpKJ/r2egym-patched)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token (default: HF_TOKEN env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show stats without writing files",
    )
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")

    try:
        from datasets import load_dataset, Dataset, Features, Value
    except ImportError:
        print("ERROR: pip install datasets huggingface_hub")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1: Load R2E-Gym-Lite and build commit→test_files index
    # ------------------------------------------------------------------
    print(f"Loading {args.lite_repo} (test file content)...")
    lite_ds = load_dataset(args.lite_repo, split="train", token=token)
    print(f"  Loaded {len(lite_ds)} rows from R2E-Gym-Lite")

    # Index by commit_hash for fast lookup
    lite_index: dict[str, dict] = {}
    for row in lite_ds:
        commit = row.get("commit_hash", "")
        if commit:
            erc = row.get("execution_result_content", "")
            if isinstance(erc, str):
                try:
                    erc = json.loads(erc)
                except json.JSONDecodeError:
                    erc = {}
            lite_index[commit] = {
                "test_file_names": erc.get("test_file_names", []),
                "test_file_codes": erc.get("test_file_codes", []),
                "repo_name": row.get("repo_name", ""),
            }
    print(f"  Indexed {len(lite_index)} unique commits with test files")

    # ------------------------------------------------------------------
    # Step 2: Load sandbox tarballs
    # ------------------------------------------------------------------
    print(f"\nLoading {args.sandbox_repo} (task tarballs)...")
    if args.limit:
        sandbox_ds = load_dataset(
            args.sandbox_repo, split=f"train[:{args.limit}]", token=token
        )
    else:
        sandbox_ds = load_dataset(args.sandbox_repo, split="train", token=token)
    print(f"  Loaded {len(sandbox_ds)} tasks")

    # ------------------------------------------------------------------
    # Step 3: Patch each task
    # ------------------------------------------------------------------
    print(f"\nPatching tasks...")
    stats = {
        "total": 0,
        "patched": 0,
        "no_match": 0,
        "no_tests": 0,
        "error": 0,
    }
    repo_stats: dict[str, dict[str, int]] = {}
    patched_rows = []

    for i, row in enumerate(sandbox_ds):
        stats["total"] += 1
        path = row.get("path", f"r2egym-{i:04d}")
        task_binary = row.get("task_binary")
        if isinstance(task_binary, list):
            task_binary = bytes(task_binary)
        if task_binary is None:
            stats["error"] += 1
            continue

        # Extract commit hash from metadata
        try:
            with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:gz") as tf:
                m = tf.getmember("environment/workspace/metadata.json")
                meta = json.loads(tf.extractfile(m).read())
        except Exception as e:
            stats["error"] += 1
            if stats["error"] <= 5:
                print(f"  ERROR [{path}]: {e}")
            continue

        base_commit = meta.get("base_commit", "")
        repo_raw = meta.get("repo_name", "unknown")
        repo = _repo_name_from_string(repo_raw) or repo_raw

        if repo not in repo_stats:
            repo_stats[repo] = {
                "total": 0, "patched": 0, "no_match": 0,
                "no_tests": 0, "error": 0,
            }
        repo_stats[repo]["total"] += 1

        # Look up test files from R2E-Gym-Lite
        lite_row = lite_index.get(base_commit)
        if lite_row is None:
            stats["no_match"] += 1
            repo_stats[repo]["no_match"] += 1
            if stats["no_match"] <= 5:
                print(f"  NO MATCH [{path}]: commit {base_commit[:12]} not in Lite")
            continue

        test_names = lite_row["test_file_names"]
        test_codes = lite_row["test_file_codes"]
        if not test_names or not test_codes:
            stats["no_tests"] += 1
            repo_stats[repo]["no_tests"] += 1
            continue

        # Repack
        if args.dry_run:
            stats["patched"] += 1
            repo_stats[repo]["patched"] += 1
            continue

        try:
            new_binary = repack_task(task_binary, test_names, test_codes)
        except Exception as e:
            stats["error"] += 1
            repo_stats[repo]["error"] += 1
            if stats["error"] <= 10:
                print(f"  ERROR [{path}]: {e}")
            continue

        patched_rows.append({"path": path, "task_binary": new_binary})
        stats["patched"] += 1
        repo_stats[repo]["patched"] += 1

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(sandbox_ds)} (patched={stats['patched']})")

    # ------------------------------------------------------------------
    # Step 4: Report stats
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"STATS")
    print(f"{'='*60}")
    print(f"  Total tasks:     {stats['total']}")
    print(f"  Patched:         {stats['patched']}")
    print(f"  No Lite match:   {stats['no_match']}")
    print(f"  No test files:   {stats['no_tests']}")
    print(f"  Errors:          {stats['error']}")
    print()
    print(f"Per-repo breakdown:")
    print(f"  {'Repo':<15} {'Total':>6} {'Patched':>8} {'NoMatch':>8} {'NoTests':>8} {'Error':>6}")
    for repo in sorted(repo_stats.keys()):
        rs = repo_stats[repo]
        print(
            f"  {repo:<15} {rs['total']:>6} {rs['patched']:>8} "
            f"{rs['no_match']:>8} {rs['no_tests']:>8} {rs['error']:>6}"
        )

    if args.dry_run:
        print("\n(Dry run — no files written)")
        return

    if not patched_rows:
        print("\nNo tasks patched; exiting")
        return

    # Count unique Dockerfiles
    unique_dfs: set[str] = set()
    for row in patched_rows:
        try:
            with tarfile.open(
                fileobj=io.BytesIO(row["task_binary"]), mode="r:gz"
            ) as tf:
                m = tf.getmember("environment/Dockerfile")
                unique_dfs.add(tf.extractfile(m).read().decode())
        except Exception:
            pass
    print(f"\nUnique Dockerfiles: {len(unique_dfs)} (target: {len(ALL_REPOS)})")

    # ------------------------------------------------------------------
    # Step 5: Save
    # ------------------------------------------------------------------
    features = Features({"path": Value("string"), "task_binary": Value("binary")})
    out_ds = Dataset.from_list(patched_rows, features=features)

    os.makedirs(args.output_dir, exist_ok=True)
    out_ds.save_to_disk(args.output_dir)
    print(f"\nSaved {len(patched_rows)} patched tasks to {args.output_dir}")

    # ------------------------------------------------------------------
    # Step 6: Upload (optional)
    # ------------------------------------------------------------------
    if args.upload_to:
        print(f"\nUploading to {args.upload_to}...")
        out_ds.push_to_hub(args.upload_to, token=token, private=False)
        print(f"Uploaded to https://huggingface.co/datasets/{args.upload_to}")


if __name__ == "__main__":
    main()
