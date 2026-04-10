#!/usr/bin/env python3
"""
Patch R2E-Gym tasks: 4,578 unique docker images → 10 (one per repo).

Merges two HuggingFace datasets:
  - DCAgent2/r2egym_sandboxes    → task tarballs (instruction.md, Dockerfile, metadata, test.sh)
  - R2E-Gym/R2E-Gym-Lite         → test file content (test_file_codes, test_file_names)

For each task:
  1. Matches sandbox tarball to R2E-Gym-Lite row by commit hash
  2. Injects test files (test_0.py, conftest.py, etc.) from R2E-Gym-Lite
  3. Replaces Dockerfile with one of 10 shared per-repo images (namanjain12 base)
  4. Adds solution/solve.sh (oracle: git checkout base_commit)
  5. Adds tests/test_state.py (Harbor reward reader)
  6. Rewrites tests/test.sh to run injected test files + calculate reward

The shared images are picked from representative namanjain12 images already
cached on Daytona. The agent performs 'git checkout <commit>' at runtime.

Usage (on cluster):
    # Test with 30 tasks first:
    HF_TOKEN=... python data/patchers/patch_r2egym_tasks.py \
        --output-dir /mnt/sda4T/home/jajee/r2egym_patched_v8_full \
        --limit 30

    # Full run + upload:
    HF_TOKEN=... python data/patchers/patch_r2egym_tasks.py \
        --output-dir /mnt/sda4T/home/jajee/r2egym_patched_v8_full \
        --upload-to SankalpKJ/r2egym-patched-v8-full
"""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import argparse
from pathlib import Path
from datasets import load_dataset, load_from_disk

# ---------------------------------------------------------------------------
# Per-repo configuration: 10 shared base images
# ---------------------------------------------------------------------------

_SHARED_BASE_IMAGES = {
    'aiohttp':    'namanjain12/aiohttp_final:f0d74880deec8fcd982bce639c93c5e130d41198',
    'coveragepy': 'namanjain12/coveragepy_final:c1bfa7352368b63f3a9b30c02f242408d07a7ab2',
    'datalad':    'namanjain12/datalad_final:f5e1d276ab51aefcf5e48e6f7bd9833b19ef7f90',
    'numpy':      'namanjain12/numpy_final:14445500bdf67600f926c6426bad55977441dca0',
    'orange3':    'namanjain12/orange3_final:2d9617bd0cb1f0ba61771258410ab8fae8e7e24d',
    'pandas':     'namanjain12/pandas_final:fadb72cf5ef8489e409d4d33625bd16a76fa7a42',
    'pillow':     'namanjain12/pillow_final:f644adbb05d615a9902ef3643714d5fe8049cea3',
    'pyramid':    'namanjain12/pyramid_final:fbbb20c7953370c86f999e865b1a9d682690eb70',
    'scrapy':     'namanjain12/scrapy_final:fbb411a805724fec50b786f369be79dc221c798e',
    'tornado':    'namanjain12/tornado_final:b5ec807edc83c8e7d1d12553d635ebe765e5c614',
}

# ---------------------------------------------------------------------------
# Dockerfile builder
# ---------------------------------------------------------------------------

# Common test dependencies pre-installed in ALL repo images.
# Installed at Docker build time so test.sh has zero pip overhead.
_COMMON_TEST_DEPS = [
    # Image/media
    'Pillow',
    # Test utilities
    'mock', 'unittest-mixins',
    # System/path utilities
    'appdirs', 'setuptools', 'importlib-metadata',
    # Data formats
    'defusedxml', 'openpyxl',
    # VCS
    'gitpython',
    # Web frameworks (pyramid tasks)
    'pyramid', 'plaster', 'plaster-pastedeploy',
    # Scrapy ecosystem
    'itemadapter',
    # Misc
    'boto3', 'trubar',
]

# Per-repo extra dependencies on top of the common set.
_EXTRA_DEPS: dict[str, list[str]] = {
    'orange3':    ['xlsxwriter', 'anyqt', 'serverfiles'],
    'aiohttp':    [],
    'coveragepy': [],
    'datalad':    [],
    'numpy':      [],
    'pandas':     [],
    'pillow':     [],
    'pyramid':    [],
    'scrapy':     [],
    'tornado':    [],
}


def _build_dockerfile(repo_name: str) -> str:
    base_image = _SHARED_BASE_IMAGES.get(repo_name)
    if not base_image:
        raise ValueError(f"No shared base image for repo: {repo_name}")

    common_pkgs = ' '.join(_COMMON_TEST_DEPS)
    extra_deps = _EXTRA_DEPS.get(repo_name, [])
    extra_lines = ""
    if extra_deps:
        pkgs = ' '.join(extra_deps)
        extra_lines = f"\nRUN pip install {pkgs} -q 2>/dev/null || true"

    # Determine which pip to use: venv if present, else system
    pip_cmd = """RUN if [ -f /testbed/.venv/bin/python ]; then \\
        /testbed/.venv/bin/python -m pip install {pkgs} -q 2>/dev/null || true; \\
    else \\
        pip3 install {pkgs} -q 2>/dev/null || true; \\
    fi""".format(pkgs=common_pkgs)

    return f"""\
FROM {base_image}

# Shared per-repo base image.
# Agent MUST perform: cd /testbed && git checkout <commit>
# to get to the correct task state.

# Pre-install common test dependencies (avoids slow pip installs at test time)
{pip_cmd}{extra_lines}

RUN mkdir -p /logs /r2e_tests /setup_files

# WORKDIR must come last so that _ensure_output_dir_in_dockerfile
# (in validate_and_upload_from_hf.py) inserts RUN mkdir /output AFTER
# WORKDIR and not before FROM (which would break the build).
WORKDIR /testbed
"""

# ---------------------------------------------------------------------------
# test.sh template  (BUG 4 + BUG 5 fixed)
#
# FIX 4: expected_output_json is a JSON string in metadata.json, not a dict.
#         Must call json.loads() on it before iterating.
# FIX 5: pytest nodeid format is "path/test_file.py::ClassName::test_method"
#         but expected_output_json keys are "ClassName.test_method".
#         Build a lookup that normalises both sides to "ClassName.test_method".
# FIX 6: pytest-json-report crashes on Python 3.7 (AttributeError: tbstyle).
#         Use pytest --tb=line -v and parse the verbose stdout instead.
#         This works on Python 3.6+ without any extra plugins.
# ---------------------------------------------------------------------------

_TEST_SH = """\
#!/bin/bash
# R2E-Gym test runner with reward calculation.
# 1. git checkout the fixed commit (done by solve.sh / agent).
# 2. Install repo requirements to match the exact commit environment.
# 3. Run pytest on injected test files.
# 4. Compare results to expected_output_json from metadata.
# 5. Write reward (0.0 or 1.0) to /logs/verifier/reward.txt.

set -x

mkdir -p /logs/verifier

# ---------------------------------------------------------------------------
# Resolve the Python / pip to use.
#
# Most repos: R2E-Gym builds inside a uv-managed venv at /testbed/.venv.
# numpy / scipy: no venv - uses the system Python directly.
# We detect which case we're in and set PYTHON/PIP accordingly.
# ---------------------------------------------------------------------------
if [ -f /testbed/.venv/bin/python ]; then
    PYTHON=/testbed/.venv/bin/python
elif [ -f /testbed/.venv/bin/python3 ]; then
    PYTHON=/testbed/.venv/bin/python3
else
    # No venv - use system Python (numpy, scipy, etc.)
    PYTHON=$(which python3 || which python)
fi
# Always use 'python -m pip' to avoid missing pip binary issues
PIP="$PYTHON -m pip"

cd /testbed

# ---------------------------------------------------------------------------
# CRITICAL for numpy/scipy/aiohttp: after git checkout, the Cython-compiled
# .so files are stale (compiled for the previous commit). We MUST rebuild
# them before running any tests.
# Touch all .pyx files first to force incremental rebuild - git checkout
# preserves timestamps so build_ext would skip unchanged files otherwise.
# ---------------------------------------------------------------------------
if [ -f /testbed/setup.py ]; then
    echo "=== Rebuilding Cython extensions ==="
    find /testbed -name "*.pyx" -exec touch {} \\;
    $PYTHON setup.py build_ext --inplace 2>&1 | tail -20 || true
elif [ -f /testbed/pyproject.toml ]; then
    $PIP install -e . --no-build-isolation -q 2>&1 | tail -5 || true
fi

# Reinstall the repo itself in editable mode (picks up new entry points).
# Common deps are pre-installed in the Docker image - this is fast.
$PIP install -e . -q 2>&1 | tail -5 || true

# 1. Run tests.
#    PYTHONPATH includes:
#      /tests        - so 'helper' and other test-local modules are importable
#      /testbed/tests - so 'tests' package imports work (pandas, etc.)
#      /testbed      - so the repo source is on the path
#
# NOTE: Do NOT use --import-mode=importlib (breaks setuptools/pkg_resources)
# NOTE: Do NOT use --rootdir=/testbed (overrides conftest.py discovery)
# For pyramid relative imports, we copy test files into the package tree
# and run pytest from /testbed so the package structure is preserved.
if ls /testbed/tests/test_*.py 2>/dev/null | head -1 | grep -q .; then
    # Repo has its own tests/ dir - copy injected tests there and run from /testbed
    cp /tests/test_*.py /testbed/tests/ 2>/dev/null || true
    PYTHONPATH=/tests:/testbed/tests:/testbed:$PYTHONPATH $PYTHON -m pytest /testbed/tests/test_*.py \\
        --ignore=/testbed/tests/test_state.py \\
        --ignore-glob=/testbed/tests/test_OW*.py \\
        -v --tb=short \\
        -p no:qt \\
        2>&1 | tee /logs/pytest_output.txt || true
else
    PYTHONPATH=/tests:/testbed/tests:/testbed:$PYTHONPATH $PYTHON -m pytest /tests/test_*.py \\
        --ignore=/tests/test_state.py \\
        --ignore-glob=/tests/test_OW*.py \\
        -v --tb=short \\
        -p no:qt \\
        2>&1 | tee /logs/pytest_output.txt || true
fi

# 2. Calculate reward
$PYTHON - <<'PYEOF'
import json, re, sys
from pathlib import Path

def calculate():
    try:
        # Load expected results from metadata
        meta_path = Path('/setup_files/metadata.json')
        if not meta_path.exists():
            print('ERROR: /setup_files/metadata.json not found')
            return 0.0
        meta = json.loads(meta_path.read_text())

        expected_raw = meta.get('expected_output_json', {})
        if isinstance(expected_raw, str):
            expected = json.loads(expected_raw)
        else:
            expected = expected_raw

        if not expected:
            print('WARNING: expected_output_json is empty')
            return 0.0

        output_path = Path('/logs/pytest_output.txt')
        if not output_path.exists():
            print('ERROR: /logs/pytest_output.txt not found')
            return 0.0

        actual_results = {}
        line_re = re.compile(r'^(\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)', re.MULTILINE)
        for m in line_re.finditer(output_path.read_text()):
            nodeid = m.group(1)
            outcome = m.group(2)
            parts = nodeid.split('::')
            key = '.'.join(parts[1:])
            actual_results[key] = outcome

        print('Actual results:', actual_results)
        print('Expected:', expected)

        # Compare: all expected tests must match.
        # Treat ERROR and FAILED as equivalent (both mean non-passing).
        def normalize(s):
            if s in ('ERROR', 'FAILED'):
                return 'FAILED'
            return s

        mismatches = []
        for test_name, expected_status in expected.items():
            actual_status = actual_results.get(test_name)
            if normalize(actual_status) != normalize(expected_status):
                mismatches.append(
                    '  {}: expected={} actual={}'.format(test_name, expected_status, actual_status)
                )

        if mismatches:
            print('MISMATCHES:')
            for m in mismatches:
                print(m)
            return 0.0

        return 1.0

    except Exception as e:
        import traceback
        print('Error calculating reward: {}'.format(e))
        traceback.print_exc()
        return 0.0

reward = calculate()
Path('/logs/verifier/reward.txt').write_text(str(reward))
print('REWARD: {}'.format(reward))
PYEOF
"""

# ---------------------------------------------------------------------------
# tests/test_state.py - Harbor reward reader
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
    "orange3":    "orange3",
    "pillow":     "pillow",
    "pil":        "pillow",
    "coverage":   "coveragepy",
    "coveragepy": "coveragepy",
    "coverage.py":"coveragepy",
}

ALL_REPOS = set(_SHARED_BASE_IMAGES.keys())

def _repo_name_from_string(raw: str) -> str | None:
    """Normalize a repo name string to one of our 10 canonical names."""
    short = raw.split("/")[-1].lower().strip() if raw else ""
    if short in ALL_REPOS:
        return short
    if short in _REPO_ALIASES:
        return _REPO_ALIASES[short]
    # fuzzy match
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
    # Read existing tarball
    existing: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:gz") as tf:
        for m in tf.getmembers():
            if m.isfile():
                f = tf.extractfile(m)
                existing[m.name] = f.read() if f else b""

    # Extract metadata
    metadata = None
    for candidate in ["environment/workspace/metadata.json", "setup_files/metadata.json"]:
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

    # 1. Dockerfile — use shared per-repo base image
    replacements["environment/Dockerfile"] = _build_dockerfile(repo_name).encode()

    # 2. test.sh (with reward logic fixes)
    replacements["tests/test.sh"] = _TEST_SH.encode()

    # 3. test_state.py
    replacements["tests/test_state.py"] = _TEST_STATE_PY.encode()

    # 4. Inject test files from R2E-Gym-Lite
    for fname, code in zip(test_file_names, test_file_codes):
        if not fname.startswith("tests/"):
            fname = f"tests/{fname}"
        replacements[fname] = code.encode()

    # 5. setup_files/metadata.json
    meta_bytes = existing.get(
        "environment/workspace/metadata.json",
        existing.get("setup_files/metadata.json", b"{}"),
    )
    replacements["setup_files/metadata.json"] = meta_bytes

    # 6. solution/solve.sh
    replacements["solution/solve.sh"] = _SOLVE_SH_TEMPLATE.format(
        base_commit=base_commit
    ).encode()

    # 7. instruction.md — prepend setup preamble if not already present
    orig_instruction = existing.get("instruction.md", b"")
    if b"## Environment Setup" not in orig_instruction:
        preamble = _SETUP_PREAMBLE.format(base_commit=base_commit).encode()
        replacements["instruction.md"] = preamble + orig_instruction

    # Merge: existing files take replacements, then add any new files
    new_files: dict[str, bytes] = {}
    for name, data in existing.items():
        new_files[name] = replacements.pop(name, data)
    for name, data in replacements.items():
        new_files[name] = data

    # Write new tarball
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf_out:
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
    parser = argparse.ArgumentParser(description="Patch R2E-Gym tasks to use 10 shared base images")
    parser.add_argument("--output-dir", required=True, help="Local directory to save patched dataset")
    parser.add_argument("--limit", type=int, default=None, help="Only patch first N tasks (for testing)")
    parser.add_argument("--upload-to", type=str, default=None, help="HuggingFace repo to upload to")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")

    print("Loading R2E-Gym-Lite (test files)...")
    lite_ds = load_dataset("R2E-Gym/R2E-Gym-Lite", split="train", token=hf_token)
    # Key by commit_hash — this is what sandbox metadata.base_commit matches
    commit_to_tests = {row["commit_hash"]: row for row in lite_ds}
    print(f"Loaded {len(commit_to_tests)} test sets from R2E-Gym-Lite.")

    print("Loading DCAgent2/r2egym_sandboxes (tarballs)...")
    sandboxes_ds = load_dataset("DCAgent2/r2egym_sandboxes", split="train", token=hf_token)
    if args.limit:
        sandboxes_ds = sandboxes_ds.select(range(args.limit))
    print(f"Processing {len(sandboxes_ds)} sandbox tasks...")

    patched_rows = []
    skipped = 0
    errors = 0

    for i, row in enumerate(sandboxes_ds):
        try:
            task_binary = bytes(row["task_binary"])

            # Extract commit hash from tarball metadata
            with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:gz") as tf:
                meta = None
                for p in ["environment/workspace/metadata.json", "setup_files/metadata.json"]:
                    try:
                        meta = json.loads(tf.extractfile(p).read())
                        break
                    except Exception:
                        pass

            if not meta:
                print(f"[{i}] Skip: no metadata.json in tarball")
                skipped += 1
                continue

            commit = meta.get("base_commit") or meta.get("new_commit_hash")
            if not commit:
                print(f"[{i}] Skip: no commit hash in metadata")
                skipped += 1
                continue

            if commit not in commit_to_tests:
                print(f"[{i}] Skip: commit {commit[:8]} not in R2E-Gym-Lite")
                skipped += 1
                continue

            lite_row = commit_to_tests[commit]

            # BUG 1 FIX: test_file_names/codes are nested inside execution_result_content
            # (a JSON string), not top-level columns. Parse safely.
            erc_raw = lite_row.get("execution_result_content")
            if not erc_raw:
                print(f"[{i}] Skip: execution_result_content is empty for {commit[:8]}")
                skipped += 1
                continue

            try:
                erc = json.loads(erc_raw) if isinstance(erc_raw, str) else erc_raw
            except json.JSONDecodeError:
                print(f"[{i}] Skip: execution_result_content is not valid JSON for {commit[:8]}")
                skipped += 1
                continue

            test_file_names = erc.get("test_file_names", [])
            test_file_codes = erc.get("test_file_codes", [])

            if not test_file_names or not test_file_codes:
                print(f"[{i}] Skip: no test files in execution_result_content for {commit[:8]}")
                skipped += 1
                continue

            # Skip GUI widget tasks: Orange3 OW* tests require a Qt display and
            # crash pytest with SIGABRT in headless containers.
            #
            # Filter strategy: check the expected_output_json test class names.
            # Orange3 widget tests use class names like TestOWRank, TestOWDataSets,
            # WidgetTest, etc. These crash with "Fatal Python error: Aborted" in
            # headless containers regardless of the test file name (which is often
            # just test_1.py, not test_OW*.py).
            #
            # Also check test file names as a secondary filter.
            expected_raw = lite_row.get('expected_output_json', '{}')
            if isinstance(expected_raw, str):
                expected_dict = json.loads(expected_raw) if expected_raw else {}
            else:
                expected_dict = expected_raw or {}

            _GUI_PREFIXES = ("OW", "ow", "TestOW", "TestWidget", "WidgetTest")

            # GUI if ANY expected test class starts with a GUI prefix
            gui_by_class = any(
                k.split(".")[0].startswith(_GUI_PREFIXES)
                for k in expected_dict.keys()
            )
            # GUI if ALL test file names are OW* (original file-name filter)
            gui_by_file = bool(test_file_names) and all(
                any(part.startswith(("OW", "ow", "test_OW", "test_ow"))
                    for part in fname.replace("\\", "/").split("/"))
                for fname in test_file_names
            )
            # GUI if ANY test file imports orangewidget or Orange.widgets
            # (these crash pytest with SIGABRT even in non-OW* named files)
            _GUI_IMPORTS = ("orangewidget", "Orange.widgets", "AnyQt", "PyQt5", "PyQt4")
            gui_by_import = any(
                any(imp in code for imp in _GUI_IMPORTS)
                for code in test_file_codes
            )
            if gui_by_class or gui_by_file or gui_by_import:
                print(f"[{i}] Skip: GUI/widget task for {commit[:8]}")
                skipped += 1
                continue

            new_binary = repack_task(task_binary, test_file_names, test_file_codes)
            row["task_binary"] = new_binary
            patched_rows.append(row)
            print(f"[{i}] Patched: {meta.get('repo_name')} @ {commit[:8]} ({len(test_file_names)} test files)")

        except Exception as e:
            print(f"[{i}] Error: {e}")
            errors += 1

    print(f"\nDone: {len(patched_rows)} patched, {skipped} skipped, {errors} errors")

    from datasets import Dataset
    out_ds = Dataset.from_list(patched_rows)
    out_ds.save_to_disk(args.output_dir)
    print(f"Saved {len(out_ds)} tasks to {args.output_dir}")

    if args.upload_to:
        out_ds.push_to_hub(args.upload_to, token=hf_token)
        print(f"Uploaded to https://huggingface.co/datasets/{args.upload_to}")

if __name__ == "__main__":
    main()
