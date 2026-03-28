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
    # Run the patcher (test with 30 tasks first):
    python data/patchers/patch_r2egym_tasks.py \
        --output-dir /mnt/sda4T/home/jajee/r2egym_patched_v7 \
        --limit 30

    # Full run + upload:
    python data/patchers/patch_r2egym_tasks.py \
        --output-dir /mnt/sda4T/home/jajee/r2egym_patched_v7 \
        --upload-to SankalpKJ/r2egym-patched-v7
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
    'aiohttp': 'namanjain12/aiohttp_final:f0d74880deec8fcd982bce639c93c5e130d41198',
    'coveragepy': 'namanjain12/coveragepy_final:c1bfa7352368b63f3a9b30c02f242408d07a7ab2',
    'datalad': 'namanjain12/datalad_final:f5e1d276ab51aefcf5e48e6f7bd9833b19ef7f90',
    'numpy': 'namanjain12/numpy_final:14445500bdf67600f926c6426bad55977441dca0',
    'orange3': 'namanjain12/orange3_final:2d9617bd0cb1f0ba61771258410ab8fae8e7e24d',
    'pandas': 'namanjain12/pandas_final:fadb72cf5ef8489e409d4d33625bd16a76fa7a42',
    'pillow': 'namanjain12/pillow_final:f644adbb05d615a9902ef3643714d5fe8049cea3',
    'pyramid': 'namanjain12/pyramid_final:fbbb20c7953370c86f999e865b1a9d682690eb70',
    'scrapy': 'namanjain12/scrapy_final:fbb411a805724fec50b786f369be79dc221c798e',
    'tornado': 'namanjain12/tornado_final:b5ec807edc83c8e7d1d12553d635ebe765e5c614',
}

# ---------------------------------------------------------------------------
# Dockerfile builder
# ---------------------------------------------------------------------------

def _build_dockerfile(repo_name: str) -> str:
    base_image = _SHARED_BASE_IMAGES.get(repo_name)
    if not base_image:
        raise ValueError(f"No shared base image for repo: {repo_name}")
    
    return f"""\
FROM {base_image}

# Shared per-repo base image. 
# Agent MUST perform: cd /testbed && git checkout <commit>
# to get to the correct task state.

RUN mkdir -p /logs /r2e_tests /setup_files
"""

# ---------------------------------------------------------------------------
# test.sh template
# ---------------------------------------------------------------------------

_TEST_SH = """\
#!/bin/bash
# R2E-Gym test runner with reward calculation.
# 1. Run pytest on injected test files.
# 2. Compare results to expected_output_json.
# 3. Write reward (0.0 or 1.0) to /logs/verifier/reward.txt.

set -x

# Ensure log directory exists
mkdir -p /logs/verifier

# 1. Run tests
# We use -p no:terminal to keep output clean for parsing
pytest /tests/test_*.py --json-report --json-report-file=/logs/pytest_results.json || true

# 2. Calculate reward
# We use a small python script to compare pytest results with metadata.json
python3 -c "
import json
from pathlib import Path

def calculate():
    try:
        # Load expected results from metadata
        meta_path = Path('/setup_files/metadata.json')
        if not meta_path.exists():
            return 0.0
        meta = json.loads(meta_path.read_text())
        expected = meta.get('expected_output_json', {})
        
        # Load actual results from pytest
        report_path = Path('/logs/pytest_results.json')
        if not report_path.exists():
            return 0.0
        report = json.loads(report_path.read_text())
        
        actual_results = {}
        for test in report.get('tests', []):
            name = test['nodeid'].split('::')[-1]
            actual_results[name] = 'passed' if test['outcome'] == 'passed' else 'failed'
            
        # Compare
        for test_name, expected_status in expected.items():
            if actual_results.get(test_name) != expected_status:
                return 0.0
        return 1.0
    except Exception as e:
        print(f'Error calculating reward: {e}')
        return 0.0

reward = calculate()
Path('/logs/verifier/reward.txt').write_text(str(reward))
print(f'REWARD: {reward}')
"
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

ALL_REPOS = set(_SHARED_BASE_IMAGES.keys())

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

    # 2. test.sh
    replacements["tests/test.sh"] = _TEST_SH.encode()

    # 3. test_state.py
    replacements["tests/test_state.py"] = _TEST_STATE_PY.encode()

    # 4. Inject test files from R2E-Gym-Lite
    for fname, code in zip(test_file_names, test_file_codes):
        if not fname.startswith("tests/"):
            fname = f"tests/{fname}"
        replacements[fname] = code.encode()

    # 5. setup_files/metadata.json
    meta_bytes = existing.get("environment/workspace/metadata.json", existing.get("setup_files/metadata.json", b"{}"))
    replacements["setup_files/metadata.json"] = meta_bytes

    # 6. solution/solve.sh
    replacements["solution/solve.sh"] = _SOLVE_SH_TEMPLATE.format(base_commit=base_commit).encode()

    # 7. instruction.md
    orig_instruction = existing.get("instruction.md", b"")
    if b"## Environment Setup" not in orig_instruction:
        preamble = _SETUP_PREAMBLE.format(base_commit=base_commit).encode()
        replacements["instruction.md"] = preamble + orig_instruction

    # Merge
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--upload-to", type=str, default=None)
    args = parser.parse_args()

    print("Loading R2E-Gym-Lite (test files)...")
    lite_ds = load_dataset("R2E-Gym/R2E-Gym-Lite", split="train", token=os.environ.get("HF_TOKEN"))
    commit_to_tests = {row["commit_hash"]: row for row in lite_ds}
    print(f"Loaded {len(commit_to_tests)} test sets.")

    print("Loading DCAgent2/r2egym_sandboxes (tarballs)...")
    sandboxes_ds = load_dataset("DCAgent2/r2egym_sandboxes", split="train", token=os.environ.get("HF_TOKEN"))
    if args.limit:
        sandboxes_ds = sandboxes_ds.select(range(args.limit))

    patched_rows = []
    for i, row in enumerate(sandboxes_ds):
        try:
            # Match by commit hash
            # Original tarball has metadata.json inside
            with tarfile.open(fileobj=io.BytesIO(bytes(row["task_binary"])), mode="r:gz") as tf:
                meta = None
                for p in ["environment/workspace/metadata.json", "setup_files/metadata.json"]:
                    try:
                        meta = json.loads(tf.extractfile(p).read())
                        break
                    except: pass
                
                if not meta: continue
                commit = meta.get("base_commit") or meta.get("new_commit_hash")
                
                if commit not in commit_to_tests:
                    print(f"[{i}] Skip: commit {commit} not in Lite dataset")
                    continue
                
                lite_row = commit_to_tests[commit]
                new_binary = repack_task(
                    bytes(row["task_binary"]),
                    lite_row["test_file_names"],
                    lite_row["test_file_codes"]
                )
                
                row["task_binary"] = new_binary
                patched_rows.append(row)
                print(f"[{i}] Patched: {meta.get('repo_name')} @ {commit[:8]}")
        except Exception as e:
            print(f"[{i}] Error: {e}")

    from datasets import Dataset
    out_ds = Dataset.from_list(patched_rows)
    out_ds.save_to_disk(args.output_dir)
    print(f"Saved {len(out_ds)} tasks to {args.output_dir}")

    if args.upload_to:
        out_ds.push_to_hub(args.upload_to, token=os.environ.get("HF_TOKEN"))
        print(f"Uploaded to {args.upload_to}")

if __name__ == "__main__":
    main()
