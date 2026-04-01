#!/usr/bin/env python3
"""
Quick validation of patched R2E-Gym dataset.

Checks that each task tarball has the expected structure after patching:
  - environment/Dockerfile (one of 10 shared namanjain12 images)
  - tests/test.sh (reward-calculating test runner)
  - tests/test_state.py (Harbor reward reader)
  - tests/test_*.py (at least one test file from R2E-Gym-Lite)
  - setup_files/metadata.json (with expected_output_json)
  - solution/solve.sh (oracle: git checkout)
  - instruction.md (with setup preamble)

Usage:
    # Validate from saved disk dataset
    python validate_r2egym_patch.py /mnt/sda4T/home/jajee/r2egym_patched

    # Validate from HuggingFace
    python validate_r2egym_patch.py --hf-repo SankalpKJ/r2egym-patched-v8-full --limit 20
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile

from patch_r2egym_tasks import _SHARED_BASE_IMAGES

_SHARED_BASE_IMAGE_SET = set(_SHARED_BASE_IMAGES.values())


def validate_tarball(task_binary: bytes, path: str) -> dict:
    """Validate a single task tarball. Returns dict with issues list."""
    issues = []
    files_found = set()
    dockerfile_content = None
    metadata_content = None

    # BUG 3 FIX: read all needed content in a single tarball open
    try:
        with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:gz") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                files_found.add(m.name)
                f = tf.extractfile(m)
                if f is None:
                    continue
                data = f.read()
                if m.name == "environment/Dockerfile":
                    dockerfile_content = data.decode(errors="replace")
                elif m.name == "setup_files/metadata.json":
                    metadata_content = data
    except Exception as e:
        return {"path": path, "issues": [f"Cannot read tarball: {e}"]}

    # Required files
    required = [
        "environment/Dockerfile",
        "tests/test.sh",
        "tests/test_state.py",
        "setup_files/metadata.json",
        "solution/solve.sh",
        "instruction.md",
    ]
    for req in required:
        if req not in files_found:
            issues.append(f"Missing: {req}")

    # At least one injected test_*.py file
    test_files = [
        f for f in files_found
        if f.startswith("tests/test_") and f.endswith(".py") and f != "tests/test_state.py"
    ]
    if not test_files:
        issues.append("No test_*.py files found in tests/")

    # BUG 2 FIX: check that Dockerfile FROM line is one of the 10 known shared
    # base images (all under namanjain12/).  The old check was inverted — it
    # flagged every correctly-patched task because the shared images ARE hosted
    # under namanjain12/.
    if dockerfile_content is not None:
        from_line = next(
            (line.strip() for line in dockerfile_content.splitlines()
             if line.strip().upper().startswith("FROM")),
            "",
        )
        image_ref = from_line[len("FROM"):].strip()
        if image_ref not in _SHARED_BASE_IMAGE_SET:
            issues.append(
                f"Dockerfile FROM is not a known shared base image: {image_ref!r}"
            )

    # Check metadata has expected_output_json
    if metadata_content is not None:
        try:
            meta = json.loads(metadata_content)
            if "expected_output_json" not in meta:
                issues.append("metadata.json missing expected_output_json")
            else:
                # Verify it is parseable (it is stored as a JSON string)
                exp_raw = meta["expected_output_json"]
                if isinstance(exp_raw, str):
                    try:
                        json.loads(exp_raw)
                    except json.JSONDecodeError:
                        issues.append("expected_output_json is a string but not valid JSON")
        except json.JSONDecodeError:
            issues.append("setup_files/metadata.json is not valid JSON")

    return {
        "path": path,
        "issues": issues,
        "files": sorted(files_found),
        "test_files": test_files,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate patched R2E-Gym dataset")
    parser.add_argument("disk_path", nargs="?", help="Path to saved dataset on disk")
    parser.add_argument("--hf-repo", help="HuggingFace repo to validate")
    parser.add_argument("--limit", type=int, default=None, help="Validate first N tasks")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")

    from datasets import load_from_disk, load_dataset

    if args.disk_path:
        print(f"Loading from disk: {args.disk_path}")
        ds = load_from_disk(args.disk_path)
    elif args.hf_repo:
        print(f"Loading from HF: {args.hf_repo}")
        split = f"train[:{args.limit}]" if args.limit else "train"
        ds = load_dataset(args.hf_repo, split=split, token=token)
    else:
        print("ERROR: provide disk_path or --hf-repo")
        sys.exit(1)

    if args.limit and args.disk_path:
        ds = ds.select(range(min(args.limit, len(ds))))

    print(f"Validating {len(ds)} tasks...\n")

    total = 0
    valid = 0
    unique_dfs = set()
    repo_counts: dict[str, int] = {}
    all_issues = []

    for i, row in enumerate(ds):
        total += 1
        path = row.get("path", f"task-{i}")
        task_binary = row.get("task_binary")
        if isinstance(task_binary, list):
            task_binary = bytes(task_binary)

        result = validate_tarball(task_binary, path)

        if not result["issues"]:
            valid += 1
        else:
            all_issues.append(result)
            if args.verbose and len(all_issues) <= 20:
                print(f"  ISSUES [{path}]: {result['issues']}")

        # Track unique Dockerfiles and repos (single open, already done in validate_tarball)
        try:
            with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:gz") as tf:
                members = {m.name: m for m in tf.getmembers() if m.isfile()}
                if "environment/Dockerfile" in members:
                    df = tf.extractfile(members["environment/Dockerfile"]).read().decode()
                    unique_dfs.add(df[:120])
                if "setup_files/metadata.json" in members:
                    meta = json.loads(tf.extractfile(members["setup_files/metadata.json"]).read())
                    repo = meta.get("repo_name", "unknown")
                    repo_counts[repo] = repo_counts.get(repo, 0) + 1
        except Exception:
            pass

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(ds)}")

    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Total:              {total}")
    print(f"  Valid:              {valid}")
    print(f"  Issues:             {total - valid}")
    print(f"  Unique Dockerfiles: {len(unique_dfs)}  (expected: 10)")
    print()
    print(f"Per-repo task counts:")
    for repo, count in sorted(repo_counts.items(), key=lambda x: -x[1]):
        print(f"  {repo}: {count}")

    if all_issues:
        print(f"\nFirst 10 issues:")
        for result in all_issues[:10]:
            print(f"  [{result['path']}]: {result['issues']}")


if __name__ == "__main__":
    main()
