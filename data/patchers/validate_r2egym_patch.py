#!/usr/bin/env python3
"""
Quick validation of patched R2E-Gym dataset.

Checks that each task tarball has the expected structure after patching:
  - environment/Dockerfile (using original per-task image as base)
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
    python validate_r2egym_patch.py --hf-repo SankalpKJ/r2egym-patched --limit 20
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile


def validate_tarball(task_binary: bytes, path: str) -> dict:
    """Validate a single task tarball. Returns dict of issues."""
    issues = []
    files_found = set()

    try:
        with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:gz") as tf:
            for m in tf.getmembers():
                if m.isfile():
                    files_found.add(m.name)
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

    # At least one test_*.py file
    test_files = [f for f in files_found if f.startswith("tests/test_") and f.endswith(".py") and f != "tests/test_state.py"]
    if not test_files:
        issues.append("No test_*.py files found in tests/")

    # Check Dockerfile content — should have a FROM line and our infrastructure dirs
    if "environment/Dockerfile" in files_found:
        with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:gz") as tf:
            df = tf.extractfile(tf.getmember("environment/Dockerfile")).read().decode()
            if "FROM " not in df:
                issues.append("Dockerfile missing FROM line")
            if "/logs" not in df:
                issues.append("Dockerfile missing /logs directory setup")

    # Check metadata has expected_output_json
    if "setup_files/metadata.json" in files_found:
        with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:gz") as tf:
            meta = json.loads(tf.extractfile(tf.getmember("setup_files/metadata.json")).read())
            if "expected_output_json" not in meta:
                issues.append("metadata.json missing expected_output_json")

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
    repo_counts = {}
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

        # Track Dockerfiles
        try:
            with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:gz") as tf:
                df = tf.extractfile(tf.getmember("environment/Dockerfile")).read().decode()
                unique_dfs.add(df[:100])  # first 100 chars as key
                # Track repo
                meta = json.loads(tf.extractfile(tf.getmember("setup_files/metadata.json")).read())
                repo = meta.get("repo_name", "unknown")
                repo_counts[repo] = repo_counts.get(repo, 0) + 1
        except Exception:
            pass

        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{len(ds)}")

    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Total:   {total}")
    print(f"  Valid:   {valid}")
    print(f"  Issues:  {total - valid}")
    print(f"  Unique Dockerfiles: {len(unique_dfs)}")
    print()
    print(f"Per-repo:")
    for repo, count in sorted(repo_counts.items(), key=lambda x: -x[1]):
        print(f"  {repo}: {count}")

    if all_issues:
        print(f"\nFirst 10 issues:")
        for result in all_issues[:10]:
            print(f"  [{result['path']}]: {result['issues']}")


if __name__ == "__main__":
    main()
