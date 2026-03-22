#!/usr/bin/env python3
"""
Patch DCAgent2/r2egym_sandboxes with Harbor-compatible solution/solve.sh and
tests/test_state.py files.

Each task in the r2egym_sandboxes dataset has:
  - instruction.md       (the issue / problem statement)
  - task.toml
  - environment/Dockerfile
  - environment/workspace/metadata.json   (contains expected_output_json,
                                           base_commit = the FIXED commit hash)
  - tests/test.sh        (verifier: runs /root/run_tests.sh, calculates reward,
                          writes to /logs/verifier/reward.txt)

The docker images are built from the FIXED commit but with a REVERSE patch
applied (SWE-bench style), so the code in the container is BROKEN.

The oracle solve.sh must apply the forward patch (broken → fixed) before the
verifier runs.  The ground-truth patch is reconstructed from R2E-Gym-Lite
(R2E-Gym/R2E-Gym-Lite) which contains old_file_content and new_file_content
for each changed file.

This patcher adds / modifies:
  - solution/solve.sh         (oracle: applies the forward patch)
  - tests/test_state.py       (Harbor reward reader)
  - environment/Dockerfile    (COPY workspace replaced with inline RUN via base64)

Usage:
    python3 data/patchers/patch_r2egym_tasks.py \
        --sandbox-repo DCAgent2/r2egym_sandboxes \
        --output-dir /path/to/output \
        [--limit N] \
        [--upload-to SankalpKJ/r2egym-patched] \
        [--hf-token TOKEN]
"""

import argparse
import base64
import difflib
import gzip
import io
import json
import tarfile
from pathlib import Path
from typing import Optional

from datasets import Dataset, Features, Value, load_dataset

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

# Harbor reads the reward via this Python module.
_TEST_STATE_PY = """\
import os

REWARD_FILE = os.environ.get("REWARD_FILE", "/logs/verifier/reward.txt")


def get_reward() -> float:
    try:
        with open(REWARD_FILE) as f:
            return float(f.read().strip())
    except Exception:
        return 0.0
"""


def _build_solve_sh(patch_b64: str) -> str:
    """
    Build the oracle solve.sh that:
    1. Decodes the base64 patch
    2. Applies it with `git apply` at /testbed
    """
    return f"""\
#!/usr/bin/env bash
# Oracle solution for R2E-Gym tasks.
#
# The docker image starts with the BROKEN code (reverse-patched from the fixed
# commit).  This script applies the forward patch to restore the fixed code,
# then the verifier (test.sh) runs the unit tests and writes the reward.

set -e

PATCH_B64="{patch_b64}"

echo "[oracle] Decoding and applying forward patch..."
echo "$PATCH_B64" | base64 -d > /tmp/oracle_fix.patch

cd /testbed
git apply /tmp/oracle_fix.patch && echo "[oracle] Patch applied successfully." || {{
    echo "[oracle] git apply failed, trying patch -p1..."
    patch -p1 < /tmp/oracle_fix.patch
}}

echo "[oracle] Done. Verifier will now run the tests."
"""


def _build_dockerfile(base_image: str, metadata_json_str: str) -> str:
    """
    Build a Daytona-compatible Dockerfile that:
    1. Inherits from the r2egym base image (which has the broken code).
    2. Writes metadata.json inline via base64 (single-line, no Dockerfile parse issues).
    3. Creates /logs directory for the reward file.
    """
    b64 = base64.b64encode(metadata_json_str.encode('utf-8')).decode('ascii')
    return (
        f"FROM {base_image}\n"
        f"RUN mkdir -p /workspace /logs/verifier\n"
        f"RUN echo '{b64}' | base64 -d > /workspace/metadata.json\n"
        f"WORKDIR /testbed\n"
    )


# ---------------------------------------------------------------------------
# Patch generation from R2E-Gym-Lite
# ---------------------------------------------------------------------------

def _is_test_file(path: str) -> bool:
    """Return True if the file is a test file (should be excluded from patch)."""
    p = path.lower()
    return (
        '/test' in p
        or p.startswith('test')
        or p.endswith('_test.py')
        or '/tests/' in p
    )


def _generate_patch_from_file_diffs(file_diffs: list) -> str:
    """
    Reconstruct a unified diff from the file_diffs in parsed_commit_content.
    Only includes non-test source files.
    """
    patch_parts = []
    for fd in file_diffs:
        path = fd.get('header', {}).get('file', {}).get('path', '')
        if not path:
            # Try to infer from minus/plus file
            minus = fd.get('minus_file', {})
            if isinstance(minus, dict):
                path = minus.get('path', '').lstrip('a/')
        if not path or _is_test_file(path):
            continue
        old_content = fd.get('old_file_content', '')
        new_content = fd.get('new_file_content', '')
        if old_content == new_content:
            continue
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f'a/{path}',
            tofile=f'b/{path}',
        ))
        if diff:
            patch_parts.extend(diff)
    return ''.join(patch_parts)


def build_patch_lookup(r2egym_lite_repo: str = 'R2E-Gym/R2E-Gym-Lite') -> dict[str, str]:
    """
    Build a dict mapping commit_hash -> unified_patch_str from R2E-Gym-Lite.
    """
    print(f'Loading {r2egym_lite_repo} to build patch lookup...')
    ds = load_dataset(r2egym_lite_repo, split='train')
    lookup: dict[str, str] = {}
    for row in ds:
        commit = row['commit_hash']
        try:
            parsed = json.loads(row['parsed_commit_content'])
            patch = _generate_patch_from_file_diffs(parsed.get('file_diffs', []))
        except Exception as e:
            patch = ''
        lookup[commit] = patch
    print(f'Patch lookup built: {len(lookup)} entries, {sum(1 for v in lookup.values() if v)} with non-empty patches')
    return lookup


# ---------------------------------------------------------------------------
# Tarball manipulation
# ---------------------------------------------------------------------------

def repack_task(task_binary: bytes, patch_str: str) -> bytes:
    """
    Repack the task tarball with:
    - solution/solve.sh          (oracle: applies the forward patch)
    - tests/test_state.py        (Harbor reward reader)
    - environment/Dockerfile     (rewritten to inline metadata.json via base64)
    """
    # Read existing tarball
    with gzip.open(io.BytesIO(task_binary)) as gz_in:
        with tarfile.open(fileobj=gz_in) as tar_in:
            members: dict[str, tuple] = {}
            for member in tar_in.getmembers():
                f = tar_in.extractfile(member)
                members[member.name] = (member, f.read() if f else b'')

    # Extract metadata.json
    meta_key = 'environment/workspace/metadata.json'
    if meta_key not in members:
        raise ValueError(f'Missing {meta_key} in tarball')
    metadata_json_str = members[meta_key][1].decode('utf-8')
    metadata = json.loads(metadata_json_str)
    base_image = metadata.get('docker_image', '')
    if not base_image:
        raise ValueError('docker_image not found in metadata.json')

    # Build Dockerfile and solve.sh
    new_dockerfile = _build_dockerfile(base_image, metadata_json_str)
    patch_b64 = base64.b64encode(patch_str.encode('utf-8')).decode('ascii')
    solve_sh = _build_solve_sh(patch_b64)

    # Build new tarball
    out_buf = io.BytesIO()
    with gzip.GzipFile(fileobj=out_buf, mode='wb', mtime=0) as gz_out:
        with tarfile.open(fileobj=gz_out, mode='w') as tar_out:

            def add_bytes(name: str, data: bytes, mode: int = 0o644) -> None:
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mode = mode
                info.mtime = 0
                tar_out.addfile(info, io.BytesIO(data))

            def add_file(name: str, content: str, mode: int = 0o644) -> None:
                add_bytes(name, content.encode('utf-8'), mode)

            # Write existing files, replacing the Dockerfile
            for name, (member, data) in members.items():
                if member.isdir():
                    dir_info = tarfile.TarInfo(name=name)
                    dir_info.type = tarfile.DIRTYPE
                    dir_info.mode = member.mode
                    dir_info.mtime = 0
                    dir_info.size = 0
                    tar_out.addfile(dir_info)
                elif name == 'environment/Dockerfile':
                    add_file(name, new_dockerfile)
                else:
                    add_bytes(name, data, member.mode)

            # Add solution/ directory and solve.sh
            dir_info = tarfile.TarInfo(name='solution')
            dir_info.type = tarfile.DIRTYPE
            dir_info.mode = 0o755
            dir_info.mtime = 0
            dir_info.size = 0
            tar_out.addfile(dir_info)
            add_file('solution/solve.sh', solve_sh, mode=0o755)

            # Add tests/test_state.py (test.sh already exists)
            add_file('tests/test_state.py', _TEST_STATE_PY)

    return out_buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Patch r2egym sandboxes with Harbor-compatible oracle files'
    )
    parser.add_argument(
        '--sandbox-repo',
        default='DCAgent2/r2egym_sandboxes',
        help='HuggingFace repo ID for the sandbox dataset',
    )
    parser.add_argument(
        '--r2egym-lite-repo',
        default='R2E-Gym/R2E-Gym-Lite',
        help='HuggingFace repo ID for R2E-Gym-Lite (source of ground-truth patches)',
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for patched dataset',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of tasks to process (for testing)',
    )
    parser.add_argument(
        '--upload-to',
        default=None,
        help='HuggingFace repo ID to upload patched dataset',
    )
    parser.add_argument(
        '--hf-token',
        default=None,
        help='HuggingFace token for upload (falls back to HF_TOKEN env var)',
    )
    args = parser.parse_args()

    import os
    if args.hf_token is None:
        args.hf_token = os.environ.get('HF_TOKEN')

    # Build patch lookup from R2E-Gym-Lite
    patch_lookup = build_patch_lookup(args.r2egym_lite_repo)

    print(f'Loading sandbox dataset: {args.sandbox_repo}')
    ds = load_dataset(args.sandbox_repo, split='train')
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f'Total tasks: {len(ds)}')

    stats = {'total': 0, 'patched': 0, 'no_patch': 0, 'error': 0}
    patched_rows = []

    for i, row in enumerate(ds):
        stats['total'] += 1
        path = row['path']
        task_binary = bytes(row['task_binary'])

        # Get commit hash from metadata
        try:
            with gzip.open(io.BytesIO(task_binary)) as gz:
                with tarfile.open(fileobj=gz) as tar:
                    f = tar.extractfile('environment/workspace/metadata.json')
                    meta = json.loads(f.read().decode('utf-8'))
            commit = meta.get('base_commit', '')
        except Exception as e:
            stats['error'] += 1
            print(f'  [{path}] Error reading metadata: {e}')
            continue

        patch_str = patch_lookup.get(commit, '')
        if not patch_str:
            stats['no_patch'] += 1
            print(f'  [{path}] WARNING: no patch found for commit {commit[:12]}')
            # Still patch with empty patch (no-op oracle) — may not pass but
            # keeps the task in the dataset for manual inspection
            patch_str = '# No patch found\n'

        try:
            new_binary = repack_task(task_binary, patch_str)
        except Exception as e:
            stats['error'] += 1
            print(f'  [{path}] Error repacking: {e}')
            continue

        patched_rows.append({'path': path, 'task_binary': new_binary})
        stats['patched'] += 1

        if i % 500 == 0 and i > 0:
            print(f'  Progress: {i}/{len(ds)} (patched={stats["patched"]}, no_patch={stats["no_patch"]})')

    print(f'\nStats: {stats}')

    if patched_rows:
        features = Features({
            'path': Value('string'),
            'task_binary': Value('binary'),
        })
        out_ds = Dataset.from_list(patched_rows, features=features)
        out_ds.save_to_disk(args.output_dir)
        print(f'Saved {len(patched_rows)} patched tasks to {args.output_dir}')

        if args.upload_to:
            print(f'Uploading to {args.upload_to}...')
            out_ds.push_to_hub(
                args.upload_to,
                token=args.hf_token,
                private=False,
            )
            print(f'Uploaded to https://huggingface.co/datasets/{args.upload_to}')


if __name__ == '__main__':
    main()
