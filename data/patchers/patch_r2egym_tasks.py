#!/usr/bin/env python3
"""
Patch DCAgent2/r2egym_sandboxes with Harbor-compatible solution/solve.sh and
tests/test_state.py files.

Each task in the r2egym_sandboxes dataset already has:
  - instruction.md       (the issue / problem statement)
  - task.toml
  - environment/Dockerfile
  - environment/workspace/metadata.json   (contains expected_output_json)
  - tests/test.sh        (verifier: runs /root/run_tests.sh, calculates reward,
                          writes to /logs/verifier/reward.txt)

This patcher adds:
  - solution/solve.sh    (oracle: no-op, because the docker image already has
                          the FIXED code at the tagged commit; the verifier
                          test.sh handles everything)
  - tests/test_state.py  (Harbor test-state reader)

Why no-op oracle?
  The docker images are named `namanjain12/<repo>_final:<commit_hash>` where
  <commit_hash> is the FIXED commit.  The image therefore starts with the
  correct code already applied.  The existing test.sh runs the unit tests and
  writes 1.0 to /logs/verifier/reward.txt when they all pass — so the oracle
  just needs to let the verifier run without modifying anything.

Usage:
    python3 data/patchers/patch_r2egym_tasks.py \
        --sandbox-repo DCAgent2/r2egym_sandboxes \
        --output-dir /path/to/output \
        [--limit N] \
        [--upload-to SankalpKJ/r2egym-patched] \
        [--hf-token TOKEN]
"""

import argparse
import gzip
import io
import tarfile
from pathlib import Path
from typing import Optional

from datasets import Dataset, Features, Value, load_dataset

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

# The oracle is a no-op: the docker image already has the fixed code.
# The verifier (test.sh) runs the tests and writes reward to
# /logs/verifier/reward.txt automatically.
_SOLVE_SH = """\
#!/usr/bin/env bash
# Oracle solution for R2E-Gym tasks.
#
# The docker image is built from the FIXED commit, so the repository at
# /testbed already contains the correct code.  The verifier (test.sh) will
# run the unit tests and write the reward to /logs/verifier/reward.txt.
#
# Nothing needs to be done here.
echo "[oracle] R2E-Gym oracle: docker image already has the fixed code."
echo "[oracle] The verifier will run the tests and compute the reward."
"""

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


# ---------------------------------------------------------------------------
# Tarball manipulation
# ---------------------------------------------------------------------------

def repack_task(task_binary: bytes) -> bytes:
    """Add solution/solve.sh and tests/test_state.py to the task tarball."""
    # Read existing tarball
    with gzip.open(io.BytesIO(task_binary)) as gz_in:
        with tarfile.open(fileobj=gz_in) as tar_in:
            members: dict[str, tuple] = {}
            for member in tar_in.getmembers():
                f = tar_in.extractfile(member)
                members[member.name] = (member, f.read() if f else b'')

    # Build new tarball
    out_buf = io.BytesIO()
    with gzip.GzipFile(fileobj=out_buf, mode='wb', mtime=0) as gz_out:
        with tarfile.open(fileobj=gz_out, mode='w') as tar_out:
            # Write existing files
            for name, (member, data) in members.items():
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mode = member.mode
                info.mtime = 0
                tar_out.addfile(info, io.BytesIO(data))

            def add_file(name: str, content: str, mode: int = 0o644) -> None:
                data = content.encode('utf-8')
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mode = mode
                info.mtime = 0
                tar_out.addfile(info, io.BytesIO(data))

            # Add solution/
            add_file('solution/solve.sh', _SOLVE_SH, mode=0o755)

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
        help='HuggingFace token for upload',
    )
    args = parser.parse_args()

    print(f'Loading sandbox dataset: {args.sandbox_repo}')
    ds = load_dataset(args.sandbox_repo, split='train')
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f'Total tasks: {len(ds)}')

    stats = {'total': 0, 'patched': 0, 'error': 0}
    patched_rows = []

    for i, row in enumerate(ds):
        stats['total'] += 1
        path = row['path']
        task_binary = bytes(row['task_binary'])

        try:
            new_binary = repack_task(task_binary)
        except Exception as e:
            stats['error'] += 1
            print(f'  [{path}] Error repacking: {e}')
            continue

        patched_rows.append({'path': path, 'task_binary': new_binary})
        stats['patched'] += 1

        if i % 500 == 0 and i > 0:
            print(f'  Progress: {i}/{len(ds)} (patched={stats["patched"]})')

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
