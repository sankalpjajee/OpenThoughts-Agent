#!/usr/bin/env python3
"""
patch_r2egym_tasks.py
---------------------
Patches DCAgent2/r2egym_sandboxes for Harbor/Daytona compatibility.

KEY DESIGN GOAL: Reduce unique Dockerfiles from 4578 → 10 (one per repo).
The RL training pipeline has a safety limit of ~10 unique Daytona snapshots.

Strategy:
  - There are exactly 10 repos in the dataset:
    pandas, numpy, pillow, orange3, aiohttp, tornado, scrapy, pyramid, datalad, coveragepy
  - Each repo has a single representative docker image (one pinned commit hash).
  - All tasks for a given repo share the SAME Dockerfile (FROM <repo_image>).
  - The oracle solve.sh does: git checkout <task_base_commit> -- .
    This restores the fixed code for the specific task, since the git history
    is intact inside the container.
  - The test.sh is unchanged — it uses /testbed/.venv which is in the image.

The docker images are built at a fixed commit but with a REVERSE patch applied
(SWE-bench style), so the code in the container starts in the BROKEN state.
The git history is intact, so git checkout restores the fixed state.

This patcher adds / modifies:
  - solution/solve.sh         (oracle: git checkout to restore fixed code)
  - tests/test_state.py       (Harbor reward reader)
  - environment/Dockerfile    (replaced with shared per-repo Dockerfile)

Usage:
    python3 data/patchers/patch_r2egym_tasks.py \\
        --sandbox-repo DCAgent2/r2egym_sandboxes \\
        --output-dir /path/to/output \\
        [--limit N] \\
        [--upload-to SankalpKJ/r2egym-patched] \\
        [--hf-token TOKEN]
"""

import argparse
import base64
import gzip
import io
import json
import os
import tarfile

from datasets import Dataset, Features, Value, load_dataset

# ---------------------------------------------------------------------------
# One representative docker image per repo (10 total).
# These are the first-seen images from the dataset scan.
# All tasks for a given repo will use this shared Dockerfile.
# ---------------------------------------------------------------------------

REPO_IMAGES = {
    "aiohttp":    "namanjain12/aiohttp_final:f0d74880deec8fcd982bce639c93c5e130d41198",
    "coveragepy": "namanjain12/coveragepy_final:c1bfa7352368b63f3a9b30c02f242408d07a7ab2",
    "datalad":    "namanjain12/datalad_final:f5e1d276ab51aefcf5e48e6f7bd9833b19ef7f90",
    "numpy":      "namanjain12/numpy_final:14445500bdf67600f926c6426bad55977441dca0",
    "orange3":    "namanjain12/orange3_final:2d9617bd0cb1f0ba61771258410ab8fae8e7e24d",
    "pandas":     "namanjain12/pandas_final:fadb72cf5ef8489e409d4d33625bd16a76fa7a42",
    "pillow":     "namanjain12/pillow_final:f644adbb05d615a9902ef3643714d5fe8049cea3",
    "pyramid":    "namanjain12/pyramid_final:fbbb20c7953370c86f999e865b1a9d682690eb70",
    "scrapy":     "namanjain12/scrapy_final:fbb411a805724fec50b786f369be79dc221c798e",
    "tornado":    "namanjain12/tornado_final:b5ec807edc83c8e7d1d12553d635ebe765e5c614",
}

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


def _build_dockerfile(repo_name: str, metadata_json_str: str) -> str:
    """
    Build a shared per-repo Dockerfile.
    All tasks in the same repo use the SAME base image.
    metadata.json is inlined via base64 to avoid Daytona build context issues.
    """
    base_image = REPO_IMAGES[repo_name]
    b64 = base64.b64encode(metadata_json_str.encode('utf-8')).decode('ascii')
    return (
        f"FROM {base_image}\n"
        f"RUN mkdir -p /workspace /logs/verifier\n"
        f"RUN echo '{b64}' | base64 -d > /workspace/metadata.json\n"
        f"WORKDIR /testbed\n"
    )


def _build_solve_sh(base_commit: str) -> str:
    """
    Oracle solve.sh: restore the fixed code via git checkout.
    The docker image has the git history intact; the broken state is a
    reverse patch on top of the fixed commit.
    """
    return f"""\
#!/usr/bin/env bash
# Oracle solution for R2E-Gym tasks.
#
# The docker image starts with BROKEN code (reverse-patched from the fixed commit).
# The git history is intact, so we simply restore the fixed files.

set -e

BASE_COMMIT="{base_commit}"

echo "[oracle] Restoring fixed code at commit $BASE_COMMIT..."
cd /testbed
git checkout "$BASE_COMMIT" -- .
echo "[oracle] Done. Verifier will now run the tests."
"""


# ---------------------------------------------------------------------------
# Tarball manipulation
# ---------------------------------------------------------------------------

def repack_task(task_binary: bytes) -> bytes:
    """
    Repack the task tarball with:
    - solution/solve.sh          (oracle: git checkout to restore fixed code)
    - tests/test_state.py        (Harbor reward reader)
    - environment/Dockerfile     (shared per-repo Dockerfile, 10 unique total)
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

    repo_name = metadata.get('repo_name', '')
    if not repo_name:
        raise ValueError('repo_name not found in metadata.json')
    if repo_name not in REPO_IMAGES:
        raise ValueError(f'Unknown repo_name: {repo_name!r} (not in REPO_IMAGES)')

    base_commit = metadata.get('base_commit', '')
    if not base_commit:
        raise ValueError('base_commit not found in metadata.json')

    # Build new Dockerfile and solve.sh
    new_dockerfile = _build_dockerfile(repo_name, metadata_json_str)
    solve_sh = _build_solve_sh(base_commit)

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
        description='Patch r2egym sandboxes with Harbor-compatible oracle files (10 unique Dockerfiles)'
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
        help='HuggingFace token for upload (falls back to HF_TOKEN env var)',
    )
    args = parser.parse_args()

    if args.hf_token is None:
        args.hf_token = os.environ.get('HF_TOKEN')

    print(f'Loading sandbox dataset: {args.sandbox_repo}')
    ds = load_dataset(args.sandbox_repo, split='train')
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f'Total tasks: {len(ds)}')
    print(f'Unique Dockerfiles will be: {len(REPO_IMAGES)} (one per repo)')

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
