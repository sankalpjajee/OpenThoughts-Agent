#!/usr/bin/env python3
"""
Patch penfever/nemotron-tc-adapters-math-sandboxes with verifiers and oracle solutions.

Each task in the math sandboxes dataset has:
  - instruction.md  (the math problem)
  - task.toml
  - environment/Dockerfile

This patcher adds:
  - tests/test.sh          (verifier: reads /app/solution.txt, compares to expected)
  - tests/test_state.py    (Harbor test state file)
  - tests/expected.txt     (expected answer)
  - solution/solve.sh      (oracle: writes expected answer to /app/solution.txt)

The oracle answer is extracted from the NTC SFT trajectories by finding the last
`cat solution.txt` terminal output in the conversation.

Usage:
    python3 patch_nemotron_math_tasks.py \
        --trajectory-parquet /path/to/ntc_math.parquet \
        --output-dir /path/to/output \
        [--limit N] \
        [--upload-to HF_REPO_ID] \
        [--hf-token TOKEN]
"""

import argparse
import gzip
import io
import json
import os
import re
import tarfile
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq
from datasets import Dataset, Features, Value, load_dataset

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_TEST_SH = """\
#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/logs"
VERIFIER_DIR="/logs/verifier"
REWARD_FILE="$VERIFIER_DIR/reward.txt"

mkdir -p "$LOG_DIR" "$VERIFIER_DIR"
echo 0 > "$REWARD_FILE"

log() { echo "[test] $*" >&2; }

# ---------------------------------------------------------------------------
# Check solution file exists
# ---------------------------------------------------------------------------
if [ ! -f /app/solution.txt ]; then
    log "FAIL: /app/solution.txt not found"
    exit 0
fi

ACTUAL=$(cat /app/solution.txt | tr -d '\\r' | sed 's/[[:space:]]*$//' | sed '/^$/d')
EXPECTED=$(cat /app/expected.txt | tr -d '\\r' | sed 's/[[:space:]]*$//' | sed '/^$/d')

# ---------------------------------------------------------------------------
# Exact string comparison (case-insensitive, whitespace-normalized)
# ---------------------------------------------------------------------------
ACTUAL_NORM=$(echo "$ACTUAL" | tr -s ' ' | tr '[:upper:]' '[:lower:]')
EXPECTED_NORM=$(echo "$EXPECTED" | tr -s ' ' | tr '[:upper:]' '[:lower:]')

if [ "$ACTUAL_NORM" = "$EXPECTED_NORM" ]; then
    log "PASS: exact match"
    echo 1 > "$REWARD_FILE"
    exit 0
fi

# ---------------------------------------------------------------------------
# Numeric comparison with tolerance (for single-number answers)
# ---------------------------------------------------------------------------
NUMERIC_CHECK=$(python3 -c "
import sys, re, math

def to_num(s):
    s = s.strip()
    # Try direct float
    try:
        return float(s)
    except ValueError:
        pass
    # Try fraction like 3/4
    m = re.match(r'^(-?\\d+)\\s*/\\s*(\\d+)$', s)
    if m:
        return int(m.group(1)) / int(m.group(2))
    return None

actual = to_num('''$ACTUAL''')
expected = to_num('''$EXPECTED''')

if actual is None or expected is None:
    sys.exit(1)

if expected == 0:
    ok = abs(actual - expected) < 1e-6
else:
    ok = abs(actual - expected) / max(abs(expected), 1e-10) < 1e-4

sys.exit(0 if ok else 1)
" 2>/dev/null && echo "pass" || echo "fail")

if [ "$NUMERIC_CHECK" = "pass" ]; then
    log "PASS: numeric match"
    echo 1 > "$REWARD_FILE"
    exit 0
fi

log "FAIL: expected='$EXPECTED_NORM' got='$ACTUAL_NORM'"
exit 0
"""

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

_SOLVE_SH_TEMPLATE = """\
#!/usr/bin/env bash
# Oracle solution: write the expected answer to /app/solution.txt
set -euo pipefail

mkdir -p /app

cat > /app/solution.txt << 'ORACLE_ANSWER_EOF'
{answer}
ORACLE_ANSWER_EOF

echo "[oracle] Written answer to /app/solution.txt"
cat /app/solution.txt
"""

# ---------------------------------------------------------------------------
# Answer extraction from NTC trajectories
# ---------------------------------------------------------------------------

def extract_answer(convs: list) -> Optional[str]:
    """
    Extract the final answer from the NTC SFT trajectory.
    Looks for the last `cat solution.txt` or `cat /app/solution.txt` terminal output.
    """
    for turn in reversed(convs):
        if turn['role'] != 'user':
            continue
        content = turn['content']

        # Pattern: cat [/app/]solution.txt\n<answer>\nroot@
        patterns = [
            r'cat (?:/app/)?solution\.txt\s*\n(.*?)(?=\nroot@)',
            r'cat -A (?:/app/)?solution\.txt\s*\n(.*?)\$\s*\n',
        ]
        for pat in patterns:
            m = re.search(pat, content, re.DOTALL)
            if m:
                answer = m.group(1).strip()
                # Remove trailing $ from cat -A output
                answer = answer.rstrip('$').strip()
                # Reject if it looks like a shell prompt or error
                if (answer
                        and not answer.startswith('root@')
                        and not answer.startswith('cat:')
                        and len(answer) < 500):
                    return answer
    return None


def extract_problem(first_user_content: str) -> Optional[str]:
    """Extract the problem text from the first user turn."""
    m = re.search(
        r'Task Description:\s*\n(.*?)\n\nPlease place your final answer',
        first_user_content,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Tarball manipulation
# ---------------------------------------------------------------------------

def repack_task(task_binary: bytes, answer: str) -> bytes:
    """Add test files and solve.sh to the task tarball."""
    # Read existing tarball
    with gzip.open(io.BytesIO(task_binary)) as gz_in:
        with tarfile.open(fileobj=gz_in) as tar_in:
            members = {}
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

            def add_file(name: str, content: str, mode: int = 0o644):
                data = content.encode('utf-8')
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                info.mode = mode
                info.mtime = 0
                tar_out.addfile(info, io.BytesIO(data))

            # Add tests/
            add_file('tests/test.sh', _TEST_SH, mode=0o755)
            add_file('tests/test_state.py', _TEST_STATE_PY)
            add_file('tests/expected.txt', answer + '\n')

            # Add solution/
            solve_content = _SOLVE_SH_TEMPLATE.format(answer=answer)
            add_file('solution/solve.sh', solve_content, mode=0o755)

    return out_buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Patch nemotron math sandboxes with verifiers')
    parser.add_argument('--trajectory-parquet', required=True,
                        help='Path to NTC math.parquet trajectory file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for patched dataset')
    parser.add_argument('--sandbox-repo', default='penfever/nemotron-tc-adapters-math-sandboxes',
                        help='HuggingFace repo ID for the sandbox dataset')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of tasks to process (for testing)')
    parser.add_argument('--upload-to', default=None,
                        help='HuggingFace repo ID to upload patched dataset')
    parser.add_argument('--hf-token', default=None,
                        help='HuggingFace token for upload')
    args = parser.parse_args()

    # Load sandbox dataset
    print(f'Loading sandbox dataset: {args.sandbox_repo}')
    ds = load_dataset(args.sandbox_repo, split='train')
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f'Total tasks: {len(ds)}')

    # Build trajectory index: row_index -> answer
    # The sandbox and NTC parquet have the same number of rows in the same order
    print(f'Loading trajectory index from {args.trajectory_parquet}...')
    pf = pq.ParquetFile(args.trajectory_parquet)
    total_traj = pf.metadata.num_rows
    print(f'Total trajectories: {total_traj}')

    # We need to build a mapping from sandbox row index to answer
    # Strategy: scan NTC parquet row by row, match by problem text
    # Since both have 162692 rows, try direct index mapping first,
    # then fall back to text matching.

    # First, build a problem_text -> answer index from NTC parquet
    print('Building problem -> answer index from trajectories...')
    problem_to_answer: dict[str, str] = {}
    batch_size = 5000
    processed = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pydict()
        for i in range(len(df['task'])):
            convs = df['conversations'][i]
            if not convs:
                continue
            first_user = convs[0]['content']
            problem = extract_problem(first_user)
            if not problem:
                continue
            answer = extract_answer(convs)
            if answer:
                problem_to_answer[problem] = answer
        processed += len(df['task'])
        if processed % 20000 == 0:
            print(f'  Indexed {processed}/{total_traj} trajectories, '
                  f'{len(problem_to_answer)} with answers')
        if args.limit and processed >= args.limit * 3:
            break

    print(f'Index built: {len(problem_to_answer)} problems with answers')

    # Patch tasks
    stats = {
        'total': 0,
        'patched': 0,
        'no_oracle': 0,
        'error': 0,
    }
    patched_rows = []

    for i, row in enumerate(ds):
        stats['total'] += 1
        path = row['path']
        task_binary = bytes(row['task_binary'])

        # Extract problem text from instruction.md
        try:
            with gzip.open(io.BytesIO(task_binary)) as gz:
                with tarfile.open(fileobj=gz) as tar:
                    f = tar.extractfile('instruction.md')
                    instruction = f.read().decode('utf-8')
        except Exception as e:
            stats['error'] += 1
            continue

        # Remove the "Please place your final answer..." suffix
        problem = re.sub(
            r'\nPlease place your final answer.*',
            '',
            instruction,
            flags=re.DOTALL,
        ).strip()

        # Look up answer
        answer = problem_to_answer.get(problem)
        if not answer:
            stats['no_oracle'] += 1
            continue

        # Repack with verifier files
        try:
            new_binary = repack_task(task_binary, answer)
        except Exception as e:
            stats['error'] += 1
            print(f'  [{path}] Error repacking: {e}')
            continue

        patched_rows.append({'path': path, 'task_binary': new_binary})
        stats['patched'] += 1

        if i % 5000 == 0 and i > 0:
            print(f'  Progress: {i}/{len(ds)} (patched={stats["patched"]})')

    print(f'\nStats: {stats}')

    # Save as HuggingFace dataset
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
