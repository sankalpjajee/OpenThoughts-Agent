#!/usr/bin/env python3
"""
Sanity check: replace solve.sh in a small sample of tasks with a "broken oracle"
that always writes a deliberately wrong answer (e.g. "WRONG_ANSWER_SANITY_CHECK"),
then run oracle validation and confirm the pass rate drops to ~0%.

This proves that the verifier is actually checking the answer and that the 100%
pass rate we see with the real oracle is genuine.

Usage (run on the cluster from the repo root):
    python3 scripts/daytona/sanity_check_broken_oracle.py \
        --repo_id SankalpKJ/nemotron-math-patched \
        --limit 10 \
        --harbor_concurrency 4 \
        --timeout 120 \
        --cpu 2 --memory_gb 4 --disk_gb 10 \
        --extract_dir /mnt/sda4T/home/jajee/nemotron_math_sanity_check \
        --token $HF_TOKEN
"""

from __future__ import annotations

import argparse
import gzip
import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset
from scripts.harbor import tasks_parquet_converter as tpc

# ---------------------------------------------------------------------------
# Broken solve.sh — always writes a deliberately wrong answer
# ---------------------------------------------------------------------------

_BROKEN_SOLVE_SH = """\
#!/usr/bin/env bash
# Broken oracle (sanity check): always writes a wrong answer
set -euo pipefail

mkdir -p /app

cat > /app/solution.txt << 'ORACLE_ANSWER_EOF'
WRONG_ANSWER_SANITY_CHECK_99999999
ORACLE_ANSWER_EOF

echo "[broken-oracle] Written deliberately wrong answer to /app/solution.txt"
cat /app/solution.txt
"""


def _replace_solve_sh(task_dir: Path) -> None:
    """Overwrite solution/solve.sh with the broken oracle."""
    solve_sh = task_dir / "solution" / "solve.sh"
    if not solve_sh.exists():
        print(f"  [warn] No solve.sh found in {task_dir.name}, skipping")
        return
    solve_sh.write_text(_BROKEN_SOLVE_SH)
    solve_sh.chmod(0o755)
    print(f"  [broken-oracle] Replaced solve.sh in {task_dir.name}")


def _extract_hf_dataset(repo_id: str, revision: Optional[str], base_dir: Path) -> Path:
    ds = load_dataset(repo_id, revision=revision)
    if hasattr(ds, "keys"):
        split_name = "train" if "train" in ds.keys() else next(iter(ds.keys()))
        split = ds[split_name]
    else:
        split = ds

    tmp_dir = Path(tempfile.mkdtemp(prefix="hf_tasks_sanity_"))
    parquet_path = tmp_dir / "tasks.parquet"
    split.to_parquet(str(parquet_path))

    base_dir.mkdir(parents=True, exist_ok=True)
    tpc.from_parquet(str(parquet_path), base=str(base_dir), on_exist="overwrite")
    return base_dir


def _discover_tasks(extracted_root: Path, limit: Optional[int] = None):
    tasks = []
    for path in sorted(extracted_root.iterdir()):
        dockerfile = path / "environment" / "Dockerfile"
        if path.is_dir() and dockerfile.exists():
            tasks.append(path)
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def parse_args(argv: Optional[Iterable[str]] = None):
    p = argparse.ArgumentParser(description="Broken-oracle sanity check for math verifier")
    p.add_argument("--repo_id", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--extract_dir", default=None)
    p.add_argument("--limit", type=int, default=10, help="Number of tasks to test (default: 10)")
    p.add_argument("--harbor_concurrency", type=int, default=4)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--cpu", type=int, default=2)
    p.add_argument("--memory_gb", type=int, default=4)
    p.add_argument("--disk_gb", type=int, default=10)
    p.add_argument("--token", default=None)
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv=None):
    args = parse_args(argv)

    if args.token:
        os.environ["HF_TOKEN"] = args.token

    extract_dir = Path(args.extract_dir) if args.extract_dir else Path(tempfile.mkdtemp(prefix="sanity_check_"))
    print(f"[sanity-check] Extracting {args.limit} tasks from {args.repo_id} to {extract_dir}")

    _extract_hf_dataset(args.repo_id, args.revision, extract_dir)
    tasks = _discover_tasks(extract_dir, limit=args.limit)
    print(f"[sanity-check] Found {len(tasks)} tasks (limited to {args.limit})")

    # Replace solve.sh with broken oracle in all tasks
    print(f"[sanity-check] Replacing solve.sh with broken oracle in all {len(tasks)} tasks...")
    for task_dir in tasks:
        _replace_solve_sh(task_dir)

    # Now run oracle validation — should get ~0% pass rate
    print(f"[sanity-check] Running oracle validation (expect ~0% pass rate)...")

    # Import harbor components (same as validate_and_upload_from_hf.py)
    try:
        from harbor.job import Job
        from harbor.models.agent.name import AgentName
        from harbor.models.environment_type import EnvironmentType
        from harbor.models.job.config import (
            AgentConfig,
            EnvironmentConfig,
            JobConfig,
            LocalDatasetConfig,
            VerifierConfig,
        )
        from harbor.orchestrators.base import OrchestratorEvent
    except ImportError as exc:
        raise SystemExit(f"Harbor not importable: {exc}") from exc

    import asyncio
    import time
    from typing import Dict

    job_config = JobConfig()
    job_config.orchestrator.n_concurrent_trials = max(1, args.harbor_concurrency)
    job_config.orchestrator.quiet = True
    job_config.agents = [
        AgentConfig(
            name=AgentName.ORACLE.value,
            override_timeout_sec=args.timeout,
        )
    ]
    job_config.environment = EnvironmentConfig(
        type=EnvironmentType.DAYTONA,
        force_build=True,
        delete=True,
        override_cpus=args.cpu or None,
        override_memory_mb=(args.memory_gb * 1024) if args.memory_gb else None,
        override_storage_mb=(args.disk_gb * 1024) if args.disk_gb else None,
    )
    job_config.verifier = VerifierConfig(override_timeout_sec=args.timeout)
    job_config.datasets = [
        LocalDatasetConfig(
            path=extract_dir,
            task_names=sorted(task.name for task in tasks),
        )
    ]

    job = Job(job_config)

    stage_status: Dict[str, bool] = {}
    success_count = 0
    failure_count = 0
    total = len(tasks)
    start = time.perf_counter()

    def _reward_is_one(rewards):
        if not rewards:
            return False
        raw_value = rewards.get("reward")
        if raw_value is None:
            return False
        try:
            return float(raw_value) == 1.0
        except (TypeError, ValueError):
            return False

    async def _progress_hook(hook_event):
        nonlocal success_count, failure_count
        trial_result = getattr(hook_event, "result", hook_event)
        task_name = getattr(hook_event, "task_name", None) or getattr(trial_result, "task_name", "")
        rewards = None
        if trial_result is not None and getattr(trial_result, "verifier_result", None) is not None:
            rewards = trial_result.verifier_result.rewards
        is_success = (
            trial_result is not None
            and getattr(trial_result, "exception_info", None) is None
            and _reward_is_one(rewards)
        )
        stage_status[task_name] = bool(is_success)
        if is_success:
            success_count += 1
        else:
            failure_count += 1
        elapsed = max(time.perf_counter() - start, 1e-6)
        print(
            f"  [{task_name}] {'PASS' if is_success else 'FAIL'} | "
            f"rewards={rewards} | "
            f"running: {success_count}P/{failure_count}F/{total}T"
        )

    job._orchestrator.add_hook(OrchestratorEvent.TRIAL_COMPLETED, _progress_hook)

    try:
        asyncio.run(job.run())
    except Exception as e:
        print(f"[sanity-check] Oracle validation error: {e}")
        raise

    passed = [t for t in tasks if stage_status.get(t.name) is True]
    failed = [t for t in tasks if stage_status.get(t.name) is not True]

    print("\n" + "=" * 60)
    print(f"[sanity-check] RESULTS (broken oracle — expect ~0% pass rate)")
    print(f"  Total tasks:  {total}")
    print(f"  Passed:       {len(passed)}  ({100*len(passed)/max(total,1):.1f}%)")
    print(f"  Failed:       {len(failed)}  ({100*len(failed)/max(total,1):.1f}%)")
    print("=" * 60)

    if len(passed) == 0:
        print("\n[sanity-check] ✅ SANITY CHECK PASSED: verifier correctly rejects wrong answers.")
        print("   The 100% pass rate with the real oracle is GENUINE.")
    else:
        print(f"\n[sanity-check] ⚠️  WARNING: {len(passed)}/{total} tasks passed with a wrong answer!")
        print("   This suggests the verifier may not be checking answers correctly.")
        for t in passed:
            print(f"   - {t.name}")


if __name__ == "__main__":
    main()
