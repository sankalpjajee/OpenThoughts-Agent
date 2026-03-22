#!/usr/bin/env python3
"""
Diagnostic: attempt to build one r2egym task via Daytona and print the full error.

Usage:
    DAYTONA_API_KEY=... HF_TOKEN=... python3 scripts/daytona/debug_r2egym_build.py
"""
import asyncio
import gzip
import io
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets import load_dataset

REPO_ID = "SankalpKJ/r2egym-patched-sample"

print(f"Loading dataset: {REPO_ID}")
ds = load_dataset(REPO_ID, split="train")
row = ds[0]
task_binary = bytes(row["task_binary"])

# Extract to temp dir
tmpdir = Path(tempfile.mkdtemp(prefix="r2egym_debug_"))
with gzip.open(io.BytesIO(task_binary)) as gz:
    with tarfile.open(fileobj=gz) as tar:
        tar.extractall(str(tmpdir))

dockerfile = tmpdir / "environment" / "Dockerfile"
print(f"\nDockerfile path: {dockerfile}")
print(f"Dockerfile content:\n{dockerfile.read_text()}")
print(f"\nDirectory listing:")
for p in sorted(tmpdir.rglob("*")):
    print(f"  {p.relative_to(tmpdir)}")

# Try Daytona build
from daytona import AsyncDaytona, CreateSandboxFromImageParams, Image, Resources


async def try_build() -> bool:
    resources = Resources(cpu=2, memory=4, disk=20, gpu=0)
    params = CreateSandboxFromImageParams(
        image=Image.from_dockerfile(dockerfile),
        auto_delete_interval=0,
        resources=resources,
    )
    daytona = AsyncDaytona()
    sandbox = None
    try:
        print("\nAttempting Daytona build (timeout=300s)...")
        sandbox = await daytona.create(params=params, timeout=300)
        print("SUCCESS!")
        return True
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return False
    finally:
        try:
            if sandbox is not None:
                await sandbox.delete()
        finally:
            await daytona.close()


asyncio.run(try_build())
