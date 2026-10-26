#!/usr/bin/env python3
"""Run independent inductive baseline configurations across available GPUs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments" / "graph_embeddings" / "inductive_baseline.py"
DEFAULT_CACHE = ROOT / "experiments" / "graph_embeddings" / "inductive_features.pt"
DEFAULT_OUTPUT = ROOT / "experiments" / "graph_embeddings" / "results" / "inductive"

DEFAULT_CONFIGS = (
    ("d100", "audience"),
    ("d100", "semantic"),
    ("d100", "tfidf"),
    ("d100", "fusion"),
    ("inverse_degree", "semantic"),
    ("unpruned", "semantic"),
    ("inverse_degree", "fusion"),
)


def parse_config(value: str) -> tuple[str, str]:
    try:
        vendor_feature, customer_feature = value.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Configurations must use vendor_feature:customer_feature"
        ) from exc
    return vendor_feature, customer_feature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", nargs="+", type=int, default=[4, 5, 6, 7])
    parser.add_argument("--configs", nargs="*", type=parse_config, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--evaluation-protocol",
        choices=["edge_transfer", "oracle_node_holdout"],
        default="edge_transfer",
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    if not args.cache.exists():
        raise SystemExit(f"Feature cache does not exist: {args.cache}")
    if not args.gpus:
        raise SystemExit("At least one GPU is required")

    configs = args.configs or list(DEFAULT_CONFIGS)
    jobs = deque(
        {
            "vendor_feature": vendor_feature,
            "customer_feature": customer_feature,
            "seed": seed,
        }
        for seed in args.seeds
        for vendor_feature, customer_feature in configs
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    available_gpus = deque(args.gpus)
    running: dict[int, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()

    while jobs or running:
        while jobs and available_gpus:
            job = jobs.popleft()
            gpu = available_gpus.popleft()
            run_name = (
                f"{args.evaluation_protocol}__{job['vendor_feature']}"
                f"__{job['customer_feature']}"
                f"__seed{job['seed']}"
            )
            log_path = log_dir / f"{run_name}.log"
            command = [
                args.python,
                str(SCRIPT),
                "train",
                "--cache",
                str(args.cache),
                "--output-dir",
                str(args.output_dir),
                "--vendor-feature",
                job["vendor_feature"],
                "--customer-feature",
                job["customer_feature"],
                "--evaluation-protocol",
                args.evaluation_protocol,
                "--device",
                f"cuda:{gpu}",
                "--seed",
                str(job["seed"]),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
            ]
            log_handle = log_path.open("w")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running[process.pid] = {
                **job,
                "gpu": gpu,
                "run_name": run_name,
                "command": command,
                "log_path": str(log_path),
                "log_handle": log_handle,
                "process": process,
                "started_at": time.time(),
            }
            print(f"Started {run_name} on cuda:{gpu} pid={process.pid}", flush=True)

        if running:
            time.sleep(2)
        for pid, state in list(running.items()):
            return_code = state["process"].poll()
            if return_code is None:
                continue
            state["log_handle"].close()
            elapsed = time.time() - state["started_at"]
            completed.append({
                key: value
                for key, value in state.items()
                if key not in {"log_handle", "process", "started_at"}
            } | {
                "return_code": return_code,
                "elapsed_seconds": round(elapsed, 3),
            })
            available_gpus.append(state["gpu"])
            del running[pid]
            print(
                f"Finished {state['run_name']} return_code={return_code} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    manifest = {
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "cache": str(args.cache),
        "evaluation_protocol": args.evaluation_protocol,
        "gpus": args.gpus,
        "jobs": completed,
    }
    manifest_path = args.output_dir / "matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    failures = [job for job in completed if job["return_code"] != 0]
    print(f"Wrote {manifest_path}")
    if failures:
        raise SystemExit(f"{len(failures)} experiment jobs failed")


if __name__ == "__main__":
    main()
