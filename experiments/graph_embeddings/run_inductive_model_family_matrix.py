#!/usr/bin/env python3
"""Distribute inductive model-family jobs across GPUs 4-7.

The relation-aware families are directional one-hop aggregation over
pre-aggregated relation blocks, not transductive symmetric graph propagation.
This runner only launches jobs when executed; importing it is side-effect free.
"""

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
SCRIPT = (
    ROOT
    / "experiments"
    / "graph_embeddings"
    / "inductive_model_family.py"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "graph_embeddings"
    / "results"
    / "inductive_model_family"
)
DEFAULT_FAMILIES = (
    "linear",
    "mlp",
    "relation_sage",
    "relation_attention",
)
DEFAULT_VENDOR_FEATURES = ("d100", "unpruned", "inverse_degree")
DEFAULT_SLICES = ("customer_holdout", "double_holdout")
DEFAULT_VENDOR_BLOCKS = ("product:1024", "capability:1024")
DEFAULT_CUSTOMER_BLOCKS = (
    "pain_semantic:1024",
    "pain_subtheme:292",
    "audience:50",
)
MODEL_INTERPRETATION = (
    "Directional one-hop message aggregation over pre-aggregated relation "
    "blocks; not the old transductive symmetric R-GCN."
)


def _run_name(job: dict[str, Any]) -> str:
    return (
        f"{job['family']}__{job['vendor_feature']}"
        f"__{job['customer_feature']}__{job['evaluation_slice']}"
        f"__seed{job['seed']}"
    )


def _build_command(
    *,
    args: argparse.Namespace,
    job: dict[str, Any],
    gpu: int,
) -> list[str]:
    command = [
        args.python,
        str(SCRIPT),
        "train",
        "--source-cache",
        str(args.source_cache),
        "--evaluation-cache",
        str(args.evaluation_cache),
        "--output-dir",
        str(args.output_dir),
        "--vendor-feature",
        job["vendor_feature"],
        "--customer-feature",
        job["customer_feature"],
        "--vendor-block-dims",
        *args.vendor_block_dims,
        "--customer-block-dims",
        *args.customer_block_dims,
        "--evaluation-slice",
        job["evaluation_slice"],
        "--family",
        job["family"],
        "--device",
        f"cuda:{gpu}",
        "--seed",
        str(job["seed"]),
        "--hidden-dim",
        str(args.hidden_dim),
        "--output-dim",
        str(args.output_dim),
        "--dropout",
        str(args.dropout),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--negatives",
        str(args.negatives),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--patience",
        str(args.patience),
        "--validation-fraction",
        str(args.validation_fraction),
        "--temperature",
        str(args.temperature),
    ]
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--evaluation-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpus", nargs="+", type=int, default=[4, 5, 6, 7])
    parser.add_argument(
        "--families",
        nargs="+",
        choices=DEFAULT_FAMILIES,
        default=list(DEFAULT_FAMILIES),
    )
    parser.add_argument(
        "--vendor-features",
        nargs="+",
        default=list(DEFAULT_VENDOR_FEATURES),
    )
    parser.add_argument("--customer-feature", default="fusion")
    parser.add_argument(
        "--vendor-block-dims",
        nargs="+",
        default=list(DEFAULT_VENDOR_BLOCKS),
        metavar="RELATION:DIM",
    )
    parser.add_argument(
        "--customer-block-dims",
        nargs="+",
        default=list(DEFAULT_CUSTOMER_BLOCKS),
        metavar="RELATION:DIM",
    )
    parser.add_argument(
        "--evaluation-slices",
        nargs="+",
        choices=DEFAULT_SLICES,
        default=list(DEFAULT_SLICES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--negatives", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.source_cache.exists():
        raise SystemExit(f"Source cache does not exist: {args.source_cache}")
    if not args.evaluation_cache.exists():
        raise SystemExit(
            f"Evaluation cache does not exist: {args.evaluation_cache}"
        )
    if not args.gpus:
        raise SystemExit("At least one GPU is required")
    if len(set(args.gpus)) != len(args.gpus):
        raise SystemExit("GPU list contains duplicates")
    if any(gpu not in {4, 5, 6, 7} for gpu in args.gpus):
        raise SystemExit("This matrix is restricted to GPUs 4-7")

    jobs = deque(
        {
            "family": family,
            "vendor_feature": vendor_feature,
            "customer_feature": args.customer_feature,
            "evaluation_slice": evaluation_slice,
            "seed": seed,
        }
        for seed in args.seeds
        for evaluation_slice in args.evaluation_slices
        for vendor_feature in args.vendor_features
        for family in args.families
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    available_gpus = deque(args.gpus)
    running: dict[int, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    while jobs or running:
        while jobs and available_gpus:
            job = jobs.popleft()
            gpu = available_gpus.popleft()
            run_name = _run_name(job)
            log_path = log_dir / f"{run_name}.log"
            command = _build_command(args=args, job=job, gpu=gpu)
            log_handle = log_path.open("w", encoding="utf-8")
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
                "started_at": time.monotonic(),
            }
            print(
                f"Started {run_name} on cuda:{gpu} pid={process.pid}",
                flush=True,
            )

        if running:
            time.sleep(2)
        for pid, state in list(running.items()):
            return_code = state["process"].poll()
            if return_code is None:
                continue
            state["log_handle"].close()
            elapsed = time.monotonic() - state["started_at"]
            completed.append(
                {
                    key: value
                    for key, value in state.items()
                    if key not in {"log_handle", "process", "started_at"}
                }
                | {
                    "return_code": return_code,
                    "elapsed_seconds": round(elapsed, 3),
                }
            )
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
        "model_interpretation": MODEL_INTERPRETATION,
        "source_cache": str(args.source_cache),
        "evaluation_cache": str(args.evaluation_cache),
        "output_dir": str(args.output_dir),
        "gpus": args.gpus,
        "vendor_block_dims": args.vendor_block_dims,
        "customer_block_dims": args.customer_block_dims,
        "jobs": completed,
    }
    manifest_path = args.output_dir / "matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    failures = [job for job in completed if job["return_code"] != 0]
    print(f"Wrote {manifest_path}")
    if failures:
        raise SystemExit(f"{len(failures)} model-family jobs failed")


if __name__ == "__main__":
    main()
