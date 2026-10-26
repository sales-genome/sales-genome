"""Minimal stage logger. Prints timestamped lines to stdout; stages call log()."""

from __future__ import annotations

import sys
from datetime import datetime, timezone


def log(stage: str, msg: str, **fields: object) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"[{ts}] [{stage}] {msg}" + (f"  {extra}" if extra else "")
    print(line, file=sys.stderr, flush=True)
