"""Persistent USD spend counter with hard cap enforcement.

Usage:
    from lib.budget import Budget
    b = Budget.load()
    b.pre_check()                           # raises BudgetExceeded if already over
    b.record(input_tokens=500, output_tokens=200)  # persisted to .spend.json
"""

from __future__ import annotations

import fcntl
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from config.settings import (
    BUDGET_USD_CAP,
    LLM_BACKEND,
    OPENAI_INPUT_USD_PER_MTOK,
    OPENAI_OUTPUT_USD_PER_MTOK,
    SONNET_INPUT_USD_PER_MTOK,
    SONNET_OUTPUT_USD_PER_MTOK,
    SPEND_FILE,
)


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class _State:
    total_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    calls: int = 0


class Budget:
    def __init__(self, state: _State, path: Path):
        self._state = state
        self._path = path

    @classmethod
    def load(cls, path: Path = SPEND_FILE) -> "Budget":
        if path.exists():
            try:
                return cls(_State(**json.loads(path.read_text())), path)
            except Exception:
                pass
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(_State(), path)

    @property
    def total_usd(self) -> float:
        return self._state.total_usd

    @property
    def calls(self) -> int:
        return self._state.calls

    def pre_check(self) -> None:
        if self._state.total_usd >= BUDGET_USD_CAP:
            raise BudgetExceeded(
                f"Budget cap of ${BUDGET_USD_CAP:.2f} reached "
                f"(spent ${self._state.total_usd:.2f} across {self._state.calls} calls)"
            )

    def record(self, input_tokens: int, output_tokens: int) -> float:
        input_rate = SONNET_INPUT_USD_PER_MTOK
        output_rate = SONNET_OUTPUT_USD_PER_MTOK
        if LLM_BACKEND == "azure_responses":
            input_rate = OPENAI_INPUT_USD_PER_MTOK
            output_rate = OPENAI_OUTPUT_USD_PER_MTOK
        cost = (
            input_tokens * input_rate / 1_000_000
            + output_tokens * output_rate / 1_000_000
        )
        with open(self._path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                raw = f.read()
                if raw:
                    state = _State(**json.loads(raw))
                else:
                    state = _State()
                state.total_usd += cost
                state.total_input_tokens += input_tokens
                state.total_output_tokens += output_tokens
                state.calls += 1
                f.seek(0)
                f.truncate()
                f.write(json.dumps(asdict(state), indent=2))
                self._state = state
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return cost
