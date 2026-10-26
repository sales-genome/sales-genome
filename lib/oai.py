"""Local-only adapter for recovered hierarchy-induction scripts."""
from lib.llm import tool_call as _call


class _Tracker:
    usd = 0.0

    def report(self):
        return {"paid_calls_enabled": False, "historical_costs_reproduced": False}


tracker = _Tracker()


def tool_call(prompt, tool_name, schema, max_tokens=4096, **kwargs):
    return _call(
        system="Follow the supplied induction and grouping instructions.",
        user=prompt, tool_name=tool_name, schema=schema, max_tokens=max_tokens,
    )
