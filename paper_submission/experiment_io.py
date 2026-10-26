"""Offline source search and explicit local inference for recovered experiments."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os
from pathlib import Path
import re

from paper_submission.artifacts import lines


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    date: str | None = None
    position: int = 0


def require_local():
    if os.environ.get("SUBMISSION_ALLOW_LOCAL_LLM") != "1":
        raise RuntimeError("Local model execution requires SUBMISSION_ALLOW_LOCAL_LLM=1.")
    if not os.environ.get("SUBMISSION_LOCAL_MODEL"):
        raise RuntimeError("Set SUBMISSION_LOCAL_MODEL explicitly.")


def local_tool_call(**kwargs):
    from lib.providers import local_tool_call as call
    kwargs.pop("description", None)
    return call(**kwargs)


class LocalCorpus:
    """Search caller-supplied documents without fetching URLs or paid services."""

    def __init__(self, records):
        self.records = []
        seen = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Corpus records must be JSON objects.")
            if not all(isinstance(record.get(key), str) and record[key].strip()
                       for key in ("id", "url", "title", "text")):
                raise ValueError("Corpus needs nonempty id, url, title and text.")
            if record["id"] in seen:
                raise ValueError("Duplicate corpus ID: " + record["id"])
            seen.add(record["id"])
            if record.get("published_date") is not None:
                date.fromisoformat(record["published_date"])
            self.records.append(dict(record))

    def search(self, query, *, cutoff=None, limit=10, exclude_urls=()):
        if not isinstance(query, str) or not query.strip() or limit <= 0:
            raise ValueError("Search needs a nonempty query and positive limit.")
        cutoff_date = date.fromisoformat(cutoff) if cutoff else None
        terms = set(re.findall(r"\w+", query.casefold()))
        excluded = set(exclude_urls)
        ranked = []
        for record in self.records:
            if record["url"] in excluded:
                continue
            published = record.get("published_date")
            if cutoff_date is not None and (
                not published or date.fromisoformat(published) > cutoff_date
            ):
                continue
            words = set(re.findall(r"\w+", (record["title"] + " " + record["text"]).casefold()))
            score = len(terms & words)
            if score:
                ranked.append((score, record["id"], record))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        return [dict(record) for _, _, record in ranked[:limit]]


def serp(query, *, provider=None, max_results=10, use_cache=True):
    """Legacy search interface; provider labels never enable a network search."""
    path = os.environ.get("SUBMISSION_SEARCH_CORPUS")
    if not path:
        raise RuntimeError(
            "Set SUBMISSION_SEARCH_CORPUS to an authorized JSONL corpus. "
            "External search providers are not enabled by a legacy provider label."
        )
    corpus = LocalCorpus(lines(Path(path)))
    return [
        SearchResult(row["title"], row["url"], row["text"], row.get("published_date"), i)
        for i, row in enumerate(corpus.search(query, limit=max_results), 1)
    ]
