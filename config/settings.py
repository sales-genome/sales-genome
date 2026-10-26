"""Submission-local settings; no inherited service registry or credentials."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CORPUS_DIR = os.environ.get("LOCAL_CORPUS_DIR", "")
EXTRACT_CONCURRENCY = 1
LLM_BACKEND = "local"
BUDGET_USD_CAP = 0.0
SONNET_INPUT_USD_PER_MTOK = SONNET_OUTPUT_USD_PER_MTOK = 0.0
OPENAI_INPUT_USD_PER_MTOK = OPENAI_OUTPUT_USD_PER_MTOK = 0.0
SPEND_FILE = ROOT / "outputs" / "local-usage.json"
