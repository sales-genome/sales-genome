"""Stage 4 — extract the local corpus into extractions.jsonl (local LLM only).

For every vendor (folders grouped by site.json company.domain):
  1. classify each OK page with the local LLM over the rich taxonomy,
  2. for non-skip pages, clean the HTML (trafilatura) and run the schema-bound
     extraction (unified case_study_v2 schema) via the LOCAL provider,
  3. append one record per page to a per-vendor JSONL under --out-dir.

Records match the existing extractions.jsonl shape plus a fine-grained
`page_class` field. Resumable: per-vendor `.done` markers + per-page hash dedup.

Hard rule: this refuses to run unless LLM_BACKEND=local (providers.require_local),
so it can never hit Azure Foundry.

Usage:
    LLM_BACKEND=local python scripts/ingest/extract_corpus.py            # whole corpus
    LLM_BACKEND=local python scripts/ingest/extract_corpus.py --vendor datadog.com --max-pages 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings  # noqa: E402
from lib import llm, providers  # noqa: E402
from lib.ingest.classify import classify_pages, coarse_page_type, is_extractable  # noqa: E402
from lib.ingest.corpus import iter_vendor_corpora, page_markdown  # noqa: E402
from lib.logging import log as _log  # noqa: E402


def log(msg: str, **kw) -> None:
    _log("corpus-extract", msg, **kw)


ROOT = Path(__file__).resolve().parent.parent.parent
PROMPT_CS = (ROOT / "prompts" / "extract_unified_v2.md").read_text()
PROMPT_VP = (ROOT / "prompts" / "extract_vendor_page_v2.md").read_text()
SCHEMA_V2 = json.loads((ROOT / "schemas" / "unified" / "case_study_v2.json").read_text())
DEFAULT_OUT = ROOT / "data" / "extractions" / "corpus"


def _extract_page(vc, page, page_class: str) -> dict | None:
    md = page_markdown(page)
    if not md:
        return None
    coarse = coarse_page_type(page_class)
    prompt = PROMPT_CS if coarse == "case_study" else PROMPT_VP
    try:
        res = llm.extract(
            page_type=coarse, content_markdown=md,
            schema=SCHEMA_V2, prompt_template=prompt,
        )
    except Exception as e:  # noqa: BLE001 — one bad page shouldn't kill the vendor
        log(f"    extract failed {page.url}: {e}")
        return None
    return {
        "vendor_slug": vc.slug,
        "url": page.url,
        "hash": page.hash,
        "page_type": coarse,      # case_study|product — backward compatible
        "page_class": page_class,  # fine-grained taxonomy label
        "is_original": True,
        "competes_with": [],
        "data": res["data"],
        "usage": res.get("usage"),
    }


def process_vendor(vc, out_dir: Path, max_pages: int, workers: int,
                   batch: int, resume: bool) -> str:
    out_path = out_dir / f"{vc.slug}.jsonl"
    done_path = out_dir / f"{vc.slug}.done"
    if resume and done_path.exists():
        return "skip(done)"

    done_hashes: set[str] = set()
    if resume and out_path.exists():
        for line in out_path.open():
            try:
                done_hashes.add(json.loads(line)["hash"])
            except Exception:
                pass

    pages = [p for p in vc.iter_pages() if p.hash not in done_hashes]
    if max_pages:
        pages = pages[:max_pages]
    if not pages:
        done_path.write_text("")
        return "empty" if not done_hashes else "already-complete"

    labels = classify_pages(pages, batch_size=batch)
    todo = [(p, labels.get(p.hash, "skip")) for p in pages]
    todo = [(p, c) for p, c in todo if is_extractable(c)]
    log(f"  {vc.slug}: {len(pages)} pages -> {len(todo)} extractable "
        f"(skipped {len(pages) - len(todo)})")

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, out_path.open("a") as f:
        futs = [pool.submit(_extract_page, vc, p, c) for p, c in todo]
        for fut in as_completed(futs):
            rec = fut.result()
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    done_path.write_text("")
    return f"{n} extracted"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", default=settings.LOCAL_CORPUS_DIR,
                    help="Root of the local corpus (default: LOCAL_CORPUS_DIR)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--vendor", default="", help="Only this company.domain (for testing)")
    ap.add_argument("--limit", type=int, default=0, help="Max vendors (0=all)")
    ap.add_argument("--max-pages", type=int, default=0, help="Max pages/vendor (0=all)")
    ap.add_argument("--workers", type=int, default=settings.EXTRACT_CONCURRENCY)
    ap.add_argument("--classify-batch", type=int, default=40)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    providers.require_local()  # hard guard: never Azure

    if not args.corpus_dir or not Path(args.corpus_dir).exists():
        sys.exit(f"corpus dir not found: {args.corpus_dir!r} (set LOCAL_CORPUS_DIR or --corpus-dir)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log(f"Corpus extract: {args.corpus_dir} -> {args.out_dir} "
        f"(backend={providers.backend()}, model={providers.local_model()})")

    n_vendors = 0
    for vc in iter_vendor_corpora(args.corpus_dir, only_domain=args.vendor or None):
        if args.limit and n_vendors >= args.limit:
            break
        n_vendors += 1
        status = process_vendor(
            vc, args.out_dir, args.max_pages, args.workers,
            args.classify_batch, resume=not args.no_resume,
        )
        log(f"[{n_vendors}] {vc.domain} ({vc.slug}): {status}")

    print(f"\nDone. Processed {n_vendors} vendors -> {args.out_dir}")


if __name__ == "__main__":
    main()
