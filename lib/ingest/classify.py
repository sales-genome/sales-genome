"""LLM page classifier for the local corpus (Stage 4).

The corpus is a broad BFS crawl, so most pages are not case studies. Since the
local LLM is effectively free, we classify every candidate page with it (not
regex) over a rich taxonomy — the goal is completeness: capture every page that
carries graph-relevant signal, not just case studies and product pages.

`page_class` (fine-grained) drives which extraction prompt to use and whether to
extract at all; a coarse `page_type` (case_study | product) is derived for
backward compatibility with the existing graph builders.

Calls go through providers.local_tool_call directly, so this module structurally
never touches Azure.
"""

from __future__ import annotations

from lib import providers
from lib.ingest.url_sources import classify_url_path

# Approved taxonomy (see docs/local_gpu_port_plan.md). "skip" = no extraction.
PAGE_CLASSES = [
    "case_study", "customer_story", "testimonial",
    "product", "feature", "integration", "docs",
    "solution", "industry_vertical",
    "blog", "whitepaper_report", "news_press", "webinar_event",
    "comparison_vs", "pricing", "about_company",
    "skip",
]

# Customer-centric classes use the case-study prompt; everything else uses the
# vendor-page prompt. This also sets the coarse page_type stored in extractions.
_CUSTOMER_CENTRIC = {"case_study", "customer_story", "testimonial"}

_LABEL_HELP = (
    "case_study: a named customer's deployment/success story (customer + outcomes)\n"
    "customer_story: customer narrative/profile without hard metrics\n"
    "testimonial: short customer quote(s)/review\n"
    "product: a product or its overview page\n"
    "feature: a specific feature/capability page\n"
    "integration: an integration/connector page\n"
    "docs: technical documentation / API reference\n"
    "solution: solution or use-case page (problem the vendor solves)\n"
    "industry_vertical: industry/vertical landing page\n"
    "blog: blog post / thought leadership article\n"
    "whitepaper_report: whitepaper, research report, ebook, guide\n"
    "news_press: press release / news / announcement\n"
    "webinar_event: webinar, event, conference page\n"
    "comparison_vs: competitor comparison / 'vs' page\n"
    "pricing: pricing / plans page\n"
    "about_company: about / company / leadership page\n"
    "skip: navigation, search, login, careers/jobs, legal, contact, error, or generic marketing with no extractable signal\n"
)

_SYSTEM = (
    "You classify web pages by their type based on URL, title, and a text snippet. "
    "Pick exactly one label per page from the allowed list. Favor a specific content "
    "label over 'skip' whenever the page carries any vendor/customer/product signal; "
    "reserve 'skip' for navigation, login, search, careers, legal, contact, or empty pages."
)

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "page_class": {"type": "string", "enum": PAGE_CLASSES},
                },
                "required": ["index", "page_class"],
            },
        }
    },
    "required": ["classifications"],
}


def coarse_page_type(page_class: str) -> str:
    """Map a fine-grained class to the case_study|product value the graph builders
    understand (skip has no coarse type but callers filter it out first)."""
    return "case_study" if page_class in _CUSTOMER_CENTRIC else "product"


def is_extractable(page_class: str) -> bool:
    return page_class != "skip"


def _fallback(url: str) -> str:
    """URL-path heuristic used only if an LLM batch fails outright."""
    label = classify_url_path(url)
    return "case_study" if label == "case_study" else "skip"


def classify_pages(pages, *, batch_size: int = 40, snippet_chars: int = 400,
                   max_tokens: int = 4000) -> dict[str, str]:
    """Classify PageRef-like objects (need .hash/.url/.title/.text). Returns
    {page.hash: page_class}. Every input page gets a label (missing -> 'skip')."""
    result: dict[str, str] = {}
    for i in range(0, len(pages), batch_size):
        batch = pages[i:i + batch_size]
        lines = []
        for j, p in enumerate(batch):
            snippet = (getattr(p, "text", "") or "")[:snippet_chars].replace("\n", " ")
            lines.append(f"{j}. url={p.url} | title={p.title} | text={snippet}")
        user = (
            "Allowed labels:\n" + _LABEL_HELP +
            "\nClassify each page below; return one {index, page_class} per page.\n\n" +
            "\n".join(lines)
        )
        args, _usage = providers.local_tool_call(
            system=_SYSTEM, user=user, tool_name="emit_classifications",
            schema=CLASSIFY_SCHEMA, max_tokens=max_tokens, retries=4,
        )
        if not args:
            for p in batch:
                result.setdefault(p.hash, _fallback(p.url))
            continue
        for c in args.get("classifications", []):
            idx, lbl = c.get("index"), c.get("page_class")
            if isinstance(idx, int) and 0 <= idx < len(batch) and lbl in PAGE_CLASSES:
                result[batch[idx].hash] = lbl
        for p in batch:
            result.setdefault(p.hash, "skip")
    return result
