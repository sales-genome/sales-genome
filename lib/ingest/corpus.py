"""Local corpus source for Stage 4 extraction — replaces web fetching.

Reads the pre-scraped 247 GB corpus on the GPU server (LOCAL_CORPUS_DIR). Layout
(see docs/local_gpu_port_plan.md / memory):

    <root>/<domain>/site.json                      # manifest + per-page metadata
    <root>/<domain>/pages/<url-hash>/index.html    # raw HTML (only for ok pages)

`site.json.company.domain` is the authoritative vendor identity, so we **group
folders by company.domain** — this attributes explicitly-crawled subdomain folders
(e.g. careers.febe.fujifilm.com -> fujifilm.com) to the right vendor, which is how
subdomain case-study/product pages get picked up at runtime.

This module only reads/normalizes the corpus; classification and extraction live
in lib/ingest/classify.py and scripts/ingest/extract_corpus.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from lib.ingest.url_sources import domain_to_slug, normalize_domain, registrable_domain

MAX_CONTENT_CHARS = 30_000
# Root-level files that are not vendor folders.
_PIPELINE_FILES = {"scraper.log", "scraper_progress.json", "hf_uploaded.json"}


@dataclass
class PageRef:
    url: str
    hash: str
    html_path: Path
    title: str = ""
    description: str = ""
    text: str = ""          # pre-extracted plain text from site.json (has boilerplate)
    word_count: int = 0
    external_links: list[str] = field(default_factory=list)
    internal_links: list[str] = field(default_factory=list)


@dataclass
class VendorCorpus:
    domain: str                       # authoritative company.domain
    company: dict                     # site.json "company" block
    folders: list[Path]               # all crawl folders belonging to this vendor

    @property
    def slug(self) -> str:
        return domain_to_slug(self.domain)

    @property
    def name(self) -> str:
        return (self.company.get("company_name") or self.slug).strip()

    def iter_pages(self):
        """Yield a PageRef for every OK page (across all folders) whose HTML exists."""
        for folder in self.folders:
            site = _read_site_json(folder)
            if not site:
                continue
            for p in site.get("pages", []):
                if not p.get("ok"):
                    continue
                h = p.get("hash")
                if not h:
                    continue
                html_path = folder / "pages" / h / "index.html"
                if not html_path.exists():
                    continue
                links = p.get("links") or {}
                yield PageRef(
                    url=p.get("url", ""),
                    hash=h,
                    html_path=html_path,
                    title=p.get("title") or "",
                    description=p.get("description") or "",
                    text=p.get("text") or "",
                    word_count=int(p.get("word_count") or 0),
                    external_links=_link_hrefs(links.get("external")),
                    internal_links=_link_hrefs(links.get("internal")),
                )


def _link_hrefs(items) -> list[str]:
    """Href strings from a site.json link list — items may be {'href','text'}
    dicts (this corpus) or bare strings; empties dropped."""
    out: list[str] = []
    for e in items or []:
        h = e.get("href") if isinstance(e, dict) else e
        if h:
            out.append(h)
    return out


def _read_site_json(folder: Path) -> dict | None:
    path = folder / "site.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(errors="ignore"))
    except Exception:
        return None


def _peek_company_domain(site_path: Path) -> str | None:
    """Cheaply read company.domain from the head of a site.json without parsing the
    whole (possibly huge) file. `company` is the first top-level key in the schema."""
    try:
        head = site_path.read_text(errors="ignore")[:8192]
    except Exception:
        return None
    m = re.search(r'"company"\s*:\s*\{[^{}]*?"domain"\s*:\s*"([^"]+)"', head, re.S)
    if not m:
        m = re.search(r'"domain"\s*:\s*"([^"]+)"', head)
    return normalize_domain(m.group(1)) if m else None


def iter_vendor_corpora(root: Path | str, only_domain: str | None = None):
    """Group crawl folders by company.domain and yield one VendorCorpus per vendor.

    `only_domain` (a company.domain) restricts to a single vendor for testing / a
    targeted re-run.
    """
    root = Path(root)
    groups: dict[str, list[Path]] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name in _PIPELINE_FILES:
            continue
        site_path = entry / "site.json"
        if not site_path.exists():
            continue  # mid-crawl or failed-stub folder without a finished manifest
        dom = _peek_company_domain(site_path) or normalize_domain(entry.name)
        if only_domain and dom != normalize_domain(only_domain):
            continue
        groups.setdefault(dom, []).append(entry)

    for dom, folders in groups.items():
        company = {}
        for folder in folders:
            site = _read_site_json(folder)
            if site and site.get("company"):
                company = site["company"]
                break
        yield VendorCorpus(domain=dom, company=company, folders=folders)


def page_markdown(page: PageRef, max_chars: int = MAX_CONTENT_CHARS) -> str | None:
    """Clean text for a page: trafilatura over the raw HTML, falling back to the
    site.json `text` field (boilerplate-laden but better than nothing)."""
    md = None
    try:
        html = page.html_path.read_text(errors="ignore")
    except Exception:
        html = None
    if html:
        import trafilatura

        md = trafilatura.extract(
            html, url=page.url, output_format="txt",
            include_tables=True, include_links=False, favor_recall=True,
        )
    if not md or len(md) < 100:
        md = page.text or ""
    md = md.strip()
    if not md:
        return None
    return md[:max_chars]


def sibling_subdomains(vc: VendorCorpus) -> dict[str, int]:
    """Uncrawled sibling subdomains referenced in external links: hosts under the
    same registrable domain as company.domain that are NOT themselves crawled
    folders. Returns {host: link_count} for subdomain-gap targeting."""
    reg = registrable_domain(vc.domain)
    crawled = {normalize_domain(f.name) for f in vc.folders} | {normalize_domain(vc.domain)}
    counts: dict[str, int] = {}
    for page in vc.iter_pages():
        for link in page.external_links:
            host = normalize_domain(link)
            if not host or host in crawled:
                continue
            if registrable_domain(host) == reg and host != reg:
                counts[host] = counts.get(host, 0) + 1
    return counts
