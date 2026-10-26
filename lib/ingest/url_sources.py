"""Vendor URL sources for the ingestion pipeline (Stage 2 of the port).

`add_vendor.py` originally discovered a vendor's pages live (Wayback timemap +
SERP). To scale to thousands of vendors we also want a **pre-supplied** source:
a cleaned manifest of `{vendor -> case-study URLs}` produced once by
`scripts/ingest/clean_url_list.py` from a bulk URL-discovery dump.

This module owns:
  * the manifest schema (`VendorURLRecord`, one JSONL row per vendor),
  * the segment-level URL classifier both the cleaner and any live path use,
  * `domain_to_slug` (a *candidate* slug — final reconciliation against existing
    graph vendors still happens at crosslink/integrate time; never treat this as
    an entity-resolution match),
  * `to_url_entries`, which yields URL dicts in the exact shape
    `add_vendor.phase2_classify` expects, pre-typed as `case_study` so no LLM
    classification is spent on them.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

# --------------------------------------------------------------------------- #
# Segment-level URL classifier.
#
# Matching path *segments* (between slashes), not raw substrings — a substring
# test over-counts badly (a "/blog/customer-experience/" URL contains the token
# "customer"). See the QC in docs/local_gpu_port_plan.md.
# --------------------------------------------------------------------------- #
_STRICT = re.compile(
    r"/(case[-_]?stud(y|ies)|customer[-_]?stor(y|ies)|success[-_]?stor(y|ies)"
    r"|client[-_]?stor(y|ies)|customers?|case[-_]?studies|testimonials?"
    r"|our[-_]?customers|customer[-_]?success)(/|$)",
    re.I,
)
_NOISE = re.compile(
    r"/(blog|news|press|press[-_]?releases?|podcasts?|events?|webinars?|products?"
    r"|pricing|about|about[-_]?us|careers?|jobs|contact|resources?|downloads?"
    r"|whitepapers?|ebooks?|reports?|solutions?|platform|features?|integrations?"
    r"|partners?|legal|privacy|terms|support|docs?|help|glossary|videos?)(/|$)",
    re.I,
)

# Country-code second-level suffixes that need the label two positions from the
# end (e.g. tesco.co.uk -> "tesco"). Small pragmatic list; not exhaustive.
_MULTI_SUFFIX = {
    "co.uk", "com.au", "co.jp", "com.br", "co.in", "com.mx", "co.za",
    "com.sg", "co.nz", "com.tr", "co.kr", "com.cn", "com.hk",
}


def classify_url_path(url: str) -> str:
    """Return 'case_study', 'noise', or 'ambiguous' from the URL path alone.

    Legacy 3-way classifier kept for the cleaner (`clean_url_list.py`) and the
    corpus classifier fallback. The discovery pipeline uses
    `classify_url_multiclass` instead (which distinguishes product/blog).
    """
    path = urlparse(url).path
    if _STRICT.search(path):
        return "case_study"
    if _NOISE.search(path):
        return "noise"
    return "ambiguous"


# --------------------------------------------------------------------------- #
# Multiclass segment classifier for the discovery pipeline.
#
# Unlike `classify_url_path` (case_study | noise | ambiguous), this keeps
# `product` and `blog` as first-class labels instead of folding them into noise,
# and recognizes non-standard case-study homes (/stories/, /resources/case-study).
# Precedence: case_study > product > blog > skip > ambiguous. case_study wins
# whenever any case-study segment is present (a customer story filed under
# /resources/ is still a case study), matching add_vendor's ordering.
# --------------------------------------------------------------------------- #
_CS_MULTI = re.compile(
    r"/(case[-_]?stud(y|ies)|customer[-_]?stor(y|ies)|success[-_]?stor(y|ies)"
    r"|client[-_]?stor(y|ies)|client[-_]?results?|customers?|our[-_]?customers"
    r"|customer[-_]?success|testimonials?|stories|impact)(/|$)",
    re.I,
)
# Only the most universal case-study URL conventions get a deterministic
# fast-path (high precision, saves an LLM call, and we never want to MISS these).
# Product vs capability vs b2c vs blog is left to the goal-aware LLM — those
# distinctions are semantic and must not be pinned to one vendor's URL taste.
# Non-HTML assets are the one other universal, vendor-agnostic skip.
_SKIP_EXT = re.compile(
    r"\.(pdf|docx?|pptx?|xlsx?|png|jpe?g|gif|svg|webp|avif|css|js|xml|json|zip|gz"
    r"|mp4|webm|mov|mp3|ico|woff2?|ttf|otf|eot)(\?|$)",
    re.I,
)


# Noise subdomains: content here is never a case study/product page regardless
# of path (job listings on careers.*, tickets on support.*, etc.).
_SKIP_HOST_PREFIX = frozenset({
    "careers", "jobs", "apply", "recruiting", "workday",
    "support", "help", "docs", "doc", "status", "kb",
    "community", "forum", "forums", "login", "auth", "sso", "id", "account",
    "accounts", "developer", "developers", "api", "app", "my", "portal",
    "investor", "investors", "ir", "shareholder", "shareholders",
    "cdn", "static", "assets", "media", "img", "images", "videos", "video",
    "press", "pressroom", "newsroom", "events", "event", "store", "shop",
})
# Substrings that mark an infra/asset host even with an unusual label
# (e.g. prd-media-assets-cdn.rtx.com).
_SKIP_HOST_SUBSTR = ("cdn", "-assets", "assets-", "static", "media-", "-media")


# Non-production hosts (QA/staging/dev mirrors) that leak into discovery, e.g.
# wwwqaac.fromtheroad.ford.com. Match the label, not a bare substring.
_STAGING_HOST = re.compile(
    r"(^|[.-])(qa\d*|uat|stg|stage|staging|dev|test|preprod|nonprod|preview|sandbox"
    r"|wwwqa\w*)([.-]|$)",
    re.I,
)


def _noise_host(host: str) -> bool:
    labels = host.split(".")
    if len(labels) >= 3 and labels[0] in _SKIP_HOST_PREFIX:
        return True
    if len(labels) >= 3 and any(s in labels[0] for s in _SKIP_HOST_SUBSTR):
        return True
    return False


def strip_staging_host(host: str) -> str:
    """Rewrite a QA/staging/dev mirror host to its production equivalent so we
    KEEP its content (case studies live on prod too) instead of dropping it.

    e.g. wwwqaac.fromtheroad.ford.com -> fromtheroad.ford.com
         staging.acme.com            -> acme.com
    Leaves the registrable domain (last two labels) intact."""
    host = (host or "").lower().split(":")[0]
    labels = host.split(".")
    if len(labels) < 3:
        return host
    kept = [lbl for lbl in labels[:-2] if not _STAGING_HOST.search(lbl)]
    return ".".join(kept + labels[-2:])


def classify_url_multiclass(url: str) -> str:
    """Fast, universal pre-filter for the discovery pipeline. Returns one of:

      'skip'       — non-HTML asset or infra host (deterministic, vendor-agnostic)
      'case_study' — matches a universal case-study URL convention (high precision)
      'ambiguous'  — everything else; the goal-aware LLM tier decides the type

    Intentionally minimal: product / capability / b2c / blog / marketing
    distinctions are semantic and are made by the LLM, never by per-vendor path
    heuristics — so this generalizes to any vendor at scale."""
    p = urlparse(url)
    host = (p.netloc or "").lower().split(":")[0]
    if _noise_host(host):
        return "skip"
    path = p.path
    if _SKIP_EXT.search(path):
        return "skip"
    if _CS_MULTI.search(path):
        return "case_study"
    return "ambiguous"


def normalize_domain(domain: str) -> str:
    """Lowercase host, drop scheme/path/port and a leading www."""
    host = (domain or "").strip().lower()
    if "//" in host:
        host = host.split("//", 1)[1]
    host = host.split("/", 1)[0].split(":", 1)[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def domain_to_slug(domain: str) -> str:
    """Candidate vendor slug from a domain (registrable name, best-effort).

    NOT an entity match — several of these will already exist in the graph under
    a different slug; crosslink/integrate reconciles that. Deterministic so the
    same domain always maps to the same work dir.
    """
    host = normalize_domain(domain)
    labels = host.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in _MULTI_SUFFIX:
        core = labels[-3]
    elif len(labels) >= 2:
        core = labels[-2]
    else:
        core = labels[0] if labels else host
    return re.sub(r"[^a-z0-9]+", "-", core).strip("-")


def registrable_domain(host: str) -> str:
    """Best-effort eTLD+1 for grouping/sibling checks (e.g. careers.x.acme.com ->
    acme.com). Uses the same small ccTLD list as domain_to_slug; not a full PSL."""
    h = normalize_domain(host)
    labels = h.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in _MULTI_SUFFIX:
        return ".".join(labels[-3:])
    if len(labels) >= 2:
        return ".".join(labels[-2:])
    return h


# --------------------------------------------------------------------------- #
# Manifest schema
# --------------------------------------------------------------------------- #
@dataclass
class VendorURLRecord:
    slug: str
    domain: str
    resolved_domain: str
    industry: str = ""
    source: str = ""
    case_study_urls: list[str] = field(default_factory=list)
    # Extensions for the multi-domain discovery pipeline. All optional so the
    # legacy cleaner output (case_study_urls only) round-trips unchanged.
    product_urls: list[str] = field(default_factory=list)
    blog_urls: list[str] = field(default_factory=list)
    b2c_urls: list[str] = field(default_factory=list)  # consumer pages, filterable
    domains: list[str] = field(default_factory=list)  # full portfolio discovered

    @property
    def n_case_study_urls(self) -> int:
        return len(self.case_study_urls)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_case_study_urls"] = self.n_case_study_urls
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VendorURLRecord":
        return cls(
            slug=d["slug"],
            domain=d.get("domain", ""),
            resolved_domain=d.get("resolved_domain", ""),
            industry=d.get("industry", ""),
            source=d.get("source", ""),
            case_study_urls=list(d.get("case_study_urls", [])),
            product_urls=list(d.get("product_urls", [])),
            blog_urls=list(d.get("blog_urls", [])),
            b2c_urls=list(d.get("b2c_urls", [])),
            domains=list(d.get("domains", [])),
        )


def to_url_entries(record: VendorURLRecord, *, include_products: bool = True,
                   include_blogs: bool = False) -> list[dict]:
    """URL dicts in the shape add_vendor.phase2_classify consumes, pre-typed so
    classification (and its LLM cost) is skipped for them. Case studies and
    products carry their page_type; blogs are opt-in (default off)."""
    entries = [{"url": u, "page_type": "case_study"} for u in record.case_study_urls]
    if include_products:
        entries += [{"url": u, "page_type": "product"} for u in record.product_urls]
    if include_blogs:
        entries += [{"url": u, "page_type": "product"} for u in record.blog_urls]
    return entries


def write_manifest(records: list[VendorURLRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")


def load_manifest(path: Path | str) -> list[VendorURLRecord]:
    path = Path(path)
    out: list[VendorURLRecord] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(VendorURLRecord.from_dict(json.loads(line)))
    return out


def manifest_by_slug(path: Path | str) -> dict[str, VendorURLRecord]:
    return {r.slug: r for r in load_manifest(path)}
