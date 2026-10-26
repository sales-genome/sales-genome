# Executable construction, composition, induction and canonicalization

This is a versioned, source-bound implementation of the manuscript's construction
method, **not a claim to have rerun historical experiments or recovered their exact
model responses**. It builds graphs from supplied pages; it does not modify the
packaged 20%-of-customer-groups sample. Historical experiment reconstruction and
downstream task entry points are separate.

## Run the complete pipeline offline

From the submission root, choose output directories that do not already exist:

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p 'test_pipeline.py' -v
PYTHONDONTWRITEBYTECODE=1 python -m paper_submission.pipeline demo --out outputs/my-construction-demo
PYTHONDONTWRITEBYTECODE=1 python -m paper_submission.pipeline replay-demo \
  --demo outputs/my-construction-demo --out outputs/my-construction-replay
```

The synthetic example contains 17 invented source records, four sellers, four
customers, explicitly named products, a generic noncustomer product-page problem,
unattributed capabilities, audience attributes and four rhetorical fingerprints.
It executes extraction, shared-ID composition, hierarchy proposal, full
classification, coverage/balance refinement, semantic grouping, recursive
cross-batch reconciliation, and all three graph projections in **all four
dimensions**: pain, capability, audience and strategy.

The demo's authored model responses are deliberately separate from the algorithms.
The fixture backend recognizes only its invented descriptions and fails on unknown
ones. It supplies structured source/extraction/semantic decisions, **not graph
overlays**. `synthetic-recording.json` binds every response to the complete request
and response-schema hashes. Replay uses the strict `RecordedModel` backend and
checks graph-file equality. This establishes executable algorithmic behavior, not
the accuracy of a learned model or the full paper's reported metrics.

## Actual local-model execution

Use an already-running, user-managed OpenAI-compatible structured-tool endpoint.
The transport accepts only loopback addresses, disables environment proxies and
redirects, and requires explicit opt-in. These commands do not start a server,
download a model, train, or fetch web pages.

```sh
export SUBMISSION_ALLOW_LOCAL_LLM=1
export SUBMISSION_LOCAL_BASE_URL=http://localhost:8000/v1
export SUBMISSION_LOCAL_MODEL=your-public-served-model-name
PYTHONDONTWRITEBYTECODE=1 python -m paper_submission.pipeline run \
  --backend local --pages inputs/pages.jsonl --aliases inputs/approved-aliases.json \
  --settings inputs/induction-settings.json --attempts 2 --max-tokens 12000 \
  --out outputs/my-local-construction
```

Omit `--aliases` or `--settings` when unnecessary. `LocalModel` makes genuine
schema-bound extraction, classification, hierarchy and semantic grouping calls.
No live inference was used to assemble or validate this package. Endpoint choice
and model suitability remain the operator's responsibility. Use a nonprivate
served model name rather than an absolute checkpoint path.

Each input JSONL record has:

```json
{
  "vendor_slug": "example-seller",
  "vendor_name": "Example Seller",
  "url": "https://example.org/customer-story",
  "page_type": "case_study",
  "published_date": "2020-01-01",
  "content": "User-provided source text held in memory, not persisted by real runs."
}
```

`published_date` and `vendor_name` are optional. Accepted genres are `case_study`,
`customer_story`, `product`, `industry`, `documentation`, `press_release`, `other`.
Dates are source metadata, **not customer adoption dates**. Caller-provided graph
IDs and unknown input fields are rejected. Public source URLs must not contain
credentials, private hosts or credential query parameters. The CLI does not crawl.

Optional approved aliases:

```json
{
  "customer_crosslinks": [{
    "name": "Exact Source Company Alias",
    "customer_id": "customer:approved-company",
    "canonical_name": "Approved Company",
    "approved": true
  }],
  "product_aliases": [{
    "vendor_slug": "example-seller",
    "name": "Exact Source Product Alias",
    "canonical_name": "Approved Product",
    "approved": true
  }]
}
```

Only exact normalized aliases are applied. Canonical customer names are recognized
as the same explicitly approved identity. Product aliases remain seller-scoped;
approved chains are resolved and cycles/conflicts rejected. Unapproved or fuzzy
identity merges are never inferred. A slug/hash collision is an error, not a merge.

An optional settings file contains any `InductionSettings` fields, for example:

```json
{
  "seed": "my-reproducible-run",
  "sample_size": 200,
  "classification_batch": 100,
  "group_batch": 100,
  "reconciliation_batch": 50,
  "max_rounds": 5,
  "coverage_target": 0.95,
  "max_leaf_fraction": 0.45,
  "min_split_size": 3,
  "max_depth": 6,
  "explicit_singleton_fallback": false
}
```

## Stage map and invariants

1. **`construction.extract_pages`** reuses the original genre-specific v2 prompts
   and unified extraction schema. Extra support contracts require exact source
   substrings for every product, capability, pain, pair, audience value and outcome.
   Named customers must occur in their evidence and can be case subjects only on
   case pages. Customer pains/attributes require an actual supported subject.
   Each outcome quote is assessed for persuasive mechanism, evidence, tone, emotion
   and warrant; its business subject is stored separately and never used for
   strategy matching. Missing/duplicate/unknown supports and unsupported pair
   endpoints are errors. Quotes are replaced by hashed source anchors before output.

2. **`composition.compose`** combines source observations through shared seller,
   seller-scoped product and exact/approved customer IDs. Capabilities are
   seller-scoped exact phrases; pains are shared exact phrases; audience atoms
   include their attribute type. A capability links to a named product only through
   an individual evidence-backed binding. Multiple co-mentioned products do **not**
   induce a Cartesian expansion. Unbound capabilities go to a seller-specific
   `product:<seller>:unspecified`. General problems are retained as raw source
   claims but never attributed as an unsupported customer's pain.

3. **`induction.induce`** learns a full nested hierarchy per dimension with an
   explicitly selected model. It reuses the recovered taxonomy's dimension
   definitions and chisel `SubTheme`/leaf traversal, with new strict contracts:
   - Seeded samples prioritize uncovered and overloaded items after the first round.
   - Every atom is classified once per round to a known leaf index or explicit `-1`.
   - Uncovered examples and overloaded leaf examples drive full-tree refinement.
   - Coverage is assigned/total; overload is a leaf's assigned-population fraction,
     applied only when its count reaches `min_split_size`.
   - Stop on both targets, no schema progress, or the round bound. Status and every
     round's actual coverage/load/receipt are persisted.
   - Group equivalent meanings **within each narrow slot** in bounded batches.
     Recursive reconciliation compares prototypes across batch boundaries; a
     changed candidate is re-compared against the pool, with strict finite
     membership growth. Every raw atom remains exactly once in its slot partition.
   - All model references are local integer indices, never model-chosen graph IDs.
     Reject partial/duplicate/unknown indices, malformed trees, dimension
     mismatches, oversized depth, invalid labels and mixed audience attribute types.
     Pain compares actual problem/actor/process/impact; capability compares enabled
     function; audience compares attribute type/value/scope; strategy compares
     style only. Strategy payloads contain neither business subject nor source
     customer/product identities.

4. **`canonicalization.project`** maps every native raw edge through π, using
   identity for unmapped atoms, and aggregates only the contributing source claims.
   `canonical` retains company/product identity nodes, canonical concepts, hierarchy
   and unmapped raw atoms. `hybrid` preserves raw native claims unchanged and adds
   canonical definitions, `instance_of` mappings and `is_a` hierarchy. Endpoint
   types, dimensions, memberships, source/claim hashes, cycles, and preservation
   of native claim IDs are checked. Induction cannot add native company claims.

Model/schema failures retry and are recorded individually. By default the run
fails, rather than quietly converting missing responses into singleton "success".
`explicit_singleton_fallback=true` is an opt-in emergency behavior: that dimension's
fallback concepts and mappings are labeled `explicit_singleton_fallback`, with
status `explicit_fallback_not_model_success`. Unassigned atoms at stopping remain
visible and retain their native claims. `manifest.json.semantic_success` is false
when any dimension uses fallback or misses its coverage/balance targets.

## Stable Python API

```python
from paper_submission.construction import ModelSession, LocalModel, RecordedModel, extract_pages
from paper_submission.composition import compose
from paper_submission.induction import InductionSettings, induce
from paper_submission.canonicalization import project
from paper_submission.pipeline import run_pipeline

session = ModelSession(model, attempts=2)  # LocalModel(...) or RecordedModel(...)
observations = extract_pages(pages, session)
raw = compose(observations, aliases=approved_aliases)
learned = induce(raw, session, settings=InductionSettings())
views = project(raw, learned)
# Or persist the same stage sequence with:
result = run_pipeline(pages, "outputs/a-new-run", model,
                      aliases=approved_aliases, settings=InductionSettings())
```

The default `induce` runs all four dimensions. A dimension can be run directly
with `learn_dimension(raw, dimension, session, settings)`. Input observations are
source-bound extraction claims, **not independently verified facts**. Their
checksums detect changes relative to the recorded observation; they are not
cryptographic signatures or substitutes for source review.

## Files and provenance contract

Every successful run writes:

```text
raw/nodes.jsonl                 raw/edges.jsonl
canonical/nodes.jsonl           canonical/edges.jsonl
hybrid/nodes.jsonl              hybrid/edges.jsonl
observations.jsonl              sources.jsonl
mappings.jsonl                  unmapped.jsonl
induction.json                  model-receipts.json
graph-validation.json           manifest.json
```

Nodes use **`id`, `type`, `name`**; semantic nodes additionally have `dimension`
and `level` (`raw`, `canonical`, `category`). Raw pain/capability types are
`pain_point`/`capability`; canonical types are `pain_canonical`/`cap_canonical`.
Audience/strategy types retain their names and are distinguished by `level`.
Category types are `<pain|cap|audience|strategy>_domain` or `_subtheme`.

Edges use **`src`, `type`, `dst`**. Native relation types are `offers`, `provides`,
`addresses`, `experiences`, `has_case_study`, `in_audience`, `asserts`, `cites`,
`targets`. Semantic definitions use `instance_of` and `is_a`.

Each native business edge contains:

```json
{
  "src": "product:example-seller:...",
  "type": "provides",
  "dst": "canonical:capability:...",
  "source_records": [{"vendor_slug": "example-seller", "url": "https://example.org/source"}],
  "claims": [{
    "id": "claim:...",
    "role": "explicit_individual_product_capability",
    "raw_src": "product:example-seller:...",
    "raw_type": "provides",
    "raw_dst": "cap:example-seller:...",
    "evidence": {
      "source_id": "source:...",
      "vendor_slug": "example-seller",
      "url": "https://example.org/source",
      "content_sha256": "...",
      "start": 20,
      "end": 75,
      "quote_sha256": "...",
      "evidence_type": "capability_products"
    }
  }],
  "contributing_raw_edges": [{
    "src": "product:example-seller:...",
    "type": "provides",
    "dst": "cap:example-seller:...",
    "claim_ids": ["claim:..."]
  }]
}
```

`contributing_raw_edges` is canonical-only. `source_records` is the exact deduplicated
vendor/URL index for **that edge**, not all sources of a shared endpoint. Each
claim retains its original source, raw endpoints and source support; downstream
source masks should filter individual claims before reconstructing the index.
Source masking does not retroactively relearn an ontology: for strict temporal
or inductive construction, run extraction/induction using only eligible input
pages, rather than interpreting a full-input ontology as a historical snapshot.
`sources.jsonl` distinguishes source versions by content hash and retains genre
and supplied publication-date metadata. `raw_ids` on canonical concepts and
`mappings.jsonl` provide atom lineage. Definition edges are learned semantic
relations, not new source-observed company relationships.

Offsets are Python Unicode-character offsets into supplied `content`; hashes are
SHA-256 of UTF-8 encoded text. Full source bodies, quoted support passages, private
paths, endpoints, credentials, and arbitrary provider usage fields are not
persisted by real runs. Compact extracted concept/claim descriptions remain part
of the graph. Receipts contain operation, settings, request/schema/response hashes,
attempt status and safe numeric token counts, not raw model transcripts.

The demo additionally writes clearly labeled **synthetic-only** inputs/recording
for offline replay. Do not generalize that input/transcript export to real pages.
No output directory is overwritten. Failed runs retain a redacted `failure.json`
and receipts, not a success manifest. The per-run manifest hashes code and emitted
artifacts; it is not the final package seal.
