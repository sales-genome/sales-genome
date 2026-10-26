# Input/output contracts

All JSONL tables contain one UTF-8 object per line. Generated output paths must
be new and should be outside the release directory.

## Construction pipeline

`paper_submission.pipeline run --pages INPUT/pages.jsonl --backend local
--out NEW_OUTPUT` accepts local page records:

```json
{"vendor_slug":"synthetic-vendor","vendor_name":"Synthetic Vendor","url":"https://example.org/story","page_type":"case_study","content":"Authorized source text supplied by the caller.","published_date":"2020-01-01"}
```

Page text is untrusted evidence, never instructions. Supplied publication
metadata is distinct from the separately reviewed temporal-eligibility sidecar.
For recorded mode, supply `--backend recorded --recording INPUT/recording.json`;
responses must match request identities. `pipeline demo` writes a complete
synthetic recording and `replay-demo` validates exact replay.

`--aliases INPUT/aliases.json` supplies exact approved identity crosslinks;
no fuzzy merge is implied. `--settings INPUT/settings.json` supplies explicit
induction controls. See [PIPELINE.md](PIPELINE.md) for schemas and settings.

Pipeline output:

| File/directory | Contract |
|---|---|
| `observations.jsonl` | Validated source observations and evidence lineage |
| `sources.jsonl` | Exact `vendor_slug,url`, source ID, content hash, page type and supplied publication metadata |
| `raw/{nodes,edges}.jsonl` | Source-scoped atoms and typed business claims |
| `canonical/{nodes,edges}.jsonl` | Canonical concepts, hierarchy and projected claims |
| `hybrid/{nodes,edges}.jsonl` | Raw concepts plus mappings and canonical hierarchy |
| `mappings.jsonl`, `unmapped.jsonl` | Validated assignments and explicitly retained unassigned atoms |
| `induction.json`, `manifest.json` | Settings, rounds, coverage/balance/stopping and model metadata |
| `model-receipts.json`, `graph-validation.json` | Request/response identities and structural validation |

Nodes use `id,type`; labels use `text,name,canonical_name`. Edges use
`src,type,dst`, exact `source_records:[{"vendor_slug":"...","url":"..."}]`,
and per-source `claims` with `raw_src,raw_type,raw_dst,evidence`.
Canonical claims retain contributing raw-edge coordinates.

| Relation | Endpoint roles |
|---|---|
| offers | seller → product |
| provides | product → capability |
| addresses | capability → pain |
| experiences | customer → pain |
| has_case_study | seller → customer |
| in_audience | customer → audience |
| instance_of | raw atom → canonical concept |
| in_subtheme / in_domain | canonical concept → subtheme → domain |

Full all-dimension signatures and strategy relations are enforced by the core
pipeline. Product co-mention does not create capability inheritance.

## Publication snapshots and downstream tasks

The temporal adapter accepts the construction source index or a sampled source
index with `customer_ids`. It derives additional exact ownership from typed
customer-edge provenance. Eligible date records require
`temporal_use:"eligible"`, `date_semantics:"editorial_publication"` and
`reported_published_date:"YYYY-MM-DD"`; conflicting eligible dates are errors.

`graph_before` returns identity labels, eligible node evidence, eligible
per-source edges/claims and a manifest of withheld support. It does not preserve
unscoped aggregate node metadata. A source-held-out taxonomy/features protocol
is still an explicit construction input.

Local evidence corpus rows use `id,url,title,text,published_date`. IDs must be
unique. Account evidence additionally uses `independence_group` to identify the
underlying original signal. Missing/future dates are excluded for a cutoff.
No URL is fetched by this corpus adapter.

Pitch requests, account requests, judges and CRM/ratings inputs are specified in
[DOWNSTREAM.md](DOWNSTREAM.md). Model-generated identifiers must come from the
supplied finite candidate/evidence space.

## GNN inputs

The R-GCN driver requires a trusted local tensor artifact, matching SHA-256,
explicit optimization/validation pairs, seed and configuration, plus a
source-filtered construction attestation. Recovered inductive experiments
require their original cache shapes, fingerprints, feature blocks and
independent-profile contracts. [EXPERIMENTS.md](EXPERIMENTS.md) records these
interfaces; the released graph's unsplit labels are not a compatible split cache.

## Materialized historical sample

`data/sample/manifest.json` pins denominator, seed, selected-ID SHA, source
checksums, sampler implementation, closure, per-view counts and output checksums.

- `selection/customers.jsonl`: selected IDs, not the unsampled customer list.
- `sources.jsonl`: exact selected seller/URL/customer source groups.
- `{raw,raw_matched,canonical}/{nodes,edges}.jsonl`: aligned materialized views.
- `taxonomy/mappings.jsonl`: saved raw-to-canonical assignments.
- `labels.jsonl`: observed seller/customer pairs with `split:"unsplit"`.
- `temporal/evidence_dates.jsonl`: selected eligible publication metadata.

Sample provenance objects are allowlisted to
`atom_id,vendor_slug,url,evidence_type`. Native raw edges carry the normalized
original record hash, not full original aggregates. Canonical weights are
recomputed from selected claims. Some offers/provides edges are structural
context without source-specific support; source-filtered tasks withhold these
with explicit counts.

The historical sample's saved mapping order is unchanged. It is not silently
repaired or reinduced by the new pipeline, and the pinned sampler is unchanged.
