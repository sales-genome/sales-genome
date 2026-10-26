# Sales Genome: construction and downstream code

This package accompanies *Sales Genome: Modeling Persuasion in Sales
Communication as a Knowledge Graph*. It contains executable **construction,
composition, induction and canonicalization**, raw/canonical/hybrid graph
projections, and downstream prediction, pitch-generation, account-prioritization
and persuasion-analysis code.

This is an **anonymous research code submission** for
[`sales-genome/sales-genome`](https://github.com/sales-genome/sales-genome).
Publication authorization and the exact shared-data scope are recorded in
[`provenance/publication-v1.json`](provenance/publication-v1.json).
This package does not modify the source manuscript or original repository.
The method snapshot used at initial staging has SHA-256
`541960190dc838614b403296a1c6225503d6ae422a799e9d9e80530bbf56f9e4`;
the current manuscript observation is recorded separately in the completion
provenance.

## Start here

Use Python 3.11+ in a separately managed environment. The full checks include
Torch/PyG; do not bundle the environment or model caches.

```bash
python -m pip install -e '.[gnn,test]'

PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests
PYTHONDONTWRITEBYTECODE=1 python -m paper_submission.validate --sample data/sample
PYTHONDONTWRITEBYTECODE=1 python -m paper_submission.audit_package --verify-checksums

# Choose new output directories when repeating.
PYTHONDONTWRITEBYTECODE=1 python -m paper_submission.pipeline demo \
  --out ../sales-genome-demo
PYTHONDONTWRITEBYTECODE=1 python -m paper_submission.pipeline replay-demo \
  --demo ../sales-genome-demo --out ../sales-genome-replay
```

The demo executes the **same extraction, shared-identity composition, iterative
hierarchy refinement, semantic grouping, reconciliation and projection
algorithms** through explicit synthetic recorded model responses. It is not
just schema validation or a hand-authored canonical overlay. The fixture
deliberately triggers refinement and overloaded-category splitting in all four
dimensions: pain, capability, audience and persuasion strategy.

Outputs include raw, canonical and hybrid tables; mappings and unmapped atoms;
source hashes; per-source claim lineage; model receipts; induction stopping/
coverage/balance reports; and graph-validation reports. Replaying checks exact
recorded request identities. Synthetic outputs are not research measurements.

For supplied authorized pages and an explicitly configured local model:

```bash
SUBMISSION_ALLOW_LOCAL_LLM=1 SUBMISSION_LOCAL_MODEL=LOCAL_MODEL \
SUBMISSION_LOCAL_BASE_URL=http://localhost:8000/v1 \
python -m paper_submission.pipeline run \
  --pages INPUT/pages.jsonl --backend local --out ../sales-genome-run
```

There is no default paid search, model call or page fetch. The local provider
requires opt-in, rejects non-loopback endpoints and disables HTTP proxies and
redirects. Recorded mode needs no server. Input/model contracts and induction
controls are in [the pipeline guide](docs/PIPELINE.md).

## Implementation map

| Stage | Primary code | Behavior |
|---|---|---|
| Construction / extraction | `paper_submission/construction.py` | Page-type-aware, schema-bound extraction with source hash and exact evidence locations |
| Composition | `paper_submission/composition.py` | Shared exact IDs, approved aliases, seller-scoped products/capabilities, source-scoped claims; no all-product capability inheritance |
| Induction | `paper_submission/induction.py` | Sample/refine/classify, coverage and balance checks, overloaded-category splitting, per-slot semantic grouping and recursive reconciliation |
| Canonicalization | `paper_submission/canonicalization.py` | Validated mappings, preserved seller/product/customer identities, unmapped raw atoms, hierarchy closure and raw-edge lineage |
| Orchestration | `paper_submission/pipeline.py` | Local-model/recorded execution, reproducible fixture and exact replay |
| Publication snapshots | `paper_submission/temporal.py` | Exact source/date joins and target-source removal across all relation types |
| Pitch generation / Graph+AET | `paper_submission/tasks.py` | Product-aware graph retrieval/ranking, bounded adaptive evidence search, ten-point generation and merge, evidence-bound judging |
| Account prioritization | `tasks.py`, `evaluate.py` | Signal prediction, evidence grading, weighted scores, CRM label aggregation and stratified ranking evaluation |
| Relationship prediction | `paper_submission/rgcn.py`, `experiments/graph_embeddings/` | R-GCN training/resampling and recovered linear/MLP/relation-SAGE/relation-attention experiments |
| Persuasion analysis | `paper_submission/analyze.py` | Quote-deduplicated seller profiles and exact source/customer/audience joins |

The recovered original builders, taxonomy implementations, extraction prompts
and historical function references are also retained under `scripts/`,
`schemas/`, `lib/` and `reference/`. New generalized reproduction paths are
distinguished from recovered original source, rather than presented as the
historical execution log.

Guides: [methods map](docs/STAGES.md), [input/output contracts](docs/CONTRACTS.md),
[downstream commands](docs/DOWNSTREAM.md), [GNN experiments](docs/EXPERIMENTS.md)
and [sampling](docs/SAMPLING.md).

## Recovered GNN experiments

Old experiments follow the recovered **GNN lab notebook and architecture memo**,
not newer experiments substituted under old names. Eighteen final-source Python
files were recovered against recorded historical hashes, including corrected
inductive baselines, the four final model families, independent customer profiles,
product-page and canonical-pain features, degree controls and double-cold/CRM
evaluation. Complete historical R-GCN encoder and graph-builder symbols were
also recovered; a separately identified driver supplies explicit training and
validation protocols.

`provenance/experiment-source-manifest.json` records authority hashes, original
source hashes, symbol hashes and portability/local-provider adaptations.
The memo-configured R-GCN uses three 256-dimensional residual layers, InfoNCE
and capability pruning by **unique product degree >100**, not top-100 truncation.
The inductive relation models are relation-block towers, not multi-hop R-GCNs.

Real evaluations consume caller-supplied frozen splits, feature caches, evidence
and authorized labels. Torch caches/checkpoints must be trusted local inputs.
Feature-generation commands additionally use `.[embeddings]` and an explicitly
available compatible model. The demonstrations do not regenerate paper metrics.

## Shared graph data: 20% of customer groups

The materialized sample is from the manuscript-matched September 30
`raw_v7` / `canonical_v8` graph with saved v6 overlays, not the newer reviewed
research cohort. One deterministic selection controls all views:
**4,762 of 23,811 valid customer groups (19.999160%)**.

Seed: `audience-genome-paper-submission-20261026-v1`.
Selection sorts `(SHA256(seed + NUL + customer_id), customer_id)` and takes
`floor(N/5)`. Sampling does not use model scores or favorable examples.

| View | Nodes | Edges |
|---|---:|---:|
| Raw | 52,940 | 96,538 |
| Matched raw | 45,923 | 83,514 |
| Canonical + hierarchy | 22,021 | 87,392 |

The sample retains necessary shared context, 7,004 exact seller/URL source
groups, 5,391 observed seller/customer pairs, 35,364 mappings and 1,778 eligible
date records. Labels are explicitly **unsplit**. Sampled claims remain native
extracted claims, not independently verified facts.

This is **20% of customer groups, not 20% of every table**. Native raw+canonical
edges disclose 183,930/11,553,078 = 1.592043%; counting matched duplicates and
mappings gives 2.621016%. The JSONL payload is 127,929,205 bytes. No full
databases, source-page caches, private CRM records or model weights are included.
The source manifest retains its original provisional sampling-policy fields;
publication uses this already-materialized sample without changing its
selection. Publication does not assign a code/data license; see
[NOTICE.txt](NOTICE.txt).

## Provenance and package integrity

`provenance/source-manifest.json` records copied-source provenance;
`provenance/experiment-source-manifest.json` records historical experiment
recovery. Completion records are under `validation/`. The checksum registry
covers the **entire release file set**, excluding only the registry itself and
root Git control metadata. Changed, removed or unexpectedly added release files
fail verification.

After an intentional package revision, audit and explicitly reseal:

```bash
python -m paper_submission.audit_package --seal
python -m paper_submission.audit_package --verify-checksums
```

Keep generated runs outside this release directory. Resealing records local
file integrity; it does not grant redistribution permission or certify semantic
correctness. No commit, push or upload is performed by these commands.
