# Manuscript-to-implementation map

The primary path is `extract_pages → compose → induce → project`, orchestrated by
`paper_submission.pipeline.run_pipeline`. The executable construction path is a
generalized reproduction implementation. Recovered source files and historical
symbols keep their separate provenance records.

| Method | Implementation | Inputs and outputs |
|---|---|---|
| Page extraction | `construction.py`; original prompts/schema in `lib/ingest/`, `schemas/` | Supplied page records → schema-bound observations, exact evidence offsets and source hashes |
| Shared-ID composition | `composition.py` | Observations plus explicit alias policy → source-scoped raw graph; products/capabilities remain seller-scoped |
| Hierarchy induction | `induction.py` | Pain, capability, audience and strategy atoms → sampled/refined hierarchy and full assignment reports |
| Canonical grouping | `induction.py` | Within-slot semantic grouping, recursive reconciliation and validated raw-to-canonical mappings |
| Canonicalization | `canonicalization.py` | Raw graph + induction → canonical and hybrid views, hierarchy, unmapped atoms, raw-edge lineage |
| Identity/claim review | `repair.py`, `reference/repair-*` | Accepted exact-ID decisions and source/hash-bound reviews → validated merge/retain decisions; uncertainty is withheld |
| Publication filtering | `temporal.py` | Exact source/date records + graph → eligible publication snapshot, target-source mask and exclusion counts |
| Cold-start pitch | `tasks.py`, `downstream.py` | Offers→provides→addresses paths + audience peers + independent prospect context → ranked/generated ten-point pitch |
| Adaptive evidence traversal | `tasks.py`, `experiment_io.py` | Bounded question/reflection loop over supplied local documents → cited AET points, Graph+AET selection and judge ratings |
| Account prioritization | `tasks.py`, `evaluate.py` | Product/problem paths + dated independent signals → evidence grades, composite score and CRM ranking evaluation |
| Transductive link recovery | `rgcn.py`, captured `train_v2.py` symbols | Frozen PyG graph, optimization/validation labels and explicit configuration → selected checkpoint, embeddings and resampled ranking |
| Inductive link recovery | Recovered `experiments/graph_embeddings/` | Original-compatible, fingerprinted feature/split contracts → linear, MLP, relation-SAGE and relation-attention runs |
| Persuasion profiles | `analyze.py` | Source-linked spans, strategy hierarchy and customer/audience links → deduplicated seller and audience communication profiles |

## Induction and projection

The offline demo exercises extraction and actual hierarchy refinement/grouping
through recorded responses; it does not start from fabricated mapping overlays.
Coverage, category load, refinement rounds and stopping reasons are recorded
for every dimension. Mapping validation rejects conflicting, duplicate, partial,
cross-dimension or cyclic structures. Explicit singleton fallback is a named
setting, never an invisible model-error recovery.

Products, customers and sellers preserve their identities through projection.
Business edges retain exact source references and per-source raw claim
coordinates. Strategy abstraction removes business subject rather than treating
different business topics as different persuasion mechanisms.

The older `build_v4_graph.py`, `build_v5_graph.py`, `taxonomy_v4.py` and
`taxonomy_v4_chisel.py` remain available as recovered recipes. The new path
adds all-dimension contracts and source lineage without relabeling those scripts
as an exact recovered September invocation.

## Temporal and evaluation protocols

Filtering source claims must precede feature construction and graph propagation.
The temporal adapter removes target-supported edges across all relation types,
filters node evidence and retains only eligible raw-claim lineage. It also
derives source/customer ownership from typed customer edges when construction
source records do not carry a `customer_ids` list.

Publication dates are not adoption dates. A graph containing unscoped structural
context or a taxonomy induced from evaluation sources is not automatically a
valid temporal or source-held-out benchmark. Supplied protocols must establish
that taxonomy and feature inputs satisfy their split.

The GNN notebook invalidates early complete-node oracle runs and retains
training-degree popularity, degree-matched and independent-profile controls.
The final recovered inductive relation models are pre-aggregated block towers,
not full symmetric graph convolutions. Historical benchmark values remain
historical values, not results from the synthetic demonstrations.

## Graph data

`data/sample` is the manuscript-matched September raw-v7/canonical-v8 selection.
Its single customer sample is shared across raw, matched-raw and canonical views.
Saved mapping order is preserved in that **historical sample**, including its
last-target convention; the new pipeline instead validates unambiguous mappings.
The sample's labels are unsplit and its native claims unreviewed.

Exact sample policy and counts: [SAMPLING.md](SAMPLING.md).
Operational guides: [PIPELINE.md](PIPELINE.md), [DOWNSTREAM.md](DOWNSTREAM.md),
[EXPERIMENTS.md](EXPERIMENTS.md).
