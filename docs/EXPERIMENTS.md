# Relationship-prediction experiments

The historical authority is the recovered **Inductive Lead GNN Lab Notebook**
(final SHA-256
`0a7ffe44860d1758ab5bd77e3583ba4abe279df545787a547248cf7c83234279`)
and **Lead Signals + Inductive GNN Architecture** memo (final SHA-256
`67b82a96aa6617c1ef0e5218206a05206b894e9c8959112b419d3345a3958403`).
The code includes 18 files recovered exactly against their recorded final
historical hashes, with explicit portability/provider adaptations where needed.
`provenance/experiment-source-manifest.json` records each source hash and change.
The notebook, chats and private experiment datasets are not release data.

## Model families and entry points

| Experiment | Implementation | Input contract |
|---|---|---|
| Transductive R-GCN | Captured complete `FeatureProjector`, `RGCNEncoder`, `LinkPredModel` classes in `experiments/graph_embeddings/train_v2.py`; `paper_submission/rgcn.py` supplies training and resampling | Frozen source-filtered PyG base graph, explicit optimization/validation pairs and model configuration |
| Directional two-tower | Recovered `inductive_baseline.py` | Original feature-cache contract, explicit edge-transfer versus oracle-node protocol |
| Linear / MLP / relation SAGE / relation attention | Recovered `inductive_model_family.py` and matrix runner | Named feature blocks plus source/evaluation cache fingerprints and independent-profile provenance |
| Exact-degree controls | Recovered `eval_inductive_degree_matched.py` | Training degree, positive/control candidates and compatible model/cache |
| Complete-customer / double-cold evaluation | Recovered `eval_double_coldstart.py`, `eval_independent_centroids.py` | Original node split and independent held-out customer population |
| Product-page vendor features | Recovered `product_page_vendor_features.py` | Product-only extractions, source cache, local BGE model |
| Canonical-pain alignment | Recovered `canonical_vendor_pain_features.py` | Product-page pain text, canonical definitions, BGE vectors and original maps |
| Independent customer profiles | Recovered `independent_profile_experiment.py` | Identity-screened customers, independent evidence, audience catalog and compatible frozen cache |
| Frozen CRM transfer | Recovered `lead_prior.py`, `score_lead_cohort.py`, `eval_lead_ensemble.py`, `crm_independent_profile_experiment.py` | Authorized external checkpoints, company profiles, account labels and score sidecars |

The relation-aware inductive towers operate on **pre-aggregated relation
blocks**, not full multi-layer graph propagation. The recent GraphSAGE smoke
is separate; it is not substituted for the paper R-GCN or the notebook models.

## R-GCN configuration

The memo specifies 3 residual R-GCN layers, 256 hidden dimensions, at most
8 relation bases, dropout 0.2, cosine scoring, temperature 0.1, five random
customer controls per positive, Adam at `1e-3` with `1e-5` weight decay,
200 epochs and patience 30. Seller embeddings mean-pool their product
embeddings.

`collapsed_graph` constructs product/capability/pain/customer forward and
reverse relations and product/customer residual edges from **training pairs
only**. `d100` removes capability nodes with **more than 100 unique product
neighbors**. It is not a top-100-per-seller operation.

The reconstructed training driver makes validation selection explicit and
never selects a checkpoint on the final test. Its source is distinguished
from the captured encoder classes; it does not assert byte-for-byte identity
with the historical training driver.

```bash
python -m paper_submission.rgcn \
  --data INPUT/frozen-base.pt --protocol INPUT/rgcn-protocol.json \
  --trusted-local-input --seed 42 --device cpu --out ../rgcn-run
```

The protocol JSON contains `data_sha256`, an explicit
`heldout_source_evidence_removed: true` construction contract,
`optimization_pairs`, `validation_pairs`, and a `configuration` object matching
`Configuration`. The attestation is not an independent source-support audit:
construct the frozen graph by withholding evaluation source claims first.
PyG/feature caches use pickle-backed Torch formats; load **trusted local
artifacts only**.

`sampled_recovery` implements the manuscript's fixed-seller experiment:
500 resamples, 50 sellers per resample, 10 held-out positives and 90 unlabeled
controls per seller, with P@10/20/50. Its API requires an explicit seed.
It rejects insufficient populations instead of silently changing these counts.
Unobserved pairs are controls, not verified commercial negatives.

## Recovered inductive training

The original family trainer exposes its actual historical defaults:
hidden 256, output 128, GELU/dropout 0.2, 150 epochs, ten controls,
AdamW `1e-3`, weight decay `1e-5`, patience 20, validation fraction 0.1 and
temperature 0.1. All are explicit CLI parameters.

Example for the notebook's independent-customer feature contract:

```bash
python -m experiments.graph_embeddings.inductive_model_family train \
  --source-cache INPUT/source.pt --evaluation-cache INPUT/independent.pt \
  --vendor-feature unpruned --customer-feature semantic \
  --vendor-block-dims product:1024 capability:1024 \
  --customer-block-dims pain_semantic:1024 audience:50 \
  --evaluation-slice customer_holdout --family mlp \
  --device cpu --seed 42 --output-dir ../inductive-run
```

Feature names and block sizes must match the supplied cache, not be inferred
from the sample. The recovered source/evaluation validation checks exact
training rows, source fingerprints, profile construction contracts and
held-out candidates. Training rejection masks consult training labels only;
validation controls are fixed. Customer train-degree popularity and
degree-matched controls remain required comparisons.

The recorded ten seeds were `42 7 21 84 101 202 303 404 505 606`. The matrix
runner accepts `--seeds` and explicit `--gpus`; importing it launches nothing.
The notebook invalidated early oracle-node-holdout results because case-study
features retained scored-case evidence. Those runs are not model-selection
evidence. The final notebook decision favors MLP for the small true-double-cold
cohort; relation attention remains a useful customer-holdout comparison.

## Input preparation and local execution

The original preparation recipes accept the original graph/node-map/feature
contracts; the released 20%-customer graph sample has **unsplit** labels.
Do not rename its native edges to training edges or claim the historical
split by shuffling label rows while retaining their source claims.

Local LLM calls use the explicit opt-in package provider. The historical
BrightData search interface is adapted to `SUBMISSION_SEARCH_CORPUS`: an
authorized local JSONL corpus (`id,url,title,text,published_date`), never an
external paid search service. This transport change is recorded and is not
presented as a rerun of the historical web search. A model-cache path is
portable; no private machine cache, credential or model weights are packaged.
Historical contextual-pain profile features remain hypotheses, not factual
`experiences` edges in the construction pipeline.

The included checks exercise real CPU optimizer/backpropagation on explicit
synthetic fixtures. They do not regenerate the manuscript's reported metrics.
