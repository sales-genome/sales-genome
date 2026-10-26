# Downstream tasks

## Pitch generation and Graph + AET

`paper_submission.tasks` implements graph path retrieval, local-model relevance
ranking, ten-point generation, bounded question/search/reflection AET, and
one-call graph/AET deduplication and selection. It retains exact product IDs,
peer IDs and citations programmatically; model-invented citation IDs are
rejected by schemas.

A pitch request contains `seller_id`, `prospect_id`, `audience_ids`,
`reference_date`, independently obtained `prospect_profile`, optional
seller/prospect names and `reference_source_urls`. Six months means six
**calendar months**, clamped at month end. Reference-story pages and undated/
post-cutoff corpus evidence are excluded.

```bash
SUBMISSION_ALLOW_LOCAL_LLM=1 SUBMISSION_LOCAL_MODEL=LOCAL_MODEL \
python -m paper_submission.tasks \
  --task pitch --mode graph_aet --request INPUT/pitch-request.json \
  --graph INPUT/graph --sources INPUT/sources.jsonl \
  --dates INPUT/evidence_dates.jsonl --corpus INPUT/authorized-corpus.jsonl \
  --out ../pitch-result.json
```

`--mode graph`, `aet` and `graph_aet` use the same contracts. The supplied corpus
is searched locally; no URL is fetched and no paid search service is enabled.
Live model calls are loopback-only and require explicit opt-in. Without it,
the package's recorded-response demonstrations exercise the same task code.

Before retrieval, the temporal/source adapter removes **every relation
supported only by target-company source records**, not just the target from
peer counts. Explicit `reference_source_urls` are excluded even when a source
has no resolved customer identity. Claims without exact per-source support are withheld with an
explicit count. The original sampled graph contains some unscoped structural
context; such edges are not silently certified as benchmark-eligible.
Source-disjoint induction/mapping eligibility must also be established for a
scientific held-out protocol.

`judge_pitch` runs a schema-bound 0–5 rating for every reference point;
`paper_submission.evaluate pitch` computes per-triple coverage as
`sum(ratings)/(5*n_reference_points)` and averages over triples:

```bash
python -m paper_submission.evaluate pitch \
  --ratings INPUT/judged-triples.jsonl --out ../pitch-coverage.json
```

Each ratings row is `{"triple_id":"...", "ratings":[0,5,3]}`. The exact benchmark
triple list, supplied reference labels and judge model/version belong in the
evaluation inputs, not synthetic graph labels.

## Account prioritization

`tasks --task account` traverses seller/product/capability/pain paths, predicts
account-specific observable problem-signal queries, searches an authorized
local evidence corpus, applies the recovered grading rubric and computes:

`raw = 10*confirmed + 5*probable + 2*suggestive`

`coverage = non_insufficient / assessed`

`score = raw * (0.5 + 0.5*coverage)`.

The account request contains `seller_id`, `account_name`, `cutoff` and optional
`product_ids`. Corpus records use `id,url,title,text,published_date` and an
`independence_group` identifying an original signal source/announcement.
Three syndicated copies of one announcement are one group, not three signals.
Confirmed/probable/suggestive grades require at least 3/2/1 cited groups.
No results yields an explicit insufficient-evidence assessment. Generic or
solution-adoption content is discounted by the recovered assessment rubric.

CRM evaluation aggregates contacts using the memo's account-name key,
gives qualifying contacts precedence over disqualification, excludes accounts
meeting neither rule and retains zero scores:

```bash
python -m paper_submission.evaluate accounts \
  --crm INPUT/contacts.csv --scores INPUT/account-scores.jsonl \
  --positive-share 0.10 --draws 1000 --seed 42 \
  --out ../qualification-evaluation.json
```

Score rows contain one `account_name,score` per account. Within each
stratified draw, selection is without replacement. Score ties are shuffled
label-blind before sorting; positives are not artificially placed first.
The report gives actual class prevalence, random precision, P@10/20 and lift.

## Publication snapshots

```bash
python -m paper_submission.temporal \
  --graph INPUT/raw --sources INPUT/sources.jsonl \
  --dates INPUT/evidence_dates.jsonl --cutoff 2020-01-01 \
  --exclude-customer customer:held-out --out ../publication-snapshot
```

Exact `(vendor_slug,url)` joins, eligible editorial-publication semantics and
conflict rejection determine inclusion. Source/customer ownership comes from
explicit `customer_ids` and source-scoped edges incident to typed customer
nodes, so the construction pipeline's source index also works directly.
Dates are not relationship/adoption onset. Retained claims preserve their
raw-atom endpoints; edge and node evidence contain only eligible support.
Unscoped node aggregates are removed rather than carrying withheld-source
metadata into the snapshot. Identity labels and a separately frozen taxonomy
are not themselves a source-held-out feature-construction certificate.

## Persuasion profiles

`python -m paper_submission.analyze --help` exposes seller/family counts,
multi-seller strategy counts and customer-industry profiles. Inputs are:

- Spans: `span_id,text,occurrences`, where each occurrence carries exact
  `vendor_slug,url,page_type` (a flat single-occurrence row is also accepted).
- Assignments: `span_id,strategy_id,subtheme_id,family_id`.
- Source/customer links: `vendor_slug,url,customer_ids`.
- Customer audiences: `customer_id,audience_id`.

Quotations deduplicate by normalized text within seller/family. Audience
profiles count distinct span IDs, join sources to one unambiguous customer and
exclude ambiguous source owners before audience lookup. No aggregate
seller/strategy/target Cartesian product substitutes for exact source joins.
These are descriptive communication profiles, not customer preferences or
measured persuasion effects.
