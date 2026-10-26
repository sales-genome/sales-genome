# Deterministic disclosure policy — provisional, local draft

The user authorized a 20% graph subset but did not select a sampling unit.
“20% of customer groups” is an engineering default, **not user confirmation**.
External release requires sampling-policy approval, source-support/rights review,
and a separately approved publication action.

## Source and denominator

The September 30 bundle's authoritative JSONL counts match all four manuscript
graph totals. Its bundle manifest SHA-256 is
`474ff3b0713be8ed00fdce6a949a394dc2302145358b7bca3555beeb909b83fd`.
Input JSONL files are checked in full against that inventory while streamed.
Neither a stale summary version string nor later recovered/repaired cohort names
are used to infer its lineage.

A “group” is a pre-existing cleaned customer ID, not a new fuzzy or semantic
company grouping. The invalid `customer:` is excluded before selection. Of
23,811 valid groups, `floor(N/5)=4,762` are selected using the seed and SHA-256
ordering specified in the manifest. Source quality/performance is not used in
selection. Even selected customers without a retained edge remain as selected
identity nodes.

## Source/claim grouping and closure

1. Keep native customer-scoped `experiences`, `in_audience` and
   `has_case_study` edges for the selected customer groups. Remove original
   source counts, vendor aggregates and customer-name collections.
2. Build exact `(vendor_slug, URL)` → customer-ID ownership from native saved
   `has_case_study.source_urls`. A source is retained only if **all its saved
   customer references are selected**. Unsafe URLs and mixed sampled/unsampled
   sources are excluded. The observed mixed-source count was zero.
   Source lists in the original graph are capped: this rule is exclusive within
   the saved graph, not proof about the unavailable complete extraction corpus.
3. Keep `addresses` claims only when their raw capability's seller and a source
   URL match those exclusive selected groups. Recompute selected customer
   memberships from source ownership. Drop unrelated/pooled original URLs and
   confidence/count attributes.
4. Add native product–capability and seller–product edges only for the selected
   capabilities, plus their endpoint nodes. This is structural product context:
   native product co-mention edges are not an exclusive source-specific
   attribution guarantee. Product-only page claims and inferred competitor
   relations are not added.
5. Keep selected raw atom labels and minimal identity/concept metadata. Never
   carry original memberships, global counts, raw variants, product customer
   lists, nested citations, full pages or private absolute paths just because a
   node participates in closure.
6. Use the same raw claims and saved last-`instance_of` mapping to create matched
   raw and canonical views. Unmapped atoms remain raw-only. Canonical business
   triples must exist in the original canonical graph; all projected triples in
   this run did. Add only required hierarchy ancestors/edges.
7. Recover source references from original canonical customer evidence **only**
   when the raw atom is selected and exact seller/URL belongs to that sampled
   customer group. Do not copy the original evidence text or pooled records.
   Native experiences without a recoverable reference remain explicitly
   customer-scoped claims with incomplete source provenance.
8. Include only eligible date records with those exact selected source keys.
   Drop full-corpus reference counts and review-tier records.

The original raw/canonical node tables are streamed, not copied wholesale.
Taxonomy overlays contribute selected mappings only. No symlinked parent,
database, source page, original report, embedding cache or serving receipt is a
runtime dependency of the finished folder.

## Bounds and table differences

The manifest records actual node/edge/byte fractions for each view and combined
native edge disclosure. Both native graph views and their combined native edge
records are below 20%. The raw matched view physically duplicates selected
records for transparency; a second conservative count includes those duplicates
and mapping edges.

The customer group fraction is 19.999160%, but **per-type fractions differ**:
shared sellers/hierarchy context can exceed 20%, and selected customers can have
above-average numbers of claims. No edge or fact was fabricated, reordered for
performance, or selectively removed to force a flattering 20% claim. In
particular the package is not described as 20% of every node/edge table.

## Splits and semantic limitations

The selected bundle has no original paper GNN split protocol. The 5,391 retained
seller–customer link labels are explicitly `unsplit`; no `train_edges.jsonl`,
test set, qualification outcome or new performance result is created.
Sample `edges.jsonl` is all-source data for reading/structural demonstrations,
not a leakage-safe training graph. Training/evaluation would need a new approved
protocol and source exclusions; it cannot inherit the October 26 exploratory
cohort's labels by matching names.

The source bundle preserves corrected-but-imperfect identities, unmapped atoms,
ambiguous assignments and known over-merging. This export does not certify
graph-wide correctness or independent source verification. Grouping by existing
IDs inherits possible identity errors. Date filtering does not establish
reference-story exclusion from the preconstructed graph. No sample measurement
is substituted for a manuscript result.
