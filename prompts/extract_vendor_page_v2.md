You are extracting observations from a **vendor page** (product page, industry page, solutions page, or other marketing content).

These pages describe what a vendor offers, who it targets, and what problems it claims to solve. Extract:

- `customer_name`: null (vendor pages rarely name a specific customer; only fill if a specific customer success story is embedded).
- `inferred_target_persona`: the buyer or user persona this page is aimed at (e.g. "Marketing Director", "IT Operations team", "Legal teams", "CFO / Finance team"). Infer from the language, job titles, or business functions addressed. Null if unclear.
- `audience_dimensions`: a structured object capturing all audience-relevant attributes stated on the page:
  - `industry`: the target industry or vertical this page addresses, as the page labels it. Null if the page is industry-agnostic.
  - `company_size`: any explicit size/scale targeting ("enterprise", "SMB", "Fortune 500"). Null if not stated.
  - `location`: geographic region or market this page targets. Null if not stated.
  - `revenue`: revenue range targeting if stated. Null if not stated.
  - `tech_stack`: technologies, platforms, or integrations mentioned on the page (e.g. "integrates with Kubernetes", "works with AWS"). Empty list if none mentioned.
  - `other`: any other audience-relevant dimension (regulatory environment, growth stage). Null if none.
- `vendor_products_used`: products described or promoted on this page, as named.
- `capabilities_cited`: specific capabilities, features, or solutions described on this page, raw phrasing.
- `pain_points_cited`: the challenges / problems / needs the page claims to address. **Each entry must be a complete, self-contained sentence that a reader can understand without seeing the rest of the page.** Bad: "compliance risk". Good: "Organizations face increasing regulatory penalties when audit trails for sensitive data access are incomplete or delayed." Include enough context for each pain to stand alone.
- `vp_pp_pairs`: for each capability that clearly addresses a specific pain point on this page, emit a pair linking them. Each entry has `capability` (which capability), `pain_point` (which pain it addresses), and `confidence` (high/med/low). Only emit pairs where the page makes the connection clear. Empty list is fine if connections are not explicit.
- `outcomes`: any claimed results, stats, or benefits. Each has `metric`, `value`, and optional `quote`. Null fields inside an outcome are fine.

## Extraction rules

1. **Refuse if uncertain.** If a field is not explicitly stated or clearly demonstrated on the page, emit `null` (for scalars) or `[]` (for lists). Do not infer from prior knowledge about the company.
2. **Do not canonicalize.** Preserve the exact phrasing the page uses. Do not rename industries, products, pains, or audiences to more standard terms.
3. **Stay on this page.** Do not incorporate information from the company's other pages or public knowledge. Only use what this page asserts.
4. **Self-rate confidence.**
   - `high` — most required fields supported by explicit statements on the page.
   - `med` — mixed; some fields inferred from context that is strong but not explicit.
   - `low` — page is sparse, mostly navigation, or the page type doesn't match what you expected. Prefer `low` over hallucination.
5. **Call `emit_extraction` exactly once.** Do not reply with any free text; emit only the tool call.

---

Page content (markdown):

{{CONTENT}}
