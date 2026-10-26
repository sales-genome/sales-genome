You are extracting observations from a single **vendor customer success story / case study**.

A case study typically has: customer identification, a "challenge" or "before" section, the products used, and outcomes / results. Extract:

- `customer_name`: the customer company named in the study.
- `inferred_target_persona`: the buyer or primary user persona at the customer (e.g. "Marketing Director", "VP Engineering", "IT Operations team", "CFO / Finance team"). Infer from job titles mentioned, the team described, or the business function discussed. Null if truly unclear.
- `audience_dimensions`: a structured object capturing all audience-relevant attributes stated on the page:
  - `industry`: industry as the page labels it (do not normalize).
  - `company_size`: any explicit size/scale indicator ("Fortune 500", "10,000+ employees", "mid-market", "enterprise"). Null if not stated.
  - `location`: geographic location, headquarters, or primary region. Null if not stated.
  - `revenue`: revenue figure or range if explicitly stated. Null if not stated.
  - `tech_stack`: technologies, platforms, or tools the customer uses *besides* the vendor's own products (e.g. "AWS", "Kubernetes", "Salesforce", "Python"). Empty list if none mentioned.
  - `other`: any other audience-relevant dimension (regulatory environment, growth stage, business model). Null if none.
- `vendor_products_used`: vendor products the customer deployed, as named on the page.
- `capabilities_cited`: specific capabilities the customer used or cited, raw phrasing.
- `pain_points_cited`: the challenges / problems the customer faced before the deployment. **Each entry must be a complete, self-contained sentence that a reader can understand without seeing the rest of the page.** Bad: "scale issues". Good: "The company struggled to scale its data pipeline from 10M to 500M daily events without increasing infrastructure costs." Include enough context (who, what, why) for each pain point to stand alone.
- `vp_pp_pairs`: for each capability that clearly addressed a specific pain point on this page, emit a pair linking them. Each entry has `capability` (which capability), `pain_point` (which pain it addressed), and `confidence` (high/med/low). Only emit pairs where the page makes the connection clear. Empty list is fine if connections are not explicit.
- `outcomes`: list of result statements. Each has `metric` (e.g. "time-to-publish"), `value` (raw, e.g. "+42%", "from 3 weeks to 2 days"), and optional `quote` (supporting quote from the customer). Null fields inside an outcome are fine if only some are stated.

## Extraction rules

1. **Refuse if uncertain.** If a field is not explicitly stated or clearly demonstrated on the page, emit `null` (for scalars) or `[]` (for lists). Do not infer from prior knowledge about the company.
2. **Do not canonicalize.** Preserve the exact phrasing the page uses. Do not rename industries, products, pains, or audiences to more standard terms. "law firms", "legal teams", and "in-house counsel" stay distinct if the page uses each distinctly.
3. **Stay on this page.** Do not incorporate information from the company's other pages or public knowledge. Only use what this page asserts.
4. **Self-rate confidence.**
   - `high` — most required fields supported by explicit statements on the page.
   - `med` — mixed; some fields inferred from context that is strong but not explicit.
   - `low` — page is sparse, mostly navigation, or the page type doesn't match what you expected. Prefer `low` over hallucination.
5. **Call `emit_extraction` exactly once.** Do not reply with any free text; emit only the tool call.

---

Page content (markdown):

{{CONTENT}}
