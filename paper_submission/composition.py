"""Shared-ID composition with source-scoped, non-Cartesian claim bindings."""
from __future__ import annotations

import copy
from collections import defaultdict
import hashlib
import re

from jsonschema import Draft202012Validator, ValidationError

from paper_submission.construction import (
    CASE_TYPES, ContractError, FINGERPRINT_AXES, SLUG, VERSION, audience_values, digest,
    FINGERPRINT_SCHEMA, INDEX, STRING, array, extraction_schema, norm, obj, public_url, safe_data, text,
)

ATOM_TYPES = {"pain": "pain_point", "capability": "capability", "audience": "audience", "strategy": "strategy"}


def h8(value):
    return hashlib.sha256(norm(value).encode()).hexdigest()[:8]


def customer_slug(name):
    slug = re.sub("[^a-z0-9]+", "-", name.lower()).strip("-")[:60].rstrip("-")
    return slug or "u-" + hashlib.sha256(norm(name).encode()).hexdigest()[:16]


def identity_policy(policy=None):
    policy = policy or {}
    if set(policy) - {"customer_crosslinks", "product_aliases"}:
        raise ContractError("Only explicit customer crosslinks and seller-scoped product aliases are accepted.")
    customers, products, canonical_names = {}, {}, {}
    for row in policy.get("customer_crosslinks", []):
        if set(row) != {"name", "customer_id", "canonical_name", "approved"} or row["approved"] is not True:
            raise ContractError("Customer crosslinks must be explicit approved records.")
        identifier = row["customer_id"]
        if not isinstance(identifier, str) or not identifier.startswith("customer:") or not SLUG.fullmatch(identifier[9:]):
            raise ContractError("Invalid approved customer ID.")
        alias, name = norm(text(row["name"])), text(row["canonical_name"], maximum=300)
        if alias in customers or (identifier in canonical_names and canonical_names[identifier] != name):
            raise ContractError("Duplicate/conflicting approved crosslinks.")
        customers[alias] = identifier, name
        canonical_names[identifier] = name
    for row in policy.get("product_aliases", []):
        if set(row) != {"vendor_slug", "name", "canonical_name", "approved"} or row["approved"] is not True:
            raise ContractError("Product aliases must be explicit approved records.")
        if not SLUG.fullmatch(row["vendor_slug"]):
            raise ContractError("Invalid product-alias seller scope.")
        key = row["vendor_slug"], norm(text(row["name"], maximum=300))
        if key in products:
            raise ContractError("Duplicate/conflicting product aliases.")
        products[key] = text(row["canonical_name"], maximum=300)
    for identifier, name in canonical_names.items():
        existing = customers.get(norm(name))
        if existing is not None and existing != (identifier, name):
            raise ContractError("Approved canonical name conflicts with another crosslink.")
        customers[norm(name)] = identifier, name
    resolved, visiting = {}, set()

    def resolve(key):
        if key in resolved:
            return resolved[key]
        if key in visiting:
            raise ContractError("Cyclic approved product aliases.")
        visiting.add(key)
        target = products[key]
        next_key = key[0], norm(target)
        resolved[key] = resolve(next_key) if next_key != key and next_key in products else target
        visiting.remove(key)
        return resolved[key]

    for key in products:
        resolve(key)
    products = resolved
    return customers, products


def public_observation_schema():
    base = copy.deepcopy(extraction_schema())
    evidence = obj({
        "source_id": STRING, "url": STRING, "vendor_slug": STRING,
        "content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "quote_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "start": INDEX, "end": INDEX, "evidence_type": STRING,
    })
    support = base["properties"]["support"]
    for role, schema in support["properties"].items():
        row = copy.deepcopy(schema["anyOf"][1] if role == "customer" else schema["items"])
        if role == "customer":
            schema["anyOf"][1] = row
        else:
            schema["items"] = row
        del row["properties"]["quote"]
        row["properties"]["evidence"] = evidence
        row["required"].remove("quote")
        row["required"].append("evidence")
    base["properties"]["data"]["properties"]["outcomes"]["items"]["properties"]["quote"] = {"type": "null"}
    source = obj({
        "id": STRING, "vendor_slug": STRING, "vendor_name": STRING, "url": STRING, "page_type": STRING,
        "content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "published_date": {"type": ["string", "null"]},
    })
    return obj({
        "version": {"const": VERSION}, "record_id": STRING, "source": source,
        "data": base["properties"]["data"], "support": support,
        "fingerprints": FINGERPRINT_SCHEMA["properties"]["fingerprints"],
        "quoted_outcome_indices": array(INDEX, uniqueItems=True),
        "extraction_receipt": INDEX, "fingerprint_receipt": {"type": ["integer", "null"], "minimum": 0},
        "observation_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    })


def validate_observation(observation):
    expected = {"version", "record_id", "source", "data", "support", "fingerprints", "quoted_outcome_indices",
                "extraction_receipt", "fingerprint_receipt", "observation_sha256"}
    if set(observation) != expected or observation["version"] != VERSION:
        raise ContractError("Unsupported observation contract.")
    unhashed = {key: value for key, value in observation.items() if key != "observation_sha256"}
    if digest(unhashed) != observation["observation_sha256"]:
        raise ContractError("Observation changed after source-bound extraction.")
    try:
        Draft202012Validator(public_observation_schema()).validate(observation)
    except ValidationError:
        raise ContractError("Malformed or unredacted normalized observation.") from None
    data, support = observation["data"], observation["support"]
    safe_data(data)
    from paper_submission.construction import exact_partition
    for role, data_key in [
        ("products", "vendor_products_used"), ("capabilities", "capabilities_cited"),
        ("pains", "pain_points_cited"), ("pairs", "vp_pp_pairs"), ("outcomes", "outcomes"),
    ]:
        exact_partition(support[role], len(data[data_key]))
    for key in ["vendor_products_used", "capabilities_cited", "pain_points_cited"]:
        values = [norm(value) for value in data[key]]
        if len(set(values)) != len(values):
            raise ContractError("Duplicate normalized raw atoms.")
    cap_values, pain_values = {norm(value) for value in data["capabilities_cited"]}, {
        norm(value) for value in data["pain_points_cited"]}
    for pair in data["vp_pp_pairs"]:
        if norm(pair["capability"]) not in cap_values or norm(pair["pain_point"]) not in pain_values:
            raise ContractError("Pair endpoint is not a supported raw atom.")
    if sorted((row["attribute"], row["value"]) for row in support["audiences"]) != sorted(audience_values(data)):
        raise ContractError("Incomplete or duplicate normalized audience support.")
    bindings = [(row["capability_index"], row["product_index"]) for row in support["capability_products"]]
    if len(set(bindings)) != len(bindings) or any(
        type(cap) is not int or type(product) is not int
        or cap >= len(data["capabilities_cited"]) or product >= len(data["vendor_products_used"])
        for cap, product in bindings
    ):
        raise ContractError("Unknown/duplicate normalized product binding.")
    source = observation["source"]
    if (source["id"] != observation["record_id"]
            or source["id"] != "source:" + digest({key: value for key, value in source.items() if key != "id"})[:24]
            or not SLUG.fullmatch(source["vendor_slug"]) or not public_url(source["url"])
            or source["page_type"] not in CASE_TYPES | {"product", "industry", "documentation", "press_release", "other"}):
        raise ContractError("Observation source ID differs.")
    text(source["vendor_name"], maximum=300)
    if source["published_date"] is not None:
        from datetime import date
        date.fromisoformat(source["published_date"])
    if (data["customer_name"] is None) != (support["customer"] is None):
        raise ContractError("Normalized customer support mismatch.")
    subject = support["customer"] and support["customer"]["is_case_subject"]
    if subject and source["page_type"] not in CASE_TYPES:
        raise ContractError("Non-case source cannot establish a case subject.")
    if not subject and (any(row["role"] == "customer_problem" for row in support["pains"])
                        or any(row["customer_supported"] for row in support["audiences"])):
        raise ContractError("Customer claims require a supported case subject.")
    for name, rows in observation["support"].items():
        values = [rows] if name == "customer" and rows else [] if name == "customer" else rows
        for row in values:
            evidence = row.get("evidence", {})
            if evidence.get("source_id") != source["id"] or evidence.get("content_sha256") != source["content_sha256"]:
                raise ContractError("Claim support references another source/version.")
            if not (type(evidence.get("start")) is int and type(evidence.get("end")) is int
                    and 0 <= evidence["start"] < evidence["end"]
                    and re.fullmatch("[0-9a-f]{64}", evidence.get("quote_sha256", ""))):
                raise ContractError("Invalid hashed evidence offsets.")
            if ("quote" in row or evidence.get("url") != source["url"]
                    or evidence.get("vendor_slug") != source["vendor_slug"]
                    or evidence.get("evidence_type") != name):
                raise ContractError("Unredacted or mismatched citation support.")
    fingerprints = observation["fingerprints"]
    indices = [row["outcome_index"] for row in fingerprints]
    if (sorted(indices) != sorted(observation["quoted_outcome_indices"])
            or len(set(indices)) != len(indices)
            or any(type(index) is not int or index >= len(data["outcomes"]) for index in indices)):
        raise ContractError("Unknown/duplicated normalized fingerprint outcome.")
    if bool(fingerprints) != (observation["fingerprint_receipt"] is not None):
        raise ContractError("Normalized fingerprint receipt mismatch.")
    for row in fingerprints:
        if (any(type(index) is not int or index >= len(support["audiences"]) for index in row["target_audience_indices"])
                or row["cites_case_customer"] and not subject
                or not row["persuasive"] and (row["cites_case_customer"] or row["target_audience_indices"])):
            raise ContractError("Unsupported normalized strategy attribution.")
        if row["persuasive"] and any(not row["fingerprint"][axis].strip() for axis in FINGERPRINT_AXES):
            raise ContractError("Persuasive strategy requires every style axis.")
        for value in [row["claim"], *row["fingerprint"].values()]:
            if value:
                text(value, model_label=True, maximum=500)


def compose(observations, *, aliases=None):
    customer_aliases, product_aliases = identity_policy(aliases)
    nodes, identities, edges = {}, {}, {}
    sources, observation_ids = {}, set()

    def node(identifier, kind, name, *, identity_key=None, evidence=None, **fields):
        if identifier in nodes:
            if nodes[identifier]["type"] != kind or identities[identifier] != identity_key:
                raise ContractError("Identifier collision: supply an approved exact crosslink instead.")
        else:
            nodes[identifier] = {"id": identifier, "type": kind, "name": name, **fields, "evidence": []}
            identities[identifier] = identity_key
        if evidence and evidence not in nodes[identifier]["evidence"]:
            nodes[identifier]["evidence"].append(copy.deepcopy(evidence))
        return identifier

    def edge(src, relation, dst, evidence, *, role):
        if src not in nodes or dst not in nodes:
            raise ContractError("Composition created an unknown edge endpoint.")
        triple = src, relation, dst
        row = edges.setdefault(triple, {"src": src, "type": relation, "dst": dst, "claims": [],
                                       "source_records": []})
        claim = {
            "id": "claim:" + digest({"edge": triple, "evidence": evidence, "role": role})[:24],
            "role": role, "evidence": copy.deepcopy(evidence),
            "raw_src": src, "raw_type": relation, "raw_dst": dst,
        }
        if claim not in row["claims"]:
            row["claims"].append(claim)

    for observation in observations:
        validate_observation(observation)
        if observation["record_id"] in observation_ids:
            raise ContractError("Duplicate source observation.")
        observation_ids.add(observation["record_id"])
        source, data, support = observation["source"], observation["data"], observation["support"]
        sources[source["id"]] = copy.deepcopy(source)
        vendor = source["vendor_slug"]
        vendor_id = node("vendor:" + vendor, "vendor", source["vendor_name"], identity_key=vendor,
                         vendor_slug=vendor)
        customer = None
        if support["customer"] and support["customer"]["is_case_subject"] and source["page_type"] in CASE_TYPES:
            raw_name = data["customer_name"]
            identifier, display_name = customer_aliases.get(norm(raw_name), (
                "customer:" + customer_slug(raw_name), raw_name,
            ))
            identity = ("approved", identifier) if norm(raw_name) in customer_aliases else ("exact", norm(raw_name))
            customer = node(identifier, "customer", display_name, identity_key=identity,
                            evidence=support["customer"]["evidence"])
            edge(vendor_id, "has_case_study", customer, support["customer"]["evidence"], role="named_case_subject")
        products = []
        product_support = {row["index"]: row for row in support["products"]}
        for index, raw_name in enumerate(data["vendor_products_used"]):
            canonical_name = product_aliases.get((vendor, norm(raw_name)), raw_name)
            identifier = node(
                f"product:{vendor}:{h8(canonical_name)}", "product", canonical_name,
                identity_key=(vendor, norm(canonical_name)), vendor_slug=vendor,
                evidence=product_support[index]["evidence"],
            )
            products.append(identifier)
            edge(vendor_id, "offers", identifier, product_support[index]["evidence"], role="source_named_offering")
        capabilities, pains, audiences = [], [], []
        cap_support = {row["index"]: row for row in support["capabilities"]}
        for index, label in enumerate(data["capabilities_cited"]):
            identifier = node(
                f"cap:{vendor}:{h8(label)}", "capability", label, text=label, dimension="capability",
                level="raw", identity_key=(vendor, norm(label)), vendor_slug=vendor,
                evidence=cap_support[index]["evidence"],
            )
            capabilities.append(identifier)
            bindings = [row for row in support["capability_products"] if row["capability_index"] == index]
            if bindings:
                for binding in bindings:
                    edge(products[binding["product_index"]], "provides", identifier,
                         binding["evidence"], role="explicit_individual_product_capability")
            else:
                unspecified = node(
                    f"product:{vendor}:unspecified", "product", "Unspecified offering",
                    identity_key=(vendor, "unspecified"), vendor_slug=vendor,
                    attribution="seller_scoped_unspecified_not_a_named_product",
                )
                edge(vendor_id, "offers", unspecified, cap_support[index]["evidence"], role="seller_scoped_unspecified")
                edge(unspecified, "provides", identifier, cap_support[index]["evidence"], role="unattributed_capability")
        pain_support = {row["index"]: row for row in support["pains"]}
        for index, label in enumerate(data["pain_points_cited"]):
            identifier = node(
                "pain:" + h8(label), "pain_point", label, text=label, dimension="pain",
                level="raw", identity_key=norm(label), evidence=pain_support[index]["evidence"],
            )
            pains.append(identifier)
            if pain_support[index]["role"] == "customer_problem":
                if not customer:
                    raise ContractError("Unsupported customer-problem attribution.")
                edge(customer, "experiences", identifier, pain_support[index]["evidence"], role="case_customer_problem")
        cap_by_text = {norm(label): identifier for label, identifier in zip(data["capabilities_cited"], capabilities)}
        pain_by_text = {norm(label): identifier for label, identifier in zip(data["pain_points_cited"], pains)}
        pair_support = {row["index"]: row for row in support["pairs"]}
        for index, pair in enumerate(data["vp_pp_pairs"]):
            edge(cap_by_text[norm(pair["capability"])], "addresses", pain_by_text[norm(pair["pain_point"])],
                 pair_support[index]["evidence"], role="explicit_capability_problem_pair")
        for row in support["audiences"]:
            label = row["attribute"] + ": " + row["value"]
            identifier = node(
                "audience:" + h8(label), "audience", row["value"], text=label, dimension="audience",
                attribute_type=row["attribute"], level="raw", identity_key=norm(label), evidence=row["evidence"],
            )
            audiences.append(identifier)
            if row["customer_supported"]:
                if not customer:
                    raise ContractError("Unsupported customer attribute.")
                edge(customer, "in_audience", identifier, row["evidence"], role="source_customer_attribute")
        outcome_support = {row["index"]: row for row in support["outcomes"]}
        for fingerprint in observation["fingerprints"]:
            if not fingerprint["persuasive"]:
                continue
            style = {axis: fingerprint["fingerprint"][axis] for axis in FINGERPRINT_AXES}
            matching = " | ".join(axis + ": " + style[axis] for axis in FINGERPRINT_AXES)
            identifier = node(
                "strategy:" + h8(matching), "strategy", matching, text=matching,
                dimension="strategy", level="raw", fingerprint=style, identity_key=norm(matching),
                evidence=outcome_support[fingerprint["outcome_index"]]["evidence"],
            )
            evidence = outcome_support[fingerprint["outcome_index"]]["evidence"]
            edge(vendor_id, "asserts", identifier, evidence, role="quoted_outcome_persuasion")
            if fingerprint["cites_case_customer"]:
                if not customer:
                    raise ContractError("Unsupported strategy customer citation.")
                edge(identifier, "cites", customer, evidence, role="quote_cites_case_customer")
            for index in fingerprint["target_audience_indices"]:
                edge(identifier, "targets", audiences[index], evidence, role="explicit_strategy_audience")
    for row in nodes.values():
        row["evidence"].sort(key=digest)
    for row in edges.values():
        row["claims"].sort(key=lambda claim: claim["id"])
        row["source_records"] = [
            {"vendor_slug": vendor, "url": url}
            for vendor, url in sorted({(claim["evidence"]["vendor_slug"], claim["evidence"]["url"])
                                      for claim in row["claims"]})
        ]
    return {
        "version": VERSION, "nodes": [nodes[key] for key in sorted(nodes)],
        "edges": [edges[key] for key in sorted(edges)],
        "sources": [sources[key] for key in sorted(sources)],
        "identity_policy_sha256": digest(aliases or {}),
        "claim_status": "source_bound_extraction_claims_not_independent_factual_verification",
    }
