"""Model-driven iterative hierarchy induction and strict semantic reconciliation."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math

from paper_submission.composition import ATOM_TYPES
from paper_submission.construction import (
    ContractError, DIMENSIONS, FINGERPRINT_AXES, INDEX, ModelFailure, STRING,
    array, digest, exact_partition, obj, text,
)
from scripts.taxonomy_v4 import NOUN
from scripts.taxonomy_v4_chisel import SubTheme, schema_leaf_paths

RULES = {
    "pain": "Equivalent only for the same problem, affected actor/process, and impact. "
            "Related topics, different causes, opposite directions or unsupported specificity are NOT equivalence.",
    "capability": "Equivalent only for the same enabled product function. Branding, wording and examples "
                  "alone do not distinguish functions; a generic parent topic is not equivalence.",
    "audience": "Equivalent only for the same attribute type, scope and discriminating value. "
                "Different years, countries, sizes or units remain distinct. Do not infer customer identities.",
    "strategy": "Compare HOW claims persuade: mechanism, evidence basis, tone, emotion and warrant. "
                "Exclude business subject, product topic and company identity entirely; never group by business topic.",
}
CANONICAL_TYPES = {"pain": "pain_canonical", "capability": "cap_canonical",
                   "audience": "audience", "strategy": "strategy"}


@dataclass(frozen=True)
class InductionSettings:
    seed: str = "construction-pipeline-v1"
    sample_size: int = 200
    classification_batch: int = 100
    group_batch: int = 100
    reconciliation_batch: int = 50
    max_rounds: int = 5
    coverage_target: float = .95
    max_leaf_fraction: float = .45
    min_split_size: int = 3
    max_depth: int = 6
    explicit_singleton_fallback: bool = False

    def __post_init__(self):
        text(self.seed, maximum=300)
        for name in ["sample_size", "classification_batch", "max_rounds", "min_split_size"]:
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ContractError("Invalid positive induction setting: " + name)
        for name in ["group_batch", "reconciliation_batch"]:
            if type(getattr(self, name)) is not int or getattr(self, name) < 2:
                raise ContractError("Grouping/reconciliation batches must be at least two.")
        if type(self.max_depth) is not int or not 2 <= self.max_depth <= 8:
            raise ContractError("Hierarchy depth must be between two and eight.")
        for name in ["coverage_target", "max_leaf_fraction"]:
            value = getattr(self, name)
            if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= 1:
                raise ContractError("Coverage/balance targets must be finite in (0,1].")
        if type(self.explicit_singleton_fallback) is not bool:
            raise ContractError("Singleton fallback must be explicitly boolean.")


def hierarchy_schema(dimension):
    category = obj({"name": STRING, "description": STRING,
                    "children": array({"$ref": "#/$defs/category"})})
    root = obj({"dimension": {"const": dimension},
                "domains": array({"$ref": "#/$defs/category"}, minItems=1)})
    root["$defs"] = {"category": category}
    return root


def _hierarchy_paths(response, dimension, max_depth):
    if response["dimension"] != dimension:
        raise ContractError("Hierarchy dimension mismatch.")
    seen_objects = set()

    def validate_siblings(rows, path=()):
        names = set()
        for row in rows:
            if id(row) in seen_objects:
                raise ContractError("Cyclic/shared hierarchy object.")
            seen_objects.add(id(row))
            name = text(row["name"], model_label=True, maximum=300)
            text(row["description"], model_label=True)
            if name.lower() in names:
                raise ContractError("Duplicate sibling category.")
            names.add(name.lower())
            if len(path) + 1 > max_depth:
                raise ContractError("Hierarchy exceeds configured maximum depth.")
            if not path and not row["children"]:
                raise ContractError("Domains must contain subthemes.")
            validate_siblings(row["children"], path + (name,))

    validate_siblings(response["domains"])
    roots = [SubTheme.from_dict(row) for row in response["domains"]]
    return schema_leaf_paths(roots)


def _atoms(graph, dimension):
    atoms, ids = [], set()
    for node in graph["nodes"]:
        if node["id"] in ids:
            raise ContractError("Duplicate input graph node.")
        ids.add(node["id"])
        if node.get("dimension") == dimension and node["type"] != ATOM_TYPES[dimension]:
            raise ContractError("Raw atom dimension/type mismatch.")
        if node["type"] != ATOM_TYPES[dimension] or node.get("level", "raw") != "raw":
            continue
        if node.get("dimension", dimension) != dimension:
            raise ContractError("Wrong semantic dimension.")
        if dimension == "audience" and node.get("attribute_type") not in {
            "industry", "company_size", "location", "revenue", "tech_stack", "other",
        }:
            raise ContractError("Audience atoms require an explicit supported attribute type.")
        if dimension == "strategy":
            fingerprint = node.get("fingerprint", {})
            if set(fingerprint) != set(FINGERPRINT_AXES):
                raise ContractError("Strategy matching requires style axes only, without business subject.")
            label = " | ".join(axis + ": " + text(fingerprint[axis], model_label=True, maximum=500)
                               for axis in FINGERPRINT_AXES)
        else:
            label = text(node.get("text", node.get("name", "")), maximum=2000)
        atoms.append({"id": node["id"], "text": label,
                      "attribute_type": node.get("attribute_type") if dimension == "audience" else None})
    return sorted(atoms, key=lambda row: row["id"])


def _partition(response, count):
    members = [item for row in response["groups"] for item in row["members"]]
    if any(type(value) is not int for value in members) or sorted(members) != list(range(count)):
        raise ContractError("Canonical groups must partition all supplied items exactly once.")
    for row in response["groups"]:
        if not row["members"]:
            raise ContractError("Empty canonical group.")
        for field in ["name", "scope", "discriminator"]:
            text(row[field], model_label=True)


def group_schema(dimension):
    return obj({"dimension": {"const": dimension}, "groups": array(obj({
        "name": STRING, "scope": STRING, "discriminator": STRING,
        "members": array(INDEX, minItems=1, uniqueItems=True),
    }), minItems=1)})


def _group(items, dimension, session, *, operation, path):
    raw_ids = [member for item in items for member in item["members"]]
    if len(set(raw_ids)) != len(raw_ids):
        raise ContractError("Semantic prototypes must have disjoint raw memberships.")
    def validate(response):
        _partition(response, len(items))
        if dimension == "audience":
            for group in response["groups"]:
                kinds = {kind for index in group["members"] for kind in items[index]["attribute_types"]}
                if len(kinds) > 1:
                    raise ContractError("Audience concepts cannot merge different attribute types.")

    payload_items = []
    for index, item in enumerate(items):
        payload_items.append({
            "index": index, "text": item["name"], "scope": item["scope"],
            "discriminator": item["discriminator"],
            "representative_texts": sorted(set(item["representatives"]))[:3],
            "attribute_types": sorted(item["attribute_types"]),
        })
    result, receipt = session.ask(
        operation, {
            "dimension": dimension, "slot_path": list(path), "criteria": RULES[dimension],
            "instructions": "Group semantically EQUIVALENT items within this narrow slot. "
                            "Every supplied item index must occur exactly once. Single-member groups "
                            "are valid semantic decisions, not a missing-response fallback. Do not add "
                            "causes, mechanisms, audiences or effects absent from the item meanings. "
                            "Return labels/scopes/discriminators, never graph IDs.",
            "items": payload_items,
        }, group_schema(dimension), validate,
    )
    output = []
    for row in result["groups"]:
        selected = [items[index] for index in row["members"]]
        output.append({
            "name": row["name"], "scope": row["scope"], "discriminator": row["discriminator"],
            "members": sorted({member for item in selected for member in item["members"]}),
            "representatives": sorted({label for item in selected for label in item["representatives"]}),
            "attribute_types": sorted({kind for item in selected for kind in item["attribute_types"]}),
            "receipts": sorted({receipt} | {value for item in selected for value in item["receipts"]}),
        })
    return output


def reconcile(prototypes, dimension, session, settings, path):
    """Recursive batching plus cross-pool comparisons; no no-progress data loss."""
    if len(prototypes) <= 1:
        return prototypes
    if len(prototypes) <= settings.reconciliation_batch:
        return _group(prototypes, dimension, session, operation="reconcile_canonicals", path=path)
    midpoint = len(prototypes) // 2
    left = reconcile(prototypes[:midpoint], dimension, session, settings, path)
    right = reconcile(prototypes[midpoint:], dimension, session, settings, path)
    pool = list(left)
    for candidate in right:
        cursor = 0
        while cursor < len(pool):
            anchors = pool[cursor:cursor + settings.reconciliation_batch - 1]
            prior_members = set(candidate["members"])
            merged = _group([candidate, *anchors], dimension, session,
                            operation="reconcile_canonicals", path=path)
            active = [row for row in merged if prior_members <= set(row["members"])]
            if len(active) != 1:
                raise ContractError("Reconciliation lost or split an atomic prototype.")
            candidate = active[0]
            others = [row for row in merged if row is not candidate]
            pool[cursor:cursor + len(anchors)] = others
            # Growth removes at least one pool prototype, so restarts are finite.
            cursor = 0 if set(candidate["members"]) != prior_members else cursor + len(others)
        pool.append(candidate)
    return pool


def _taxonomy(response, dimension):
    nodes, edges, by_path = [], [], {}

    def visit(category, path=(), parent=None):
        current = path + (category["name"],)
        identifier = "taxonomy:" + dimension + ":" + digest([part.lower() for part in current])[:20]
        if identifier in by_path.values():
            raise ContractError("Hierarchy identifier collision.")
        by_path[current] = identifier
        prefix = "cap" if dimension == "capability" else "pain" if dimension == "pain" else dimension
        nodes.append({"id": identifier, "type": prefix + ("_domain" if not path else "_subtheme"),
                      "name": category["name"], "description": category["description"],
                      "dimension": dimension, "level": "category"})
        if parent:
            edges.append({"src": identifier, "type": "is_a", "dst": parent, "dimension": dimension})
        for child in category["children"]:
            visit(child, current, identifier)

    for domain in response["domains"]:
        visit(domain)
    return nodes, edges, by_path


def _fallback(atoms, dimension, reason, receipts):
    nodes, mappings = [], []
    for atom in atoms:
        identifier = "canonical:" + dimension + ":" + digest([atom["id"]])[:20]
        nodes.append({"id": identifier, "type": CANONICAL_TYPES[dimension], "name": atom["text"],
                      "text": atom["text"], "dimension": dimension, "level": "canonical",
                      "canonicalization_status": "explicit_singleton_fallback",
                      "raw_ids": [atom["id"]], "induction_receipts": receipts})
        mappings.append({"src": atom["id"], "type": "instance_of", "dst": identifier,
                         "dimension": dimension, "status": "explicit_singleton_fallback"})
    return {"dimension": dimension, "nodes": nodes, "edges": [], "mappings": mappings,
            "unmapped": [], "rounds": [],
            "status": "explicit_fallback_not_model_success", "failure_reason": reason,
            "model_attempt_receipts": receipts}


def learn_dimension(graph, dimension, session, settings=None):
    if dimension not in DIMENSIONS:
        raise ContractError("Unknown induction dimension.")
    settings = settings or InductionSettings()
    atoms = _atoms(graph, dimension)
    if not atoms:
        return {"dimension": dimension, "nodes": [], "edges": [], "mappings": [], "unmapped": [],
                "rounds": [], "status": "empty_dimension"}
    receipt_start = len(session.receipts)
    try:
        return _learn(atoms, dimension, session, settings)
    except ModelFailure:
        if not settings.explicit_singleton_fallback:
            raise
        return _fallback(atoms, dimension, "structured_model_stage_failed; inspect rejected receipts",
                         list(range(receipt_start, len(session.receipts))))


def _learn(atoms, dimension, session, settings):
    rounds, hierarchy, assignments, feedback = [], None, {}, {}
    hierarchy_receipts = []
    status = "maximum_rounds_reached"
    for round_index in range(settings.max_rounds):
        previous_overloaded = set(rounds[-1]["overloaded_leaves"]) if rounds else set()
        ordered = sorted(atoms, key=lambda row: (
            0 if assignments.get(row["id"], -1) < 0 else
            1 if assignments[row["id"]] in previous_overloaded else 2,
            digest([settings.seed, dimension, round_index, row["id"]]),
        ))
        sample = ordered[:settings.sample_size]
        response, receipt = session.ask(
            "propose_hierarchy" if hierarchy is None else "refine_hierarchy",
            {"dimension": dimension, "round": round_index + 1,
             "noun": NOUN["persuasion" if dimension == "strategy" else dimension],
             "criteria": RULES[dimension],
             "instructions": "Induce broad domains with narrow subthemes, nesting only where justified. "
                             "Use uncovered examples to add missing categories; split overloaded leaves "
                             "into more specific children. Never separate strategies by business topic. "
                             "Return the FULL revised tree, with no graph IDs.",
             "sample": [{"index": index, "text": row["text"]} for index, row in enumerate(sample)],
             "previous": hierarchy, "feedback": feedback},
            hierarchy_schema(dimension),
            lambda value: _hierarchy_paths(value, dimension, settings.max_depth),
        )
        hierarchy_receipts.append(receipt)
        unchanged = hierarchy is not None and digest(response) == digest(hierarchy)
        hierarchy = response
        leaves = _hierarchy_paths(hierarchy, dimension, settings.max_depth)
        options = [{"choice": index, "path": list(path), "description": leaf.description}
                   for index, (path, leaf) in enumerate(leaves)]
        assignments = {}
        for start in range(0, len(atoms), settings.classification_batch):
            chunk = atoms[start:start + settings.classification_batch]
            schema = obj({"dimension": {"const": dimension}, "assignments": array(obj({
                "item": INDEX, "choice": {"type": "integer", "minimum": -1, "maximum": len(leaves) - 1},
            }))})
            def validate_assignments(value):
                exact_partition(value["assignments"], len(chunk), "item")
                if any(type(row["choice"]) is not int for row in value["assignments"]):
                    raise ContractError("Hierarchy choices must be integer indices.")

            result, classification_receipt = session.ask(
                "assign_hierarchy",
                {"dimension": dimension, "criteria": RULES[dimension], "options": options,
                 "items": [{"index": index, "text": row["text"]} for index, row in enumerate(chunk)],
                 "instructions": "Assign every item once to its best supported leaf choice, or -1 when "
                                 "no leaf fits. Never force an assignment to achieve coverage."},
                schema, validate_assignments,
            )
            hierarchy_receipts.append(classification_receipt)
            for row in result["assignments"]:
                assignments[chunk[row["item"]]["id"]] = row["choice"]
        loads = Counter(choice for choice in assignments.values() if choice >= 0)
        assigned = sum(loads.values())
        coverage = assigned / len(atoms)
        overloaded = [choice for choice, count in sorted(loads.items())
                      if count >= settings.min_split_size and count / max(1, assigned) > settings.max_leaf_fraction]
        feedback = {
            "coverage": coverage, "coverage_target": settings.coverage_target,
            "max_leaf_fraction": settings.max_leaf_fraction,
            "uncovered": [row["text"] for row in atoms if assignments[row["id"]] < 0][:settings.sample_size],
            "overloaded": [{"path": list(leaves[choice][0]), "count": loads[choice],
                            "examples": [row["text"] for row in atoms if assignments[row["id"]] == choice]
                            [:settings.sample_size]} for choice in overloaded],
        }
        rounds.append({"round": round_index + 1, "coverage": coverage, "assigned": assigned,
                       "total": len(atoms), "leaf_loads": dict(loads), "overloaded_leaves": overloaded,
                       "hierarchy_sha256": digest(hierarchy), "hierarchy_receipt": receipt,
                       "schema_unchanged": unchanged})
        if coverage >= settings.coverage_target and not overloaded:
            status = "coverage_and_balance_targets_met"
            break
        if unchanged:
            status = "no_schema_progress_targets_not_met"
            break
    nodes, edges, by_path = _taxonomy(hierarchy, dimension)
    slots = defaultdict(list)
    for atom in atoms:
        choice = assignments[atom["id"]]
        if choice >= 0:
            slots[choice].append(atom)
    mappings = []
    for choice, members in sorted(slots.items()):
        path = leaves[choice][0]
        prototypes = []
        for start in range(0, len(members), settings.group_batch):
            chunk = members[start:start + settings.group_batch]
            items = [{"name": atom["text"], "scope": atom["text"], "discriminator": "Raw source description",
                      "members": [atom["id"]], "representatives": [atom["text"]], "receipts": [],
                      "attribute_types": [atom["attribute_type"]] if atom["attribute_type"] else []}
                     for atom in chunk]
            prototypes.extend(_group(items, dimension, session, operation="group_canonicals", path=path))
        prototypes = reconcile(prototypes, dimension, session, settings, path)
        partition = [member for prototype in prototypes for member in prototype["members"]]
        if sorted(partition) != sorted(atom["id"] for atom in members):
            raise ContractError("Recursive reconciliation did not preserve the complete atom partition.")
        for prototype in prototypes:
            identifier = "canonical:" + dimension + ":" + digest(prototype["members"])[:20]
            nodes.append({
                "id": identifier, "type": CANONICAL_TYPES[dimension], "name": prototype["name"],
                "text": prototype["name"], "scope_definition": prototype["scope"],
                "key_discriminator": prototype["discriminator"], "dimension": dimension,
                "level": "canonical", "canonicalization_status": "model_grouped",
                "raw_ids": prototype["members"],
                "induction_receipts": sorted(set(prototype["receipts"] + hierarchy_receipts)),
            })
            edges.append({"src": identifier, "type": "is_a", "dst": by_path[path], "dimension": dimension})
            mappings.extend({"src": member, "type": "instance_of", "dst": identifier, "dimension": dimension,
                             "status": "model_grouped"} for member in prototype["members"])
    return {
        "dimension": dimension, "nodes": sorted(nodes, key=lambda row: row["id"]),
        "edges": sorted(edges, key=lambda row: (row["src"], row["type"], row["dst"])),
        "mappings": sorted(mappings, key=lambda row: row["src"]),
        "unmapped": [{"id": atom["id"], "dimension": dimension, "reason": "no_supported_hierarchy_leaf"}
                     for atom in atoms if assignments[atom["id"]] < 0],
        "rounds": rounds, "status": status, "settings": asdict(settings),
        "hierarchy": hierarchy,
    }


def induce(graph, session, *, dimensions=DIMENSIONS, settings=None):
    if len(set(dimensions)) != len(dimensions) or any(dimension not in DIMENSIONS for dimension in dimensions):
        raise ContractError("Duplicate or unsupported induction dimensions.")
    results = [learn_dimension(graph, dimension, session, settings) for dimension in dimensions]
    return {"version": "strict-semantic-induction-1", "dimensions": results,
            "nodes": [row for result in results for row in result["nodes"]],
            "edges": [row for result in results for row in result["edges"]],
            "mappings": [row for result in results for row in result["mappings"]],
            "unmapped": [row for result in results for row in result["unmapped"]]}
