"""Pitch coverage, fixed-seller ranking and account-level stratified evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import random
import statistics

from paper_submission.artifacts import lines, save


def pitch_coverage(ratings):
    ratings = list(ratings)
    if not ratings or any(type(rating) is not int or not 0 <= rating <= 5 for rating in ratings):
        raise ValueError("Coverage requires at least one integer rubric rating in [0, 5].")
    return sum(ratings) / (5 * len(ratings))


def ranking_metrics(scores, positives, *, ks=(10, 20, 50), seed=42):
    positives = set(positives)
    if not positives or not positives <= scores.keys():
        raise ValueError("Every query needs positives present in its candidate pool.")
    if any(not math.isfinite(value) for value in scores.values()):
        raise ValueError("Non-finite candidate score.")
    if not ks or any(type(k) is not int or k <= 0 for k in ks):
        raise ValueError("Ranking cutoffs must be positive integers.")
    order = sorted(scores)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda item: -scores[item])
    labels = [int(item in positives) for item in order]
    found, ap = 0, 0.0
    for rank, label in enumerate(labels, 1):
        found += label
        if label:
            ap += found / rank
    result = {
        "average_precision": ap / len(positives),
        "n_candidates": len(order), "n_positives": len(positives),
        "chance_precision": len(positives) / len(order),
    }
    for k in ks:
        width = min(k, len(labels))
        hits = sum(labels[:width])
        dcg = sum(label / math.log2(rank + 2) for rank, label in enumerate(labels[:width]))
        ideal = sum(1 / math.log2(rank + 2) for rank in range(min(width, len(positives))))
        result[f"precision_at_{k}"] = hits / width
        result[f"recall_at_{k}"] = hits / len(positives)
        result[f"ndcg_at_{k}"] = dcg / ideal
    return result


def account_key(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Account name is required.")
    return (name.lower().replace(" ", "_").replace(".", "").replace(",", "")
            .replace("(", "").replace(")", ""))


def qualification_labels(contacts):
    """Apply the memo's cumulative account-level qualification/NQ rules."""
    grouped = defaultdict(lambda: {"statuses": set(), "reasons": set(), "names": set()})
    for row in contacts:
        name = row.get("Account Name") or row.get("Company Name")
        if not name:
            raise ValueError("CRM row has neither Account Name nor Company Name.")
        account = grouped[account_key(name)]
        account["names"].add(name)
        if row.get("Status"):
            account["statuses"].add(row["Status"])
        if row.get("Status Reason"):
            account["reasons"].add(row["Status Reason"])
    output = {}
    negative_reasons = {
        "BDR/Partner-Not Interested", "BDR - No Budget/No Need", "BDR - Account Not a Fit",
        "Not Interested", "No Budget/No Need", "Account Not a Fit",
    }
    for key, row in grouped.items():
        positive = "Qualified" in row["statuses"] or bool(
            {"Qualified", "Meeting Booked"} & row["reasons"]
        )
        negative = bool(negative_reasons & row["reasons"]) and not positive
        output[key] = {
            "qualified": positive, "negative": negative,
            "eligible": positive or negative, "account_names": sorted(row["names"]),
        }
    return output


def stratified_accounts(rows, *, positive_share, draws=1000, seed=42, ks=(10, 20)):
    """Paired sample draws and randomized score ties; zeros are retained."""
    rows = list(rows)
    if not 0 < positive_share < 1 or draws <= 0:
        raise ValueError("Need 0 < positive_share < 1 and positive draws.")
    if len({row["account_id"] for row in rows}) != len(rows):
        raise ValueError("Evaluate one row per account, not repeated contacts/products.")
    if any(type(row["qualified"]) is not bool for row in rows):
        raise ValueError("Qualified labels must be booleans.")
    if any(not math.isfinite(row["score"]) for row in rows):
        raise ValueError("Non-finite account score.")
    positives = [row for row in rows if row["qualified"]]
    negatives = [row for row in rows if not row["qualified"]]
    n_negative = min(150, len(negatives))
    n_positive = min(len(positives), max(2, round(n_negative * positive_share / (1 - positive_share))))
    if n_positive < 2 or n_negative < 5:
        raise ValueError("Stratified evaluation needs at least two positive and five negative accounts.")
    metrics = defaultdict(list)
    random_metrics = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(draws):
        pool = rng.sample(positives, n_positive) + rng.sample(negatives, n_negative)
        rng.shuffle(pool)
        ranked = sorted(pool, key=lambda row: -row["score"])
        random_order = rng.sample(pool, len(pool))
        for k in ks:
            width = min(k, len(pool))
            if width <= 0:
                raise ValueError("Rank cutoffs must be positive.")
            metrics[k].append(sum(row["qualified"] for row in ranked[:width]) / width)
            random_metrics[k].append(sum(row["qualified"] for row in random_order[:width]) / width)
    base = n_positive / (n_positive + n_negative)
    return {
        "unit": "account", "draws": draws, "seed": seed,
        "sampling": "stratified_without_replacement_within_each_draw",
        "tie_policy": "seeded_label_blind_shuffle_before_stable_score_sort",
        "n_positive_pool": len(positives), "n_negative_pool": len(negatives),
        "n_positive_per_draw": n_positive, "n_negative_per_draw": n_negative,
        "requested_positive_share": positive_share, "actual_positive_share": base,
        "metrics": {
            str(k): {
                "precision": statistics.mean(values),
                "precision_std": statistics.pstdev(values),
                "random_precision": statistics.mean(random_metrics[k]),
                "lift_over_expected_random": statistics.mean(values) / base,
            }
            for k, values in metrics.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)
    pitch = sub.add_parser("pitch")
    pitch.add_argument("--ratings", type=Path, required=True)
    pitch.add_argument("--out", type=Path, required=True)
    accounts = sub.add_parser("accounts")
    accounts.add_argument("--crm", type=Path, required=True)
    accounts.add_argument("--scores", type=Path, required=True)
    accounts.add_argument("--positive-share", type=float, required=True)
    accounts.add_argument("--draws", type=int, default=1000)
    accounts.add_argument("--seed", type=int, default=42)
    accounts.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Output already exists.")
    if args.task == "pitch":
        rows = list(lines(args.ratings))
        if len({row["triple_id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate benchmark triple.")
        if not rows:
            raise ValueError("No benchmark triples.")
        per_triple = {row["triple_id"]: pitch_coverage(row["ratings"]) for row in rows}
        report = {"per_triple": per_triple, "coverage": statistics.mean(per_triple.values())}
    else:
        with args.crm.open(encoding="utf-8-sig", newline="") as stream:
            labels = qualification_labels(csv.DictReader(stream))
        rows, seen = [], set()
        for row in lines(args.scores):
            key = account_key(row["account_name"])
            if key in seen:
                raise ValueError("Duplicate scored account: " + key)
            seen.add(key)
            if key not in labels:
                raise ValueError("Scored account absent from supplied CRM: " + key)
            if labels[key]["eligible"]:
                rows.append({"account_id": key, "qualified": labels[key]["qualified"], "score": row["score"]})
        report = stratified_accounts(
            rows, positive_share=args.positive_share, draws=args.draws, seed=args.seed,
        )
    save(args.out, report)


if __name__ == "__main__":
    main()
