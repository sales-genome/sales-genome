import pytest

from paper_submission.analyze import profiles
from paper_submission.audit_package import seal_registry, verify_registry
from paper_submission.downstream import account_score
from paper_submission.temporal import eligible_sources_before


def analysis_fixture():
    spans = [
        {"span_id": "s1", "text": "Faster service", "occurrences": [
            {"vendor_slug": "v1", "url": "https://example.org/one", "page_type": "case_study"},
            {"vendor_slug": "v1", "url": "https://example.org/two", "page_type": "case_study"},
            {"vendor_slug": "v2", "url": "https://example.org/three", "page_type": "case_study"},
        ]},
        {"span_id": "s2", "text": " FASTER   service ", "vendor_slug": "v1",
         "url": "https://example.org/ambiguous", "page_type": "case_study"},
        {"span_id": "s3", "text": "Simple onboarding", "vendor_slug": "v1",
         "url": "https://example.org/product", "page_type": "product_page"},
    ]
    assignments = [
        {"span_id": span, "strategy_id": strategy, "family_id": family, "subtheme_id": subtheme}
        for span, strategy, family, subtheme in [
            ("s1", "strategy1", "family1", "subtheme1"),
            ("s1", "strategy2", "family1", "subtheme1"),
            ("s2", "strategy1", "family1", "subtheme1"),
            ("s3", "strategy3", "family2", "subtheme2"),
        ]
    ]
    sources = [
        {"vendor_slug": vendor, "url": "https://example.org/" + suffix, "customer_ids": customers}
        for vendor, suffix, customers in [
            ("v1", "one", ["c1"]), ("v1", "two", ["c1"]),
            ("v2", "three", ["c2"]), ("v1", "ambiguous", ["c1", "c2"]),
        ]
    ]
    audiences = [
        {"customer_id": customer, "audience_id": "audience1"} for customer in ("c1", "c2")
    ]
    return spans, assignments, sources, audiences


def test_persuasion_quotations_occurrences_and_exact_customer_joins():
    result = profiles(*analysis_fixture())
    assert result["seller_profiles"]["v1"]["counts"] == {"family1": 1, "family2": 1}
    assert result["seller_profiles"]["v1"]["shares"] == {"family1": 0.5, "family2": 0.5}
    assert result["seller_profiles"]["v2"]["total"] == 1
    assert result["audience_profiles"]["audience1"]["counts"] == {"family1": 1}
    assert result["matched_case_spans"] == 1
    assert result["matched_customers"] == result["matched_sellers"] == 2
    assert result["multi_seller_strategies"] == 2
    assert result["strategies"] == 3
    assert result["ambiguous_source_groups_excluded"] == 1


def test_persuasion_rejects_unknown_spans_and_conflicting_membership():
    spans, assignments, sources, audiences = analysis_fixture()
    with pytest.raises(ValueError, match="unknown"):
        profiles(spans, [{**assignments[0], "span_id": "unknown"}], sources, audiences)
    with pytest.raises(ValueError, match="Overlapping"):
        profiles(spans, assignments + [{**assignments[0], "family_id": "other"}], sources, audiences)
    with pytest.raises(ValueError, match="Strategy ID"):
        profiles(spans, assignments + [{
            "span_id": "s3", "strategy_id": "strategy1",
            "family_id": "family2", "subtheme_id": "subtheme2",
        }], sources, audiences)
    with pytest.raises(ValueError, match="Conflicting span"):
        profiles(spans + [{**spans[0], "text": "Different"}], assignments, sources, audiences)


def test_seal_checks_file_set_and_contents(tmp_path):
    (tmp_path / "provenance").mkdir()
    payload = tmp_path / "README.txt"
    payload.write_text("original", encoding="utf-8")
    assert seal_registry(tmp_path) == verify_registry(tmp_path) == 1
    payload.write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        verify_registry(tmp_path)
    payload.write_text("original", encoding="utf-8")
    extra = tmp_path / "unexpected.txt"
    extra.write_text("new", encoding="utf-8")
    with pytest.raises(ValueError, match="file set"):
        verify_registry(tmp_path)
    assert seal_registry(tmp_path) == verify_registry(tmp_path) == 2
    extra.unlink()
    with pytest.raises(ValueError, match="file set"):
        verify_registry(tmp_path)


def test_date_conflicts_are_rejected_even_across_cutoff():
    base = {"vendor_slug": "v1", "url": "https://example.org/one",
            "temporal_use": "eligible", "date_semantics": "editorial_publication"}
    rows = [{**base, "reported_published_date": value} for value in ("2020-01-01", "2022-01-01")]
    with pytest.raises(ValueError, match="Conflicting"):
        eligible_sources_before(rows, "2021-01-01")


def test_seal_excludes_only_root_git_metadata(tmp_path):
    (tmp_path / "provenance").mkdir()
    control = tmp_path / ".git"
    control.mkdir()
    config = control / "config"
    config.write_text("control metadata", encoding="utf-8")
    payload = tmp_path / "README.txt"
    payload.write_text("release payload", encoding="utf-8")
    assert seal_registry(tmp_path) == verify_registry(tmp_path) == 1
    config.write_text("changed control metadata", encoding="utf-8")
    assert verify_registry(tmp_path) == 1
    nested = tmp_path / "nested/.git"
    nested.mkdir(parents=True)
    (nested / "config").write_text("not root control metadata", encoding="utf-8")
    with pytest.raises(ValueError, match="file set"):
        verify_registry(tmp_path)


def test_account_score_accepts_one_pass_grade_iterables():
    grades = ["CONFIRMED", "PROBABLE", "SUGGESTIVE", "INSUFFICIENT"]
    assert account_score(iter(grades)) == account_score(grades)
