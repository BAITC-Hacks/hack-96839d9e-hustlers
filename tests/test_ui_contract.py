import json

import pandas as pd
import pytest

from windops.ui.adapter import AdapterError, BackendAdapter, combine_bundles, load_bundle_json, validate_bundle
from windops.ui.exports import ExportBlocked, selected_csv
from windops.ui.fixtures import synthetic_bundle


@pytest.mark.parametrize("horizon", [24, 48])
@pytest.mark.parametrize("sites", [("site-a",), ("site-a", "site-b")])
def test_exact_hourly_contract_without_optional_quantiles(make_bundle, horizon, sites):
    bundle = make_bundle(horizon=horizon, sites=sites)
    result = validate_bundle(bundle)
    assert result.exportable, result.issues
    assert len(bundle.rows) == horizon * len(sites)
    assert "p50" not in bundle.rows


@pytest.mark.parametrize("problem", ["missing_hour", "duplicate_hour", "same_count_wrong_grid", "missing_site", "extra_site"])
def test_grid_errors_block_export(make_bundle, problem):
    bundle = make_bundle()
    if problem == "missing_hour":
        bundle.rows = bundle.rows.drop(index=3)
    elif problem == "duplicate_hour":
        bundle.rows = pd.concat([bundle.rows, bundle.rows.iloc[[3]]], ignore_index=True)
    elif problem == "same_count_wrong_grid":
        bundle.rows.loc[3, "target_time"] = bundle.rows.loc[4, "target_time"]
    elif problem == "missing_site":
        bundle.rows = bundle.rows[bundle.rows.site_id == "site-a"]
    else:
        bundle.rows.loc[0, "site_id"] = "unrequested-site"
    result = validate_bundle(bundle)
    assert not result.exportable
    assert result.issues
    with pytest.raises(ExportBlocked):
        selected_csv(bundle, final=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -0.01, 1.01])
def test_invalid_numbers_are_rejected_without_clipping(make_bundle, value):
    bundle = make_bundle()
    bundle.rows.loc[0, "prediction"] = value
    assert not validate_bundle(bundle).exportable
    if pd.isna(value):
        assert pd.isna(bundle.rows.loc[0, "prediction"])
    else:
        assert bundle.rows.loc[0, "prediction"] == value


@pytest.mark.parametrize("field", ["prediction", "p10", "horizon_step", "lead_hours"])
def test_boolean_is_not_a_power_value_or_hour(make_bundle, field):
    bundle = make_bundle()
    if field not in bundle.rows:
        bundle.rows[field] = 0.1
    bundle.rows[field] = bundle.rows[field].astype(object)
    bundle.rows.loc[0, field] = True
    assert not validate_bundle(bundle).exportable


@pytest.mark.parametrize("problem", ["missing_column", "unknown_zone", "missing_zone", "naive_timestamp", "wrong_step", "wrong_lead", "no_convention"])
def test_required_fields_and_time_convention(make_bundle, problem):
    bundle = make_bundle()
    if problem == "missing_column":
        bundle.rows = bundle.rows.drop(columns="model_version")
    elif problem == "unknown_zone":
        bundle.timezone = "NoSuch/Zone"
    elif problem == "missing_zone":
        bundle.timezone = None
    elif problem == "naive_timestamp":
        bundle.rows.loc[0, "target_time"] = "2026-02-01T00:00:00"
    elif problem == "wrong_step":
        bundle.rows.loc[0, "horizon_step"] = 2
    elif problem == "wrong_lead":
        bundle.rows.loc[0, "lead_hours"] = 2
    else:
        bundle.provenance.pop("timestamp_convention")
    assert not validate_bundle(bundle).exportable


def test_timezone_offset_represents_same_absolute_grid(make_bundle):
    bundle = make_bundle(origin="2026-02-01T04:00:00+05:00")
    bundle.timezone = "Asia/Qyzylorda"
    assert validate_bundle(bundle).exportable


def test_quantiles_validated_independently_of_point_forecast(make_bundle):
    bundle = make_bundle()
    bundle.rows["p10"] = 0.2
    bundle.rows["p50"] = 0.4
    bundle.rows["p90"] = 0.8
    # Point forecasts are deliberately below p10; prediction is not implicitly P50.
    assert validate_bundle(bundle).exportable
    bundle.rows.loc[0, "p10"] = 0.5
    assert not validate_bundle(bundle).exportable


def test_two_present_quantile_bounds_are_ordered(make_bundle):
    bundle = make_bundle()
    bundle.rows["p10"] = 0.7
    bundle.rows["p90"] = 0.6
    assert not validate_bundle(bundle).exportable


def test_entirely_absent_optional_quantile_is_allowed(make_bundle):
    bundle = make_bundle()
    bundle.rows["p10"] = None
    bundle.rows["p50"] = None
    bundle.rows["p90"] = None
    assert validate_bundle(bundle).exportable
    header = selected_csv(bundle, final=True).decode("utf-8-sig").splitlines()[0].split(",")
    assert not {"p10", "p50", "p90"}.intersection(header)


def test_partially_missing_quantile_is_not_fabricated(make_bundle):
    bundle = make_bundle()
    bundle.rows["p10"] = 0.1
    bundle.rows.loc[0, "p10"] = float("nan")
    assert not validate_bundle(bundle).exportable


@pytest.mark.parametrize("problem", ["future_weather", "future_cutoff", "unverified_inputs", "unverified_model", "ineligible_model", "actual_weather", "missing_availability", "mixed_synthetic", "synthetic_row", "untrusted_upload"])
def test_historical_evidence_and_trust_gate(make_bundle, problem):
    bundle = make_bundle()
    if problem == "future_weather":
        bundle.provenance["weather"]["available_at"] = "2026-02-01T00:00:00+00:00"
    elif problem == "future_cutoff":
        bundle.provenance["model"]["training_cutoff"] = "2026-02-01T00:00:00+00:00"
    elif problem == "unverified_inputs":
        bundle.validation_summary["inputs_verified"] = False
    elif problem == "unverified_model":
        bundle.validation_summary["model_verified"] = False
    elif problem == "ineligible_model":
        bundle.provenance["model"]["historically_eligible"] = False
    elif problem == "actual_weather":
        bundle.provenance["weather"]["data_kind"] = "observations"
    elif problem == "missing_availability":
        bundle.provenance["weather"].pop("available_at")
    elif problem == "mixed_synthetic":
        bundle.provenance["weather"]["synthetic"] = True
    elif problem == "synthetic_row":
        bundle.rows.loc[0, "quality_flag"] = "synthetic"
    else:
        bundle.verification_origin = "uploaded_unverified"
    with pytest.raises(ExportBlocked):
        selected_csv(bundle, final=True)


def test_recent_download_does_not_invalidate_historically_available_archive(make_bundle):
    bundle = make_bundle()
    assert validate_bundle(bundle).exportable


def test_demo_is_blocked_programmatically():
    bundle = synthetic_bundle()
    assert bundle.source_mode == "demo"
    with pytest.raises(ExportBlocked):
        selected_csv(bundle, final=True)
    assert not bundle.events, "Fixture must not fabricate agent execution events"


def test_upload_cannot_self_assert_backend_trust(make_bundle):
    bundle = make_bundle()
    payload = {
        "forecast_id": bundle.forecast_id, "rows": bundle.rows.to_dict("records"),
        "validation_summary": bundle.validation_summary, "provenance": bundle.provenance,
        "report_facts": [], "site_ids": bundle.site_ids, "horizon_hours": bundle.horizon_hours,
        "timezone": bundle.timezone, "source_mode": "cached", "verification_origin": "backend",
    }
    loaded = load_bundle_json(json.dumps(payload))
    assert loaded.verification_origin == "uploaded_unverified"
    assert not validate_bundle(loaded).exportable


def test_combine_keeps_original_identifiers_and_validates_every_member(make_bundle):
    first = make_bundle(sites=("site-a",), forecast_id="turbine-a")
    second = make_bundle(sites=("site-b",), forecast_id="turbine-b")
    group = combine_bundles([first, second])
    assert set(group.site_ids) == {"site-a", "site-b"}
    assert set(group.rows.forecast_id) == {"turbine-a", "turbine-b"}
    assert len(group.rows) == 96
    assert validate_bundle(group).exportable
    group.rows.loc[0, "prediction"] = 0.8
    assert not validate_bundle(group).exportable, "Group must not silently rewrite original forecasts"


def test_group_cannot_promote_unverified_member(make_bundle):
    first = make_bundle(sites=("site-a",), forecast_id="a")
    second = make_bundle(sites=("site-b",), forecast_id="b")
    second.verification_origin = "uploaded_unverified"
    group = combine_bundles([first, second])
    assert not validate_bundle(group).exportable


def test_group_allows_quantiles_for_only_one_turbine(make_bundle):
    first = make_bundle(sites=("site-a",), forecast_id="a")
    second = make_bundle(sites=("site-b",), forecast_id="b")
    first.rows["p10"] = 0.1
    first.rows["p90"] = 0.8
    group = combine_bundles([first, second])
    assert validate_bundle(group).exportable
    # Missing one supplied value is invalid, even when the other site has no band.
    first.rows.loc[0, "p10"] = float("nan")
    assert not validate_bundle(combine_bundles([first, second])).exportable


def test_quantile_order_checked_only_where_both_bounds_supplied(make_bundle):
    bundle = make_bundle()
    bundle.rows["p10"] = float("nan")
    bundle.rows["p90"] = float("nan")
    bundle.rows.loc[bundle.rows.site_id == "site-a", "p10"] = 0.2
    bundle.rows.loc[bundle.rows.site_id == "site-b", "p90"] = 0.8
    assert validate_bundle(bundle).exportable
    bundle.rows.loc[bundle.rows.site_id == "site-a", "p90"] = 0.1
    assert not validate_bundle(bundle).exportable


def test_missing_backend_stays_missing_without_demo_fallback():
    adapter = BackendAdapter("windops_backend_that_does_not_exist")
    assert not any(adapter.capabilities.values())
    with pytest.raises(AdapterError):
        adapter.run_forecast(["site-a"], "2026-01-31T23:00:00Z", 24)
