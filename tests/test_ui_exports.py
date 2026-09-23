from io import BytesIO
import json

import pandas as pd
import pytest

from windops.ui.adapter import combine_bundles
from windops.ui.exports import ExportBlocked, february_csv, passport_json, replay_csv, selected_csv


def read_csv(content):
    return pd.read_csv(BytesIO(content), float_precision="round_trip")


def test_selected_csv_preserves_ids_timestamps_and_numeric_precision(make_bundle):
    bundle = make_bundle(horizon=24)
    actual = read_csv(selected_csv(bundle, final=True)).sort_values(["site_id", "target_time"])
    expected = bundle.rows.sort_values(["site_id", "target_time"])
    for column in ["run_id", "site_id", "model_version", "weather_version", "horizon_step", "prediction"]:
        assert actual[column].tolist() == expected[column].tolist()
    for column in ["issued_at", "target_time"]:
        assert pd.to_datetime(actual[column], utc=True).tolist() == pd.to_datetime(expected[column], utc=True).tolist()
        assert actual[column].str.contains(r"(?:Z|[+-]\d\d:\d\d)$", regex=True).all()
    assert "p50" not in actual
    assert "0.124456789012345" in selected_csv(bundle, final=True).decode("utf-8-sig")


def test_export_reruns_validation_after_bundle_mutation(make_bundle):
    bundle = make_bundle()
    selected_csv(bundle, final=True)
    bundle.rows.loc[0, "prediction"] = float("inf")
    with pytest.raises(ExportBlocked):
        selected_csv(bundle, final=True)


def test_group_csv_preserves_turbine_specific_optional_quantiles(make_bundle):
    first = make_bundle(sites=("site-a",), forecast_id="a")
    second = make_bundle(sites=("site-b",), forecast_id="b")
    first.rows["p10"] = 0.123456789012345
    first.rows["p90"] = 0.8
    group = combine_bundles([first, second])
    exported = read_csv(selected_csv(group, final=True))
    assert len(exported) == 96
    assert exported.loc[exported.site_id == "site-a", "p10"].eq(0.123456789012345).all()
    assert exported.loc[exported.site_id == "site-b", ["p10", "p90"]].isna().all().all()
    assert set(exported.forecast_id) == {"a", "b"}


def test_same_issue_two_versions_preserved(make_bundle):
    first = make_bundle(forecast_id="version-a")
    second = make_bundle(forecast_id="version-b")
    second.rows["prediction"] += 0.05
    exported = read_csv(replay_csv([first, second], final=True))
    assert len(exported) == 192
    assert set(exported.run_id) == {"version-a-run", "version-b-run"}
    assert not exported.duplicated(["run_id", "site_id", "target_time"]).any()


def test_selected_late_february_release_keeps_march_hours(make_bundle):
    bundle = make_bundle(origin="2026-02-28T00:00:00+00:00")
    frame = read_csv(selected_csv(bundle, final=True))
    assert len(frame) == 96
    assert pd.to_datetime(frame.target_time, utc=True).max().month == 3


def month_bundles(make_bundle):
    # Every release predicts the next complete UTC day; later releases overlap.
    return [make_bundle(origin=origin.isoformat(), forecast_id=f"day-{day}")
            for day, origin in enumerate(pd.date_range("2026-01-31T23:00:00Z", periods=28, freq="D"))]


def test_february_grid_and_latest_historical_release(make_bundle):
    bundles = month_bundles(make_bundle)
    frame = read_csv(february_csv(bundles, ["site-a", "site-b"], "UTC",
                                  rule="latest_available_before_target", final=True))
    assert len(frame) == 1344
    times = pd.to_datetime(frame.target_time, utc=True)
    expected = set(pd.date_range("2026-02-01T00:00:00Z", periods=672, freq="h"))
    for site in ("site-a", "site-b"):
        assert set(times[frame.site_id == site]) == expected
    assert not frame.duplicated(["site_id", "target_time"]).any()
    selected = frame[(frame.site_id == "site-a") & times.eq(pd.Timestamp("2026-02-02T00:00:00Z"))]
    assert selected.run_id.tolist() == ["day-1-run"]
    assert (pd.to_datetime(frame.issued_at, utc=True) < times).all()


def test_february_partial_grid_never_passes_as_complete(make_bundle):
    with pytest.raises(ExportBlocked):
        february_csv([make_bundle()], ["site-a", "site-b"], "UTC", final=True)


def test_unknown_february_selection_rule_rejected(make_bundle):
    with pytest.raises((ExportBlocked, ValueError)):
        february_csv([make_bundle()], ["site-a", "site-b"], "UTC", rule="choose_future", final=True)


def test_february_refuses_ambiguous_same_time_recalculation(make_bundle):
    bundles = month_bundles(make_bundle)
    replacement = make_bundle(origin=bundles[1].rows.issued_at.iloc[0], forecast_id="recalculation")
    with pytest.raises(ExportBlocked, match="Неоднозначная"):
        february_csv([*bundles, replacement], ["site-a", "site-b"], "UTC", final=True)


def test_replay_duplicate_run_identifier_rejected(make_bundle):
    first = make_bundle()
    with pytest.raises(ExportBlocked):
        replay_csv([first, first], final=True)


def test_replay_duplicate_absolute_hour_with_different_offsets_rejected(make_bundle):
    first = make_bundle(forecast_id="utc-version")
    second = make_bundle(forecast_id="offset-version", origin="2026-02-01T04:00:00+05:00")
    second.rows["run_id"] = first.rows["run_id"]
    with pytest.raises(ExportBlocked):
        replay_csv([first, second], final=True)


def test_passport_redacts_credentials_and_retains_evidence(make_bundle):
    bundle = make_bundle()
    bundle.provenance["api_key"] = "TEST_SECRET_VALUE"
    bundle.events = [{"status": "error", "authorization": "Bearer TEST_CREDENTIAL",
                      "message": "api_key=TEST_SECRET_VALUE request failed"}]
    content = passport_json(bundle).decode("utf-8")
    assert "TEST_SECRET_VALUE" not in content
    assert "TEST_CREDENTIAL" not in content
    assert "training_cutoff" in content
    assert json.loads(content)
