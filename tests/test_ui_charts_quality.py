import math

import pandas as pd
import pytest

from windops.ui.charts import forecast_chart
from windops.ui.quality import evaluate_backtest, load_backtest


def test_chart_reindexes_missing_hour_as_gap(make_bundle):
    rows = make_bundle(sites=("site-a",), horizon=24).rows.drop(index=5)
    chart = forecast_chart(rows, "UTC", {"site-a": "Турбина 1"})
    trace = next(trace for trace in chart.data if "Прогноз" in trace.name)
    assert len(trace.y) == 24
    assert pd.isna(trace.y[5])
    assert not trace.connectgaps
    assert "P50" not in trace.name
    assert chart.layout.yaxis.range == (0, 1)


def test_february_never_gets_actual_trace_or_fabricated_interval(make_bundle):
    rows = make_bundle(sites=("site-a",)).rows
    rows["actual"] = 0.99
    chart = forecast_chart(rows, "UTC", {"site-a": "Турбина 1"}, actuals=True)
    assert len(chart.data) == 1
    assert "Факт" not in chart.data[0].name
    assert "P50" not in chart.data[0].name


def test_only_supplied_quantiles_produce_interval(make_bundle):
    rows = make_bundle(sites=("site-a",)).rows
    rows["p10"], rows["p90"] = 0.1, 0.8
    chart = forecast_chart(rows, "UTC", {"site-a": "Турбина 1"})
    assert any(trace.fill == "tonexty" for trace in chart.data)
    assert not any("P50" in trace.name for trace in chart.data)


def backtest_payload():
    origin = pd.Timestamp("2026-01-10T00:00:00Z")
    return {
        "schema_version": 1, "kind": "january_backtest", "data_mode": "real_saved",
        "timezone": "UTC", "methodology": "Test matched forecast/observation pairs only.",
        "provenance": {"data_source": "contract-fixture-no-real-observations"},
        "rows": [
            {"run_id": "january-run", "site_id": "site-a", "issued_at": origin.isoformat(),
             "target_time": (origin + pd.Timedelta(hours=step)).isoformat(),
             "horizon_step": step, "prediction": predicted, "actual": actual,
             "model_version": "january-test", "training_cutoff": "2025-12-31T00:00:00Z",
             "baselines": {"persistence": baseline}}
            for step, predicted, actual, baseline in [
                (1, 0.2, 0.4, 0.0), (2, 0.8, 0.4, None), (3, 0.3, None, 0.9),
                (25, 0.6, 0.4, 0.4),
            ]
        ],
    }


def test_quality_uses_matched_pairs_and_same_baseline_cohort():
    metrics = evaluate_backtest(load_backtest(backtest_payload()))
    first = metrics[metrics.horizon_band == "1–24"].set_index("comparison")
    assert first.loc["model", "n_used"] == 2
    assert first.loc["model", "n_excluded"] == 1
    assert first.loc["model", "model_mae"] == pytest.approx(0.3)
    assert first.loc["model", "model_rmse"] == pytest.approx(math.sqrt(0.1))
    assert first.loc["persistence", "n_used"] == 1
    assert first.loc["persistence", "model_mae"] == pytest.approx(0.2)
    assert first.loc["persistence", "baseline_mae"] == pytest.approx(0.4)
    assert first.loc["persistence", "mae_improvement_pct"] == pytest.approx(50)
    long = metrics[(metrics.horizon_band == "25–48") & (metrics.comparison == "persistence")].iloc[0]
    assert pd.isna(long.mae_improvement_pct), "Zero baseline error cannot produce an improvement percentage"


def test_worsening_is_reported_as_negative_improvement():
    payload = backtest_payload()
    payload["rows"][0]["baselines"] = {"persistence": 0.3}
    metrics = evaluate_backtest(load_backtest(payload))
    value = metrics[(metrics.horizon_band == "1–24") & (metrics.comparison == "persistence")].iloc[0]
    assert value.mae_improvement_pct == pytest.approx(-100)


@pytest.mark.parametrize("problem", ["future_cutoff", "february", "duplicate", "naive_time", "demo"])
def test_invalid_backtest_rejected(problem):
    payload = backtest_payload()
    if problem == "future_cutoff":
        payload["rows"][0]["training_cutoff"] = "2026-01-31T00:00:00Z"
    elif problem == "february":
        payload["rows"][0]["target_time"] = "2026-02-01T00:00:00Z"
    elif problem == "duplicate":
        payload["rows"].append(dict(payload["rows"][0]))
    elif problem == "naive_time":
        payload["rows"][0]["issued_at"] = "2026-01-10T00:00:00"
    else:
        payload["data_mode"] = "demo"
    with pytest.raises(ValueError):
        load_backtest(payload)


@pytest.mark.parametrize("invalid", [float("inf"), float("nan"), True, 2.0])
def test_quality_excludes_invalid_pairs_without_zero_filling(invalid):
    payload = backtest_payload()
    payload["rows"][0]["actual"] = invalid
    metrics = evaluate_backtest(load_backtest(payload))
    first = metrics[(metrics.horizon_band == "1–24") & (metrics.comparison == "model")].iloc[0]
    assert first.n_used == 1
    assert first.n_excluded == 2
    assert first.model_mae == pytest.approx(0.4)
