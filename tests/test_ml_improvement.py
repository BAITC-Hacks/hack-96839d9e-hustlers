"""No-regression decisions and geometry invariants; fixtures are not real data."""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from windops.ml.features import prepare_features
from test_ml import weather_rows


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("improve_model")


def scores(factor=1.):
    return {band: {"n_used": n, "mae": mae * factor, "rmse": rmse * factor}
            for band, n, mae, rmse in (("all", 100, .2, .3), ("1-24", 50, .18, .28), ("25-48", 50, .22, .32))}


def test_promotion_requires_material_gain_and_no_regression(experiment):
    assert experiment.gate(scores(.95), scores(), .02)["passed"]
    result = experiment.gate(scores(.995), scores(), .02)
    assert not result["passed"] and any("below" in reason for reason in result["reasons"])


@pytest.mark.parametrize("band,metric", [("all", "rmse"), ("1-24", "mae"), ("25-48", "rmse")])
def test_average_gain_does_not_hide_horizon_or_metric_degradation(experiment, band, metric):
    candidate = scores(.95)
    candidate[band][metric] = scores()[band][metric] + .001
    result = experiment.gate(candidate, scores(), .02)
    assert not result["passed"] and f"{band}: {metric} degraded" in result["reasons"]


def test_comparison_cannot_pass_on_different_pair_counts(experiment):
    candidate = scores(.9)
    candidate["25-48"]["n_used"] = 49
    assert not experiment.gate(candidate, scores(), .02)["passed"]


def test_geometry_values_zero_wind_and_permutation():
    rows = weather_rows(horizon=24)
    rows[0].update(wind_u_10m_ms=3., wind_v_10m_ms=4., wind_speed_10m_ms=5.,
                   wind_u_100m_ms=-6., wind_v_100m_ms=8., wind_speed_100m_ms=10.)
    for name in ("wind_u_10m_ms", "wind_v_10m_ms", "wind_speed_10m_ms", "wind_u_100m_ms", "wind_v_100m_ms", "wind_speed_100m_ms"):
        rows[1][name] = 0.
    x = prepare_features(rows, "weather_hour_lead_geometry")
    assert x.iloc[0].wind_unit_u_10m == pytest.approx(.6)
    assert x.iloc[0].wind_unit_v_100m == pytest.approx(.8)
    assert x.iloc[0].wind_vector_alignment == pytest.approx(.28)
    assert x.iloc[0].wind_speed_vertical_ratio == pytest.approx(10 / 5.5)
    assert x.iloc[0].wind_speed_vertical_difference == 5
    assert x.iloc[1].wind_vector_alignment == 0 and x.iloc[1].wind_speed_vertical_ratio == 0
    assert np.isfinite(x.to_numpy()).all()
    reverse = prepare_features(list(reversed(rows)), "weather_hour_lead_geometry")
    pd.testing.assert_frame_equal(reverse, x.iloc[::-1].reset_index(drop=True))


def test_protected_artifact_change_prevents_promotion(experiment, tmp_path):
    path = tmp_path / "saved_model.cbm"
    path.write_bytes(b"explicit-test-artifact")
    from windops.core import digest
    plan = {"identity": {"protected_files": {str(path): digest(path.read_bytes())}}}
    experiment.unchanged(plan)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Protected production artifact changed"):
        experiment.unchanged(plan)
