"""Independent contract examples; these are not real turbine observations."""

import pandas as pd
import pytest

from windops.ui.adapter import ForecastBundle


@pytest.fixture
def make_bundle():
    def make(*, horizon=48, sites=("site-a", "site-b"), origin="2026-01-31T23:00:00+00:00", forecast_id="forecast-1"):
        issued = pd.Timestamp(origin)
        rows = [
            {
                "run_id": forecast_id + "-run",
                "site_id": site,
                "issued_at": issued.isoformat(),
                "target_time": (issued + pd.Timedelta(hours=step)).isoformat(),
                "horizon_step": step,
                "lead_hours": step,
                "prediction": 0.123456789012345 + step / 1000 + site_index / 100,
                "model_version": "model-contract-test",
                "weather_version": "weather-contract-test",
                "quality_flag": "ok",
            }
            for site_index, site in enumerate(sites)
            for step in range(1, horizon + 1)
        ]
        return ForecastBundle(
            forecast_id=forecast_id,
            rows=pd.DataFrame(rows),
            validation_summary={"inputs_verified": True, "model_verified": True},
            provenance={
                "timestamp_convention": "target_time = issued_at + horizon_step hours",
                "weather": {
                    "source": "contract-test-only",
                    "version": "weather-contract-test",
                    "available_at": (issued - pd.Timedelta(hours=1)).isoformat(),
                    "issued_at": (issued - pd.Timedelta(hours=2)).isoformat(),
                    "retrieved_at": "2026-09-01T00:00:00+00:00",
                    "data_kind": "archived_forecast",
                    "synthetic": False,
                },
                "model": {
                    "version": "model-contract-test",
                    "training_cutoff": (issued - pd.Timedelta(days=1)).isoformat(),
                    "normalization": "0_1",
                    "historically_eligible": True,
                },
            },
            report_facts=[],
            horizon_hours=horizon,
            site_ids=list(sites),
            timezone="UTC",
            source_mode="cached",
            executor="test_fixture",
            verification_origin="backend",
        )

    return make
