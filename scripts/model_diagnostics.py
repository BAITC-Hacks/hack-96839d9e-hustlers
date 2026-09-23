"""Post-hoc diagnosis of saved January errors; no fitting, tuning or calibration.

Observed SCADA wind is read only to describe errors, never as a model feature.
Outputs are an audit report and a standalone figure, not new model artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from windops.core import atomic_json, data_root, digest, read_json
from windops.ml.features import prepare_features
from windops.ml.models import find_version, model_root
from windops.ui.quality import load_backtest

ZONE = ZoneInfo("Asia/Almaty")


def summarize(frame, prediction="prediction"):
    error = frame[prediction] - frame.actual
    return {"pairs": len(frame), "mae": float(error.abs().mean()),
            "rmse": float(np.sqrt(error.pow(2).mean())), "bias_prediction_minus_actual": float(error.mean()),
            "actual_mean": float(frame.actual.mean()), "prediction_mean": float(frame[prediction].mean()),
            "absolute_error_p90": float(error.abs().quantile(.9)),
            "absolute_error_p95": float(error.abs().quantile(.95)),
            "absolute_error_max": float(error.abs().max()),
            "pairs_absolute_error_over_0_50": int(error.abs().gt(.5).sum()),
            "fraction_absolute_error_over_0_50": float(error.abs().gt(.5).mean())}


def diagnose(root, output):
    output.mkdir(parents=True, exist_ok=True)
    frozen = [p for p in model_root().glob("*/ml-*/*") if p.is_file()]
    frozen += [root / "ml" / n for n in ("selection.json", "january_backtest.json", "january_metrics.json", "february_hourly.csv")]
    before = {str(p): digest(p.read_bytes()) for p in frozen}
    artifact = load_backtest((root / "ml/january_backtest.json").read_bytes())
    rows = artifact.rows
    valid = rows.actual_valid & rows.prediction_valid & rows.actual.between(0, 1) & rows.prediction.between(0, 1)
    # This report is deliberately specific to the real, fully valid saved cohort.
    assert len(rows) == 2928 and valid.all()
    weather = pd.read_csv(root / "weather/weather_for_ml.csv")
    for field in ("forecast_origin", "target_time"):
        weather[field] = pd.to_datetime(weather[field], utc=True)
    joined = rows.merge(weather, left_on=["site_id", "issued_at", "target_time"],
                        right_on=["site_id", "forecast_origin", "target_time"],
                        validate="one_to_one", suffixes=("", "_weather"))
    assert len(joined) == len(rows)
    results = {"purpose": "post-hoc diagnosis only; no January tuning, no model changes",
               "unit": "dimensionless normalized hourly mean power, 0–1",
               "normalization_formula_confirmed": False, "rated_power_mw": None,
               "primary_evaluation_unit": "site + issuance + target hour; overlapping releases retained",
               "sites": {}, "frozen_input_checksums": before}
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    for index, (site, part) in enumerate(joined.groupby("site_id", sort=True)):
        part = part.copy()
        part["absolute_error"] = (part.prediction - part.actual).abs()
        daily = part.loc[part.horizon_step.le(24)].sort_values("target_time")
        assert len(daily) == 744 and not daily.target_time.duplicated().any()
        source_path = root / "scada" / f"{site}.csv"
        raw = pd.read_csv(source_path)
        local = pd.to_datetime(raw["Статистическое время"], format="mixed")
        # Restrict BEFORE timezone conversion: pre-March-2024 semantics are outside this audit.
        selected = local.ge("2026-01-01") & local.lt("2026-02-01")
        raw = raw.loc[selected].copy()
        raw["target_time"] = local.loc[selected].dt.tz_localize(ZONE).dt.tz_convert("UTC").dt.floor("h")
        observed = raw.groupby("target_time").agg(
            scada_observations=("Нормализованная активная мощность", "size"),
            scada_power_mean=("Нормализованная активная мощность", "mean"),
            observed_scada_wind_ms=("Средняя скорость ветра(m/s)", "mean"))
        cases = part.nlargest(5, "absolute_error").merge(observed, left_on="target_time", right_index=True, validate="many_to_one")
        np.testing.assert_allclose(cases.actual, cases.scada_power_mean, atol=1e-12, rtol=0)
        assert cases.scada_observations.eq(6).all()
        case_columns = ["issued_at", "target_time", "horizon_step", "actual", "prediction", "absolute_error",
                        "wind_speed_10m_ms", "wind_speed_100m_ms", "observed_scada_wind_ms", "scada_observations"]
        case_output = cases[case_columns].copy()
        for name in ("issued_at", "target_time"):
            case_output[name] = case_output[name].dt.tz_convert(ZONE).map(lambda t: t.isoformat())
        bands = {}
        for name, mask in (("actual_0_to_0.05", part.actual.le(.05)),
                           ("actual_over_0.05_under_0.95", part.actual.gt(.05) & part.actual.lt(.95)),
                           ("actual_0.95_to_1", part.actual.ge(.95))):
            bands[name] = summarize(part.loc[mask])
        results["sites"][site] = {"all_pairs": summarize(part),
            "horizon_1_24": summarize(daily), "horizon_25_48": summarize(part.loc[part.horizon_step.gt(24)]),
            "constant_median": summarize(part, "baseline__constant_median"),
            "wind_table": summarize(part, "baseline__wind_table"),
            "by_actual_power": bands,
            "prediction_range": [float(part.prediction.min()), float(part.prediction.max())],
            "unique_predictions": int(part.prediction.nunique()),
            "largest_errors": case_output.to_dict("records"),
            "observed_wind_caveat": "Diagnostic only. SCADA sensor height is unconfirmed; do not equate to GFS 10/100 m or infer a shutdown cause."}
        results["sites"][site]["scada_sha256"] = digest(source_path.read_bytes())
        ax = axes[index, 0]
        ax.plot(daily.target_time, daily.actual, color="#263238", linewidth=1, label="Факт SCADA")
        ax.plot(daily.target_time, daily.prediction, color="#1976D2", linewidth=1, alpha=.85, label="CatBoost")
        ax.set(title=f"{site}: январь, шаги 1–24 исходного выпуска", ylabel="Нормализованная мощность 0–1", ylim=(-.03, 1.03))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5, tz=ZONE))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m", tz=ZONE))
        ax.set_xlabel("Начало целевого часа · Asia/Almaty")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=.2)
        ax = axes[index, 1]
        for column, label in (("prediction", "CatBoost"), ("baseline__constant_median", "Постоянная медиана"), ("baseline__wind_table", "Ветровая таблица")):
            error = np.sort((part[column] - part.actual).abs().to_numpy())
            ax.step(error, np.arange(1, len(error) + 1) / len(error), label=label)
        ax.set(title=f"{site}: ошибки всех 1464 январских пар", xlabel="Абсолютная ошибка в шкале 0–1", ylabel="Доля пар с ошибкой не больше X", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("Независимая январская проверка: реальные ошибки сохранённых моделей\nСлева один прогноз на час; справа все выпуски. Диагностика без настройки по январю.", fontsize=13)
    fig.savefig(output / "january_quality.png", dpi=150)
    plt.close(fig)
    february = pd.read_csv(root / "ml/february_hourly.csv")
    results["february"] = {"rows": len(february), "actual_available": False, "mae": None, "rmse": None, "sites": {}}
    for site, part in february.groupby("site_id"):
        version = part.model_version.unique()
        assert len(version) == 1
        _, card = find_version(site, version[0])
        matching = weather.loc[weather.site_id.eq(site) & weather.forecast_origin.between(pd.Timestamp("2026-01-31T18:00:00Z"), pd.Timestamp("2026-02-28T18:00:00Z"))]
        x = prepare_features(matching, card["config"]["feature_set"])
        assert np.isfinite(part.prediction).all() and part.prediction.between(0, 1).all()
        results["february"]["sites"][site] = {
            "prediction_min": float(part.prediction.min()), "prediction_max": float(part.prediction.max()),
            "unique_predictions": int(part.prediction.nunique()),
            "full_release_weather_pairs": len(x),
            "features_outside_training_range": {name: int((~x[name].between(limits["min"], limits["max"])).sum()) for name, limits in card["feature_ranges"].items()}}
    assert before == {str(p): digest(p.read_bytes()) for p in frozen}
    results["model_and_original_results_unchanged"] = True
    atomic_json(output / "report.json", results)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data_root())
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model_review"))
    args = parser.parse_args()
    result = diagnose(args.data_dir, args.output_dir)
    for site, item in result["sites"].items():
        print(site, item["all_pairs"])
    print("Saved report.json and january_quality.png to", args.output_dir)
