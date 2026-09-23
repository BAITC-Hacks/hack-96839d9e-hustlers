"""Independent audit from raw SCADA/cache; never changes models or selection.

Recomputes labels without windops.ml.data and features without
windops.ml.features. Fixed December experiments and saved model fits are
reproduced in memory; January is never used to choose anything.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from windops.core import atomic_json, data_root, digest, load_forecast, read_json
from windops.ml.models import implementation_hash, model_root, read_passport

ZONE = ZoneInfo("Asia/Almaty")
FIELDS = ["wind_speed_10m_ms", "wind_speed_100m_ms", "wind_u_10m_ms", "wind_v_10m_ms",
          "wind_u_100m_ms", "wind_v_100m_ms", "temperature_2m_c"]
CUTOFFS = {"baseline_smoke": "2025-11-30T22:00:00+05:00", "january_evaluation": "2025-12-31T22:00:00+05:00",
           "february_final": "2026-01-31T22:00:00+05:00"}


def same(actual, expected, name):
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True, err_msg=name)


def features(frame, feature_set):
    result = frame[FIELDS].astype(float).copy()
    hours = frame.target_time.dt.tz_convert(ZONE).dt.hour
    result["local_hour_sin"] = np.sin(hours * (2 * np.pi) / 24)
    result["local_hour_cos"] = np.cos(hours * (2 * np.pi) / 24)
    if feature_set == "weather_hour_lead":
        result["lead_hours"] = frame.lead_hours.astype(float)
        result["gfs_lead_hours"] = frame.gfs_lead_hours.astype(float)
    else:
        assert feature_set == "weather_hour"
    return result


def baseline_states(frame):
    groups = frame.groupby(np.floor(frame.wind_speed_100m_ms).astype(int)).actual.median()
    return {"constant_median": {"value": float(frame.actual.median())},
            "wind_table": {"bins": groups.index.tolist(), "values": groups.tolist()}}


def baseline(kind, states, frame):
    if kind == "constant_median":
        return np.full(len(frame), states[kind]["value"])
    occupied = np.array(states[kind]["bins"])
    wanted = np.floor(frame.wind_speed_100m_ms.to_numpy())
    index = np.abs(wanted[:, None] - occupied).argmin(axis=1)
    return np.array(states[kind]["values"])[index]


def score(actual, prediction):
    error = np.asarray(prediction) - np.asarray(actual)
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2)))}


def keys(frame):
    return set(zip(frame.site_id, frame.forecast_origin, frame.target_time))


def check_manifest(root, name, site, expected):
    saved = pd.read_csv(root / "ml/splits" / f"{name}.csv")
    saved = saved.loc[saved.site_id.eq(site)].copy()
    for field in ("forecast_origin", "target_time", "label_available_at"):
        saved[field] = pd.to_datetime(saved[field], utc=True)
    assert len(saved) == len(expected) and keys(saved) == keys(expected), name
    order = ["site_id", "forecast_origin", "target_time"]
    saved = saved.sort_values(order).reset_index(drop=True)
    expected = expected.sort_values(order).reset_index(drop=True)
    assert saved.label_available_at.equals(expected.label_available_at), name
    assert saved.complete.tolist() == expected.complete.tolist(), name


def reconstruct_labels(root, prepared):
    saved = pd.read_csv(root / "ml/hourly_scada.csv")
    for name in ("target_time", "interval_start", "interval_end", "label_available_at"):
        saved[name] = pd.to_datetime(saved[name], utc=True)
    policy = prepared["label_policy"]
    assert policy["timezone"] == "Asia/Almaty" and policy["interval_label"] in ("start", "end")
    shift = timedelta(minutes=10 if policy["interval_label"] == "end" else 0)
    start, end = datetime(2025, 10, 1), datetime(2026, 2, 1)
    targets = pd.date_range(start, end, freq="h", inclusive="left", tz=ZONE).tz_convert("UTC")
    expected_keys = {(site, target) for site in ("turbine_1", "turbine_2") for target in targets}
    assert len(saved) == len(expected_keys) and set(zip(saved.site_id, saved.target_time)) == expected_keys
    rebuilt = []
    for site in ("turbine_1", "turbine_2"):
        groups = defaultdict(list)
        with (root / "scada" / f"{site}.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                # Exact source format, independently checked; no host timezone.
                time = datetime.strptime(row["Статистическое время"], "%Y-%m-%d %H:%M:%S") - shift
                if start <= time < end:
                    value = float(row["Нормализованная активная мощность"])
                    time = pd.Timestamp(time.replace(tzinfo=ZONE)).tz_convert("UTC")
                    groups[time.floor("h")].append((time, value))
        for row in saved.loc[saved.site_id.eq(site)].itertuples():
            observations = groups[row.target_time]
            expected_grid = {row.target_time + pd.Timedelta(minutes=m) for m in range(0, 60, 10)}
            complete = (len(observations) == 6 and {t for t, _ in observations} == expected_grid and
                        all(math.isfinite(y) and 0 <= y <= 1 for _, y in observations))
            actual = math.fsum(y for _, y in observations) / 6 if complete else float("nan")
            assert row.n_observations == len(observations) and row.complete == complete
            same(row.actual, actual, "hourly target from raw SCADA")
            assert row.interval_start == row.target_time
            assert row.interval_end == row.target_time + pd.Timedelta(hours=1)
            assert row.label_available_at == row.interval_end + pd.Timedelta(minutes=policy["scada_delay_minutes"])
            rebuilt.append({"site_id": site, "target_time": row.target_time, "actual": actual,
                            "complete": complete, "label_available_at": row.label_available_at})
    return pd.DataFrame(rebuilt)


def audit(root, models):
    prepared = read_json(root / "ml/prepared.json")
    selection = read_json(root / "ml/selection.json")
    frozen = [root / "ml" / name for name in ("prepared.json", "selection.json", "january_backtest.json", "january_metrics.json",
                                              "february_hourly.csv", "february_full_releases.csv")]
    frozen += [p for p in models.glob("*/ml-*/*") if p.is_file()]
    before = {str(p): digest(p.read_bytes()) for p in frozen}
    assert selection["identity"]["code_sha256"] == implementation_hash(), "Frozen ML implementation changed"
    assert read_json(root / "ml/selection_checksum.json")["sha256"] == digest(selection)
    for path, checksum in prepared["inputs"].items():
        assert digest((root / path).read_bytes()) == checksum, path
    for path, checksum in prepared["outputs"].items():
        assert digest((root / "ml" / path).read_bytes()) == checksum, path
    labels = reconstruct_labels(root, prepared)
    print("Raw SCADA: every hourly target and availability time verified", flush=True)
    weather = pd.read_csv(root / "weather/weather_for_ml.csv")
    for name in ("target_time", "forecast_origin", "run_initialized_at", "availability_upper_bound"):
        weather[name] = pd.to_datetime(weather[name], utc=True)
    origins = pd.date_range("2025-09-30 23:00", "2026-02-28 23:00", freq="D", tz=ZONE).tz_convert("UTC")
    expected = {(site, origin, origin + pd.Timedelta(hours=lead)) for site in ("turbine_1", "turbine_2") for origin in origins for lead in range(1, 49)}
    assert len(weather) == len(expected) == 14592 and keys(weather) == expected
    cached = {}
    for row in weather.itertuples():
        assert row.run_initialized_at <= row.availability_upper_bound <= row.forecast_origin
        assert row.target_time == row.forecast_origin + pd.Timedelta(hours=row.lead_hours)
        assert row.target_time == row.run_initialized_at + pd.Timedelta(hours=row.gfs_lead_hours)
        path = root / "weather/gfs" / row.run_initialized_at.strftime("%Y%m%dT%H") / f"f{int(row.gfs_lead_hours):03}.json"
        if path not in cached:
            record = read_json(path)
            assert record["sha256"] == digest(record["payload"]), path
            cached[path] = record["payload"]
        payload = cached[path]
        assert payload["source_url"].startswith("https://noaa-gfs-bdp-pds.s3.amazonaws.com/")
        assert pd.Timestamp(payload["target_time"]) == row.target_time
        assert pd.Timestamp(payload["availability_upper_bound"]) == row.availability_upper_bound
        assert len(payload["fields"]) == 5
        assert max(pd.Timestamp(f["last_modified"]) for f in payload["fields"]) == row.availability_upper_bound
        for field in FIELDS:
            same(getattr(row, field), payload["points"][row.site_id][field], "CSV versus original GFS cache")
    print(f"Weather: {len(cached)} original cache objects and all 14592 CSV rows verified", flush=True)
    joined = weather.merge(labels, on=["site_id", "target_time"], how="left", validate="many_to_one")
    cards, fitted, states_by_version = {}, {}, {}
    for path in sorted(models.glob("*/ml-*/passport.json")):
        card = read_passport(path.parent)
        cutoff = pd.Timestamp(CUTOFFS[card["purpose"]])
        assert pd.Timestamp(card["training_cutoff"]) == cutoff
        train = joined.loc[joined.site_id.eq(card["site_id"]) & joined.complete.eq(True) & (joined.label_available_at <= cutoff)]
        train = train.sort_values(["site_id", "forecast_origin", "target_time"])
        manifest = {"baseline_smoke": "baseline", "january_evaluation": "january", "february_final": "final"}[card["purpose"]]
        check_manifest(root, f"{manifest}_train", card["site_id"], train)
        assert (train.target_time < cutoff).all() and (train.forecast_origin < cutoff).all()
        summary = card["training_summary"]
        assert len(train) == summary["rows"] and train.target_time.nunique() == summary["unique_target_hours"]
        assert train.label_available_at.max() == pd.Timestamp(summary["max_label_available_at"])
        config = card["config"]
        x = features(train, config["feature_set"])
        assert list(x.columns) == card["features"]
        states = baseline_states(train)
        saved_states = read_json(path.parent / "baselines.json")
        same(states["constant_median"]["value"], saved_states["constant_median"]["value"], "constant baseline uses train only")
        same(states["wind_table"]["bins"], saved_states["wind_table"]["bins"], "wind bins use train only")
        same(states["wind_table"]["values"], saved_states["wind_table"]["values"], "wind baseline uses train only")
        states_by_version[card["version"]] = states
        if config["kind"] == "catboost":
            assert config == selection["selected"][card["site_id"]]
            fresh = CatBoostRegressor(**config["params"])
            fresh.fit(x, train.actual)  # Fixed config, no eval_set or January tuning.
            saved = CatBoostRegressor().load_model(str(path.parent / "model.cbm"))
            assert saved.tree_count_ == config["params"]["iterations"]
            probe = features(joined.loc[joined.site_id.eq(card["site_id"])], config["feature_set"])
            same(saved.predict(probe), fresh.predict(probe), "independent refit versus saved model")
            fitted[card["version"]] = saved
        cards[(card["site_id"], card["purpose"])] = card
    print("All model training cohorts, baseline states and four fixed CatBoost refits verified", flush=True)
    for site in ("turbine_1", "turbine_2"):
        part = joined.loc[joined.site_id.eq(site)]
        train = part.loc[part.complete.eq(True) & (part.label_available_at <= pd.Timestamp(CUTOFFS["baseline_smoke"]))]
        train = train.sort_values(["site_id", "forecast_origin", "target_time"])
        validation = part.loc[(part.forecast_origin >= pd.Timestamp("2025-11-30T23:00:00+05:00")) &
                              (part.target_time >= pd.Timestamp("2025-12-01T00:00:00+05:00")) &
                              (part.target_time < pd.Timestamp("2026-01-01T00:00:00+05:00")) & part.complete.eq(True) &
                              (part.label_available_at <= pd.Timestamp(CUTOFFS["january_evaluation"]))]
        validation = validation.sort_values(["site_id", "forecast_origin", "target_time"])
        check_manifest(root, "december_train", site, train)
        check_manifest(root, "december_evaluation", site, validation)
        assert not set(train.target_time) & set(validation.target_time)
        states = baseline_states(train)
        results = []
        for experiment in [e for e in selection["experiments"] if e["site_id"] == site]:
            config = experiment["requested_config"]
            if config["kind"] == "catboost":
                estimator = CatBoostRegressor(**config["params"])
                estimator.fit(features(train, config["feature_set"]), train.actual,
                              eval_set=(features(validation, config["feature_set"]), validation.actual),
                              early_stopping_rounds=40, use_best_model=True)
                assert estimator.tree_count_ == experiment["config"]["params"]["iterations"]
                prediction = np.clip(estimator.predict(features(validation, config["feature_set"])), 0, 1)
            else:
                prediction = baseline(config["kind"], states, validation)
            metrics = score(validation.actual, prediction)
            same([metrics["mae"], metrics["rmse"]], [experiment["mae"], experiment["rmse"]], "December experiment")
            assert len(validation) == experiment["n_used"]
            results.append((metrics["mae"], experiment["candidate_index"], experiment["config"]))
        assert min(results, key=lambda r: (r[0], r[1]))[2] == selection["selected"][site]
    print("All 16 frozen December experiments and both original winners reproduced", flush=True)
    january = read_json(root / "ml/january_backtest.json")
    rows = pd.DataFrame(january["rows"])
    for name in ("issued_at", "target_time", "training_cutoff"):
        rows[name] = pd.to_datetime(rows[name], utc=True)
    jan_origins = pd.date_range("2025-12-31 23:00", "2026-01-30 23:00", freq="D", tz=ZONE).tz_convert("UTC")
    expected = {(s, o, o + pd.Timedelta(hours=h)) for s in ("turbine_1", "turbine_2") for o in jan_origins
                for h in range(1, 49) if (o + pd.Timedelta(hours=h)).tz_convert(ZONE).month == 1}
    assert len(rows) == len(expected) == 2928 and set(zip(rows.site_id, rows.issued_at, rows.target_time)) == expected
    for site in ("turbine_1", "turbine_2"):
        card = cards[(site, "january_evaluation")]
        part = rows.loc[rows.site_id.eq(site)].sort_values(["issued_at", "target_time"])
        samples = joined.loc[joined.site_id.eq(site) & joined.forecast_origin.isin(jan_origins) &
                             joined.target_time.dt.tz_convert(ZONE).dt.month.eq(1)].sort_values(["forecast_origin", "target_time"])
        check_manifest(root, "january_evaluation", site, samples)
        assert samples.target_time.min() > pd.Timestamp(card["training_cutoff"])
        assert part.model_version.eq(card["version"]).all() and (part.training_cutoff < part.issued_at).all()
        assert part.actual_valid.tolist() == samples.complete.tolist()
        same(part.actual, samples.actual, "January actuals independently aggregated")
        same(part.prediction, np.clip(fitted[card["version"]].predict(features(samples, card["config"]["feature_set"])), 0, 1), "January prediction from actual saved model")
        for name in ("constant_median", "wind_table"):
            same(part.baselines.map(lambda b: b[name]), baseline(name, states_by_version[card["version"]], samples), "January baseline prediction")
    for metric in read_json(root / "ml/january_metrics.json")["metrics"]:
        cohort = rows.loc[rows.site_id.eq(metric["site_id"])]
        if metric["horizon_band"] != "all":
            cohort = cohort.loc[cohort.horizon_step.le(24) if metric["horizon_band"] == "1–24" else cohort.horizon_step.gt(24)]
        valid = cohort.actual_valid & cohort.prediction_valid & cohort.actual.between(0, 1)
        assert int(valid.sum()) == metric["n_used"] and int((~valid).sum()) == metric["n_excluded"]
        cohort = cohort.loc[valid]
        prediction = cohort.prediction if metric["estimator"] == "model" else cohort.baselines.map(lambda b: b[metric["estimator"]])
        metrics = score(cohort.actual, prediction)
        same([metrics["mae"], metrics["rmse"]], [metric["mae"], metric["rmse"]], "independent January metrics")
    full = []
    replay = read_json(root / "reports/replay.json")
    assert not replay["errors"] and len(replay["forecast_ids"]) == 58
    for forecast_id in replay["forecast_ids"]:
        bundle = load_forecast(root / "forecasts" / forecast_id)
        part = pd.DataFrame(bundle["rows"])
        for name in ("issued_at", "target_time"):
            part[name] = pd.to_datetime(part[name], utc=True)
        card = cards[(part.site_id.iloc[0], "february_final")]
        assert not bundle["revises_forecast_id"] and len(part) == 48
        assert part.model_version.eq(card["version"]).all()
        samples = joined.loc[joined.site_id.eq(part.site_id.iloc[0]) & joined.forecast_origin.eq(part.issued_at.iloc[0])].sort_values("target_time")
        same(part.prediction, np.clip(fitted[card["version"]].predict(features(samples, card["config"]["feature_set"])), 0, 1), "February saved forecast versus model")
        assert part.weather_version.tolist() == samples.weather_version.tolist()
        full.append(part)
    full = pd.concat(full, ignore_index=True)
    exported = pd.read_csv(root / "ml/february_full_releases.csv", float_precision="round_trip")
    assert len(exported) == len(full) == 2784
    for field in ("issued_at", "target_time"):
        exported[field] = pd.to_datetime(exported[field], utc=True)
    export_keys = ["site_id", "issued_at", "target_time"]
    saved_sorted = full.sort_values(export_keys).reset_index(drop=True)
    csv_sorted = exported.sort_values(export_keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(csv_sorted[export_keys], saved_sorted[export_keys])
    same(csv_sorted.prediction, saved_sorted.prediction, "Full February CSV versus saved forecasts")
    for field in ("model_version", "weather_version", "horizon_step"):
        assert csv_sorted[field].equals(saved_sorted[field]), field
    chosen = pd.read_csv(root / "ml/february_hourly.csv", float_precision="round_trip")
    chosen["target_time"] = pd.to_datetime(chosen.target_time, utc=True)
    chosen["forecast_origin"] = pd.to_datetime(chosen.forecast_origin, utc=True)
    targets = pd.date_range("2026-02-01", "2026-03-01", freq="h", inclusive="left", tz=ZONE).tz_convert("UTC")
    assert len(chosen) == 1344 and set(zip(chosen.site_id, chosen.target_time)) == {(s, t) for s in ("turbine_1", "turbine_2") for t in targets}
    assert chosen.forecast_origin.eq(chosen.target_time.dt.tz_convert(ZONE).dt.normalize() - pd.Timedelta(hours=1)).all()
    merged = chosen.merge(full, left_on=["site_id", "forecast_origin", "target_time"], right_on=["site_id", "issued_at", "target_time"], suffixes=("_csv", "_saved"), validate="one_to_one")
    assert len(merged) == 1344 and merged.horizon_step_csv.between(1, 24).all()
    same(merged.prediction_csv, merged.prediction_saved, "February CSV versus saved original forecasts")
    assert merged.model_version_csv.eq(merged.model_version_saved).all() and merged.weather_version_csv.eq(merged.weather_version_saved).all()
    assert before == {str(p): digest(p.read_bytes()) for p in frozen}, "Audit modified a frozen artifact"
    print("January metrics and all February predictions verified; frozen files unchanged", flush=True)
    return {"status": "passed", "raw_hourly_targets": len(labels), "weather_rows": len(weather), "cache_objects": len(cached),
            "december_experiments_reproduced": 16, "fixed_catboost_refits_reproduced": 4,
            "january_pairs": len(rows), "february_full_rows": len(full), "february_hourly_rows": len(chosen),
            "frozen_files_unchanged": True, "ml_code_sha256": implementation_hash(), "selection_sha256": digest(selection),
            "max_numeric_tolerance": 1e-12, "training_methodology_changed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=data_root())
    parser.add_argument("--models-dir", type=Path, default=model_root())
    parser.add_argument("--report", type=Path, default=Path("artifacts/tz_ml_audit.json"))
    args = parser.parse_args()
    try:
        result = audit(args.data_dir, args.models_dir)
    except Exception as exc:
        atomic_json(args.report, {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    atomic_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
