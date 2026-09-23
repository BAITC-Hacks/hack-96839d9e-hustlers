"""Second bounded experiment: row-local GFS geometry, same frozen safety gate.

Experimental weights are never selected by the running production plugin.
One December winner per turbine may receive a January development comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd

from windops.core import SITES, atomic_json, data_root, digest, now, read_json
from windops.ml.data import assert_disjoint, evaluation_rows, load_prepared, training_rows, write_csv
from windops.ml.features import bounded_predictions, prepare_features
from windops.ml.models import fit_model, implementation_hash, library_versions, model_root, predict_model
from improve_model import (EPS, MIN_DECEMBER_GAIN, MIN_JANUARY_GAIN, gate, locked_read,
                           lock_write, metrics, protection_snapshot, unchanged)

REPORTS = Path("data/experiments/ml_weather_geometry_v1")
MODELS = Path("models/experiments/ml_weather_geometry_v1")


def features(rows, base):
    x = prepare_features(rows, base).copy()
    for height in (10, 100):
        speed = x[f"wind_speed_{height}m_ms"].to_numpy()
        for component in ("u", "v"):
            values = x[f"wind_{component}_{height}m_ms"].to_numpy()
            x[f"wind_unit_{component}_{height}m"] = np.divide(values, speed, out=np.zeros_like(speed), where=speed > 1e-6)
    x["wind_speed_vertical_difference"] = x.wind_speed_100m_ms - x.wind_speed_10m_ms
    x["wind_speed_vertical_ratio"] = x.wind_speed_100m_ms / (x.wind_speed_10m_ms + .5)
    x["wind_vector_alignment"] = x.wind_unit_u_10m * x.wind_unit_u_100m + x.wind_unit_v_10m * x.wind_unit_v_100m
    assert np.isfinite(x.to_numpy()).all()
    return x.astype("float64")


def configs():
    return [{"base_features": base, "params": {"depth": depth, "iterations": 600,
             "learning_rate": .05, "loss_function": "MAE", "eval_metric": "MAE", "l2_leaf_reg": 3,
             "random_seed": 2026, "thread_count": 4, "task_type": "CPU", "allow_writing_files": False,
             "verbose": False}} for base in ("weather_hour", "weather_hour_lead") for depth in (3, 4, 6)]


def protocol(root):
    identity = {"candidates": configs(), "december_min_mae_gain": MIN_DECEMBER_GAIN,
                "january_min_mae_gain": MIN_JANUARY_GAIN,
                "selection": "lowest December MAE among candidates with >=1% gain and no MAE/RMSE regression in either horizon",
                "january_rule": "only frozen December winner; >=2% MAE gain and no MAE/RMSE regression in either horizon",
                "january_previously_observed": True, "new_independent_evaluation": False,
                "feature_semantics": "GFS unit wind vectors, 100m minus 10m speed, ratio with fixed 0.5 m/s denominator offset, vector alignment; each row independently",
                "libraries": library_versions(), "ml_code_sha256": implementation_hash(),
                "scripts": {p.name: digest(p.read_bytes()) for p in (Path(__file__), Path(__file__).with_name("improve_model.py"))},
                "protected_files": protection_snapshot(root, model_root())}
    path = REPORTS / "protocol.json"
    if path.exists():
        saved = read_json(path)
        assert saved["identity"] == identity, "Protocol changed; use separate experiment"
        return saved
    saved = {"created_at": now(), "identity": identity, "protocol_sha256": digest(identity)}
    atomic_json(path, saved)
    return saved


def select(root, plan):
    if (REPORTS / "selection.json").exists():
        return locked_read(REPORTS / "selection.json")
    frame, _ = load_prepared(root)
    train, validation = training_rows(frame, "selection_train"), evaluation_rows(frame, "december")
    assert_disjoint(train, validation)
    original = read_json(root / "ml/selection.json")
    result = {"protocol_sha256": plan["protocol_sha256"], "january_used_in_this_selection": False,
              "january_previously_observed": True, "selected": {}, "incumbents": {}, "experiments": []}
    for site in SITES:
        t, v = train.loc[train.site_id.eq(site)], validation.loc[validation.site_id.eq(site)]
        config = original["selected"][site]
        old_model, _, _ = fit_model(config, t)
        old_prediction, _ = predict_model(old_model, config, v)
        old_metrics = metrics(v, old_prediction)
        result["incumbents"][site] = old_metrics
        eligible = []
        for index, candidate in enumerate(plan["identity"]["candidates"]):
            x, xv = features(t, candidate["base_features"]), features(v, candidate["base_features"])
            estimator = CatBoostRegressor(**candidate["params"])
            estimator.fit(x, t.actual, eval_set=(xv, v.actual), early_stopping_rounds=40, use_best_model=True)
            fixed = {**candidate, "params": {**candidate["params"], "iterations": estimator.tree_count_}}
            prediction, clipped = bounded_predictions(estimator.predict(xv))
            score = metrics(v, prediction)
            decision = gate(score, old_metrics, MIN_DECEMBER_GAIN)
            path = MODELS / "december" / site / f"candidate-{index:02}.cbm"
            path.parent.mkdir(parents=True, exist_ok=True)
            estimator.save_model(str(path))
            row = {"site_id": site, "candidate_index": index, "requested_config": candidate, "config": fixed,
                   "features": list(x.columns), "metrics": score, "gate": decision, "clipped_predictions": clipped,
                   "model_path": str(path), "model_sha256": digest(path.read_bytes())}
            result["experiments"].append(row)
            if decision["passed"]:
                eligible.append(row)
            table = v[["site_id", "forecast_origin", "target_time", "lead_hours", "actual"]].copy()
            table["incumbent_prediction"], table["candidate_prediction"] = old_prediction, prediction
            write_csv(REPORTS / "december_predictions" / f"{site}_{index:02}.csv", table)
            print(f"{site} geometry {index:02}: MAE={score['all']['mae']:.6f}, RMSE={score['all']['rmse']:.6f}, gate={decision['passed']}", flush=True)
        result["selected"][site] = min(eligible, key=lambda r: (r["metrics"]["all"]["mae"], r["candidate_index"])) if eligible else None
    unchanged(plan)
    lock_write(REPORTS / "selection.json", result)
    return result


def compare(root, plan):
    if (REPORTS / "comparison.json").exists():
        return locked_read(REPORTS / "comparison.json")
    selected = locked_read(REPORTS / "selection.json")
    assert selected["protocol_sha256"] == plan["protocol_sha256"]
    frame, _ = load_prepared(root)
    train, validation = training_rows(frame, "january"), evaluation_rows(frame, "january")
    assert_disjoint(train, validation)
    previous = pd.DataFrame(read_json(root / "ml/january_backtest.json")["rows"])
    for name in ("issued_at", "target_time"):
        previous[name] = pd.to_datetime(previous[name], utc=True)
    result = {"kind": "posthoc_model_comparison", "independent": False,
              "selection_sha256": digest(selected), "sites": {}}
    for site, candidate in selected["selected"].items():
        if candidate is None:
            result["sites"][site] = {"replace": False, "reason": "No candidate passed the December gate"}
            continue
        config = candidate["config"]
        t, v = train.loc[train.site_id.eq(site)], validation.loc[validation.site_id.eq(site)]
        v = v.merge(previous.loc[previous.site_id.eq(site), ["issued_at", "target_time", "actual", "prediction"]],
                    left_on=["forecast_origin", "target_time"], right_on=["issued_at", "target_time"],
                    validate="one_to_one", suffixes=("", "_old"))
        assert len(v) == 1464 and v.complete.eq(True).all()
        np.testing.assert_allclose(v.actual, v.actual_old, atol=EPS, rtol=0)
        x, xv = features(t, config["base_features"]), features(v, config["base_features"])
        estimator = CatBoostRegressor(**config["params"])
        estimator.fit(x, t.actual)  # Frozen December iteration count; NO January eval_set.
        prediction, clipped = bounded_predictions(estimator.predict(xv))
        score, incumbent = metrics(v, prediction), metrics(v, v.prediction)
        decision = gate(score, incumbent, MIN_JANUARY_GAIN)
        path = MODELS / "january" / site / "model.cbm"
        path.parent.mkdir(parents=True, exist_ok=True)
        estimator.save_model(str(path))
        loaded = CatBoostRegressor().load_model(str(path))
        assert np.array_equal(estimator.predict(xv), loaded.predict(xv))
        table = v[["site_id", "forecast_origin", "target_time", "lead_hours", "actual", "prediction"]].rename(columns={"prediction": "incumbent_prediction"})
        table["candidate_prediction"] = prediction
        write_csv(REPORTS / "january_predictions" / f"{site}.csv", table)
        result["sites"][site] = {"replace": decision["passed"], "gate": decision, "incumbent": incumbent,
            "candidate": score, "clipped_predictions": clipped, "config": config,
            "training_cutoff": "2025-12-31T17:00:00Z", "train_rows": len(t), "features": list(x.columns),
            "model_path": str(path), "model_sha256": digest(path.read_bytes()), "save_load_identical": True}
        print(f"{site}: January development MAE={score['all']['mae']:.6f}, RMSE={score['all']['rmse']:.6f}, replace={decision['passed']}", flush=True)
    unchanged(plan)
    lock_write(REPORTS / "comparison.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "compare"))
    args = parser.parse_args()
    root = data_root()
    plan = protocol(root)
    (select if args.stage == "select" else compare)(root, plan)
    print("Production models unchanged; report directory:", REPORTS)
