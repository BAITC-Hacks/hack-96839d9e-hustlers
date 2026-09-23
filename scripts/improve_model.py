"""Bounded, isolated CatBoost experiment with an explicit no-regression gate.

December selects one candidate per site. January is a previously observed
development comparison, never early stopping or a new independent test.
This script does not activate or overwrite production models or forecasts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from windops.core import SITES, atomic_json, data_root, digest, iso, now, read_json
from windops.ml.data import CUTOFFS, assert_disjoint, evaluation_rows, load_prepared, training_rows, write_csv
from windops.ml.models import (fit_model, implementation_hash, library_versions, load_model,
                               model_root, predict_model, train_artifact)

DEFAULT_REPORTS = Path("data/experiments/ml_guarded_v1")
DEFAULT_MODELS = Path("models/experiments/ml_guarded_v1")
MIN_DECEMBER_GAIN = .01
MIN_JANUARY_GAIN = .02
EPS = 1e-12


def candidates():
    common = {"iterations": 800, "learning_rate": .03, "eval_metric": "MAE",
              "random_seed": 2026, "thread_count": 4, "task_type": "CPU",
              "allow_writing_files": False, "verbose": False, "l2_leaf_reg": 10}
    # Six settings × two existing feature contracts; fixed before any new fit.
    settings = [("MAE", 3), ("MAE", 5), ("MAE", 6), ("RMSE", 4), ("RMSE", 6), ("Huber:delta=0.1", 4)]
    return [{"kind": "catboost", "feature_set": features,
             "params": {**common, "loss_function": loss, "depth": depth}}
            for features in ("weather_hour", "weather_hour_lead") for loss, depth in settings]


def metrics(frame, prediction):
    prediction = np.asarray(prediction, dtype=float)
    assert len(prediction) == len(frame) and np.isfinite(prediction).all()
    result = {}
    for band, mask in (("all", np.ones(len(frame), dtype=bool)),
                       ("1-24", frame.lead_hours.to_numpy() <= 24),
                       ("25-48", frame.lead_hours.to_numpy() > 24)):
        error = prediction[mask] - frame.actual.to_numpy()[mask]
        assert len(error) and np.isfinite(error).all()
        result[band] = {"n_used": len(error), "mae": float(np.abs(error).mean()),
                        "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(error.mean())}
    return result


def gate(challenger, incumbent, min_gain):
    reasons = []
    for band in ("all", "1-24", "25-48"):
        if challenger[band]["n_used"] != incumbent[band]["n_used"]:
            reasons.append(f"{band}: comparison cohort size differs")
        for metric in ("mae", "rmse"):
            if not np.isfinite(challenger[band][metric]) or challenger[band][metric] > incumbent[band][metric] + EPS:
                reasons.append(f"{band}: {metric} degraded")
    if challenger["all"]["mae"] > incumbent["all"]["mae"] * (1 - min_gain) + EPS:
        reasons.append(f"overall MAE improvement below {min_gain:.0%}")
    return {"passed": not reasons, "reasons": reasons,
            "mae_improvement_pct": 100 * (incumbent["all"]["mae"] - challenger["all"]["mae"]) / incumbent["all"]["mae"]}


def protection_snapshot(root, incumbent_models):
    paths = [p for p in incumbent_models.glob("turbine_*/ml-*/*") if p.is_file()]
    paths += [root / "ml" / name for name in ("prepared.json", "selection.json", "selection_checksum.json",
             "january_backtest.json", "january_metrics.json", "final_models.json", "february_hourly.csv", "february_full_releases.csv")]
    paths += [root / "scada" / f"{site}.csv" for site in SITES]
    paths += [root / "weather/weather_for_ml.csv"]
    return {str(p.resolve()): digest(p.read_bytes()) for p in paths}


def unchanged(protocol):
    for path, sha in protocol["identity"]["protected_files"].items():
        if digest(Path(path).read_bytes()) != sha:
            raise ValueError(f"Protected production artifact changed: {path}")


def protocol(root, reports, incumbent_models):
    identity = {"candidates": candidates(), "december_min_mae_gain": MIN_DECEMBER_GAIN,
                "january_min_mae_gain": MIN_JANUARY_GAIN,
                "metric_guard": "MAE and RMSE must not increase overall or in either horizon band",
                "selection": "smallest December MAE among candidates passing December gate; ties prefer candidate order",
                "january_rule": "evaluate ONLY the frozen December winner once; reject means keep incumbent, no runner-up search",
                "january_previously_observed": True, "new_independent_evaluation": False,
                "early_stopping": "December only, existing fit_model patience=40; fixed iterations in all refits",
                "ml_code_sha256": implementation_hash(), "script_sha256": digest(Path(__file__).read_bytes()),
                "libraries": library_versions(), "protected_files": protection_snapshot(root, incumbent_models)}
    path = reports / "protocol.json"
    if path.exists():
        saved = read_json(path)
        if saved["identity"] != identity:
            raise ValueError("Protocol/data/code changed; use a separately named experiment. Never overwrite this protocol.")
        return saved
    reports.mkdir(parents=True, exist_ok=True)
    saved = {"schema_version": 1, "created_at": now(), "identity": identity, "protocol_sha256": digest(identity)}
    atomic_json(path, saved)
    return saved


def lock_write(path, value):
    if path.exists():
        raise ValueError(f"Refusing to overwrite completed stage: {path}")
    atomic_json(path, value)
    atomic_json(path.with_name(path.stem + "_checksum.json"), {"sha256": digest(value)})


def locked_read(path):
    value = read_json(path)
    if read_json(path.with_name(path.stem + "_checksum.json"))["sha256"] != digest(value):
        raise ValueError(f"Stage checksum mismatch: {path}")
    return value


def select(root, reports, models, plan):
    if (reports / "selection.json").exists():
        return locked_read(reports / "selection.json")
    frame, _ = load_prepared(root)
    train = training_rows(frame, "selection_train")
    validation = evaluation_rows(frame, "december")
    assert_disjoint(train, validation)
    original = read_json(root / "ml/selection.json")
    result = {"protocol_sha256": plan["protocol_sha256"], "january_used_in_this_selection": False,
              "january_previously_observed": True, "selected": {}, "incumbents": {}, "experiments": []}
    for site in SITES:
        t, v = train.loc[train.site_id.eq(site)], validation.loc[validation.site_id.eq(site)]
        baseline_config = original["selected"][site]
        baseline, _, fixed = fit_model(baseline_config, t)
        assert fixed == baseline_config
        baseline_prediction, _ = predict_model(baseline, fixed, v)
        baseline_metrics = metrics(v, baseline_prediction)
        result["incumbents"][site] = baseline_metrics
        expected = next(e for e in original["experiments"] if e["site_id"] == site and e["config"] == baseline_config)
        assert abs(baseline_metrics["all"]["mae"] - expected["mae"]) < EPS
        eligible = []
        for index, config in enumerate(plan["identity"]["candidates"]):
            estimator, _, fixed = fit_model(config, t, v)
            prediction, clipped = predict_model(estimator, fixed, v)
            score = metrics(v, prediction)
            decision = gate(score, baseline_metrics, MIN_DECEMBER_GAIN)
            folder = models / "december" / site
            folder.mkdir(parents=True, exist_ok=True)
            weight_path = folder / f"candidate-{index:02}.cbm"
            estimator.save_model(str(weight_path))
            row = {"site_id": site, "candidate_index": index, "requested_config": config,
                   "config": fixed, "metrics": score, "gate": decision, "clipped_predictions": clipped,
                   "model_path": str(weight_path), "model_sha256": digest(weight_path.read_bytes())}
            result["experiments"].append(row)
            if decision["passed"]:
                eligible.append(row)
            table = v[["site_id", "forecast_origin", "target_time", "lead_hours", "actual"]].copy()
            table["incumbent_prediction"], table["candidate_prediction"] = baseline_prediction, prediction
            write_csv(reports / "december_predictions" / f"{site}_{index:02}.csv", table)
            print(f"{site} candidate {index:02}: MAE={score['all']['mae']:.6f}, RMSE={score['all']['rmse']:.6f}, gate={decision['passed']}", flush=True)
        result["selected"][site] = min(eligible, key=lambda r: (r["metrics"]["all"]["mae"], r["candidate_index"])) if eligible else None
    unchanged(plan)
    lock_write(reports / "selection.json", result)
    return result


def compare(root, reports, models, plan):
    if (reports / "comparison.json").exists():
        return locked_read(reports / "comparison.json")
    selection = locked_read(reports / "selection.json")
    assert selection["protocol_sha256"] == plan["protocol_sha256"]
    frame, prepared = load_prepared(root)
    train, evaluation = training_rows(frame, "january"), evaluation_rows(frame, "january")
    assert_disjoint(train, evaluation)
    old = pd.DataFrame(read_json(root / "ml/january_backtest.json")["rows"])
    for field in ("issued_at", "target_time"):
        old[field] = pd.to_datetime(old[field], utc=True)
    result = {"kind": "posthoc_model_comparison", "independent": False,
              "selection_sha256": digest(selection), "sites": {}}
    for site, candidate in selection["selected"].items():
        if candidate is None:
            result["sites"][site] = {"replace": False, "reason": "No candidate passed the December gate"}
            continue
        v = evaluation.loc[evaluation.site_id.eq(site)].copy()
        assert v.complete.eq(True).all() and v.actual.between(0, 1).all()
        previous = old.loc[old.site_id.eq(site)]
        matched = v.merge(previous[["issued_at", "target_time", "actual", "prediction"]],
                          left_on=["forecast_origin", "target_time"], right_on=["issued_at", "target_time"],
                          validate="one_to_one", suffixes=("", "_old"))
        assert len(matched) == len(v) == 1464
        np.testing.assert_allclose(matched.actual, matched.actual_old, atol=EPS, rtol=0)
        baseline = metrics(matched, matched.prediction)
        card = train_artifact(train.loc[train.site_id.eq(site)], candidate["config"], site_id=site,
                              purpose="january_evaluation", cutoff=CUTOFFS["january"], prepared=prepared,
                              models=models / "january", selection_digest=digest(selection))
        estimator = load_model(models / "january" / site / card["version"])
        prediction, clipped = predict_model(estimator, card["config"], matched)
        score = metrics(matched, prediction)
        decision = gate(score, baseline, MIN_JANUARY_GAIN)
        table = matched[["site_id", "forecast_origin", "target_time", "lead_hours", "actual", "prediction"]].rename(columns={"prediction": "incumbent_prediction"})
        table["candidate_prediction"] = prediction
        table["candidate_model_version"] = card["version"]
        write_csv(reports / "january_predictions" / f"{site}.csv", table)
        result["sites"][site] = {"replace": decision["passed"], "gate": decision, "incumbent": baseline,
                                 "candidate": score, "candidate_model_version": card["version"], "clipped_predictions": clipped}
        print(f"{site}: January development comparison, MAE={score['all']['mae']:.6f}, RMSE={score['all']['rmse']:.6f}, replace={decision['passed']}", flush=True)
    unchanged(plan)
    lock_write(reports / "comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("select", "compare"))
    parser.add_argument("--data-dir", type=Path, default=data_root())
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--incumbent-models", type=Path, default=model_root())
    args = parser.parse_args()
    incumbent = args.incumbent_models.resolve()
    if args.models.resolve() == incumbent or args.reports.resolve() == (args.data_dir / "ml").resolve():
        raise ValueError("Experiment must use separate report and model directories")
    plan = protocol(args.data_dir, args.reports, incumbent)
    result = (select if args.stage == "select" else compare)(args.data_dir, args.reports, args.models, plan)
    print("Stage completed; production models unchanged. Report:", args.reports / ("selection.json" if args.stage == "select" else "comparison.json"))
    return result


if __name__ == "__main__":
    main()
