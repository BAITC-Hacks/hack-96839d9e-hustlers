"""December-only selection, frozen January evaluation and February refit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from windops.core import BackendError, SITES, atomic_json, digest, iso, read_json
from .data import CUTOFFS, assert_disjoint, evaluation_rows, load_prepared, training_rows, write_csv
from .features import bounded_predictions, prepare_features
from .models import (BASELINES, baseline_prediction, fit_model, implementation_hash,
                     library_versions, load_model, model_root, predict_model, train_artifact)

# Declared before any validation result. Ties favor earlier/simpler candidates.
BASELINE_CONFIGS = [{"kind": name, "feature_set": "weather_hour"} for name in BASELINES]
CATBOOST_CONFIGS = [
    {"kind": "catboost", "feature_set": features,
     "params": {"depth": depth, "iterations": 500, "learning_rate": 0.05, "loss_function": "MAE",
                "eval_metric": "MAE", "random_seed": 2026, "thread_count": 4,
                "task_type": "CPU", "allow_writing_files": False, "verbose": False}}
    for features in ("weather_hour", "weather_hour_lead") for depth in (4, 6, 8)
]
CRITERION = "minimum December MAE after fixed [0,1] clip, same valid pairs; exact ties favor candidate order: median, wind table, CatBoost"


def scores(actual, predictions):
    error = np.asarray(predictions) - np.asarray(actual)
    if not len(error) or not np.isfinite(error).all():
        raise BackendError("ML_METRICS", "Метрики требуют непустых конечных пар.")
    return {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean()))}


def save_split(root, name, train, evaluation=None):
    if evaluation is not None:
        assert_disjoint(train, evaluation)
    columns = ["site_id", "forecast_origin", "target_time", "interval_end", "label_available_at", "complete", "quality_reason"]
    folder = root / "ml" / "splits"
    write_csv(folder / f"{name}_train.csv", train[columns])
    if evaluation is not None:
        write_csv(folder / f"{name}_evaluation.csv", evaluation[columns])
    summary = {}
    for kind, frame in (("train", train), ("evaluation", evaluation)):
        if frame is None:
            continue
        summary[kind] = {site: {"rows": len(part), "unique_target_hours": int(part.target_time.nunique()),
                               "target_min": iso(part.target_time.min()), "target_max": iso(part.target_time.max()),
                               "max_label_available_at": iso(part.label_available_at.max())}
                         for site, part in frame.groupby("site_id")}
    atomic_json(folder / f"{name}.json", summary)
    return summary


def baseline_stage(root, models=None):
    frame, prepared = load_prepared(root)
    train = training_rows(frame, "selection_train")
    save_split(root, "baseline", train)
    # Fixed wind-table baseline is historically usable for December origins.
    # A December-selected winner would NOT have been selectable in November.
    cards = [train_artifact(train.loc[train.site_id == site], BASELINE_CONFIGS[1], site_id=site,
                            purpose="baseline_smoke", cutoff=CUTOFFS["selection_train"], prepared=prepared, models=models)
             for site in SITES]
    atomic_json(root / "ml" / "baseline_models.json", {c["site_id"]: c["version"] for c in cards})
    return {c["site_id"]: c["version"] for c in cards}


def select(root):
    frame, prepared = load_prepared(root)
    identity = {"prepared": digest(prepared), "candidates": BASELINE_CONFIGS + CATBOOST_CONFIGS,
                "criterion": CRITERION, "code_sha256": implementation_hash(), "libraries": library_versions()}
    path = root / "ml" / "selection.json"
    if path.exists():
        previous = read_json(path)
        if previous["identity"] != identity:
            raise BackendError("ML_SELECTION_LOCKED", "Конфигурация выбора зафиксирована. Изменение данных/методики требует отдельного каталога и раскрытия влияния на независимость января.")
        return load_selection(root, prepared)
    train, validation = training_rows(frame, "selection_train"), evaluation_rows(frame, "december")
    all_december = evaluation_rows(frame, "december", available_labels_only=False)
    split = save_split(root, "december", train, validation)
    winners, experiments = {}, []
    for site in SITES:
        site_train, site_validation = train.loc[train.site_id == site], validation.loc[validation.site_id == site]
        all_site = all_december.loc[all_december.site_id == site]
        if site_train.empty or site_validation.empty:
            raise BackendError("ML_EVALUATION_EMPTY", f"Нет train/validation для {site}.")
        results = []
        for candidate_index, config in enumerate(BASELINE_CONFIGS + CATBOOST_CONFIGS):
            model, _, fixed = fit_model(config, site_train, site_validation)
            prediction, clipped = predict_model(model, fixed, site_validation)
            row = {"site_id": site, "candidate_index": candidate_index, "config": fixed,
                   "requested_config": config, "n_used": len(site_validation), "clipped_predictions": clipped,
                   "n_excluded": len(all_site) - len(site_validation),
                   **scores(site_validation.actual, prediction)}
            row["horizons"] = []
            for band, mask, total in (("1–24", site_validation.lead_hours <= 24, int((all_site.lead_hours <= 24).sum())),
                                      ("25–48", site_validation.lead_hours > 24, int((all_site.lead_hours > 24).sum()))):
                count = int(mask.sum())
                row["horizons"].append({"band": band, "n_used": count, "n_excluded": total - count,
                                        **(scores(site_validation.loc[mask, "actual"], prediction[mask.to_numpy()])
                                           if count else {"mae": None, "rmse": None})})
            results.append(row)
            print(f"{site}: {config['kind']} {config.get('params', {}).get('depth', '')} {config['feature_set']} MAE={row['mae']:.6f}", flush=True)
        winner = min(results, key=lambda row: (row["mae"], row["candidate_index"]))
        winners[site] = winner["config"]
        experiments.extend(results)
    selection = {"identity": identity, "criterion": CRITERION, "selected": winners,
                 "selection_labels_cutoff": iso(CUTOFFS["january"]), "split": split, "experiments": experiments,
                 "january_used_for_selection": False,
                 "early_stopping": "December eval_set only, raw MAE, patience 40; tree_count fixed for subsequent refits"}
    excluded = all_december.copy()
    excluded["evaluation_eligible"] = (excluded.complete.eq(True) & excluded.actual.between(0, 1) &
                                        (excluded.label_available_at <= CUTOFFS["january"]))
    excluded["exclusion_reason"] = np.where(excluded.evaluation_eligible, "used",
                                            np.where(excluded.complete.eq(True), "label_after_selection_cutoff", excluded.quality_reason))
    write_csv(root / "ml" / "splits" / "december_eligibility.csv", excluded[["site_id", "forecast_origin", "target_time", "label_available_at", "evaluation_eligible", "exclusion_reason"]])
    atomic_json(path, selection)
    atomic_json(root / "ml" / "selection_checksum.json", {"sha256": digest(selection)})
    return selection


def load_selection(root, prepared):
    path = root / "ml" / "selection.json"
    if not path.exists():
        raise BackendError("ML_SELECTION_MISSING", "Сначала выполните ML select на декабре.")
    selection = read_json(path)
    checksum = read_json(root / "ml" / "selection_checksum.json")
    if (checksum["sha256"] != digest(selection) or selection["identity"]["prepared"] != digest(prepared) or
            selection["identity"]["code_sha256"] != implementation_hash() or
            selection["identity"]["libraries"] != library_versions()):
        raise BackendError("ML_SELECTION_CHANGED", "Изменены данные, код, окружение или зафиксированная конфигурация выбора.")
    return selection


def comparison_metrics(rows):
    frame = pd.DataFrame(rows)
    results = []
    for site, part in frame.groupby("site_id"):
        for band, cohort in (("all", part), ("1–24", part.loc[part.horizon_step <= 24]), ("25–48", part.loc[part.horizon_step > 24])):
            valid = cohort.actual_valid & cohort.prediction_valid & cohort.actual.between(0, 1) & cohort.prediction.between(0, 1)
            for baseline in BASELINES:
                valid &= cohort.baselines.map(lambda values: np.isfinite(values[baseline]) and 0 <= values[baseline] <= 1)
            selected = cohort.loc[valid]
            common = {"site_id": site, "horizon_band": band, "n_used": len(selected), "n_excluded": int((~valid).sum()),
                      "exclusion_reasons": {str(k): int(v) for k, v in cohort.loc[~valid, "actual_reason"].value_counts().items()}}
            model_scores = scores(selected.actual, selected.prediction) if len(selected) else {"mae": None, "rmse": None}
            for name in ("model", *BASELINES):
                prediction = selected.prediction if name == "model" else selected.baselines.map(lambda values: values[name])
                metric = scores(selected.actual, prediction) if len(selected) else {"mae": None, "rmse": None}
                improvement = ((metric["mae"] - model_scores["mae"]) / metric["mae"] * 100
                               if name != "model" and metric["mae"] and model_scores["mae"] is not None else None)
                results.append({**common, "estimator": name, **metric, "model_mae_improvement_pct": improvement})
    return results


def january_backtest(root, models=None):
    models = models or model_root()
    frame, prepared = load_prepared(root)
    selection = load_selection(root, prepared)
    train, evaluation = training_rows(frame, "january"), evaluation_rows(frame, "january")
    split = save_split(root, "january", train, evaluation)
    rows, cards, clipping = [], {}, {}
    for site in SITES:
        part = evaluation.loc[evaluation.site_id == site]
        if part.empty:
            raise BackendError("ML_EVALUATION_EMPTY", f"Нет январских выпусков для {site}.")
        card = train_artifact(train.loc[train.site_id == site], selection["selected"][site], site_id=site,
                              purpose="january_evaluation", cutoff=CUTOFFS["january"], prepared=prepared,
                              models=models, selection_digest=digest(selection))
        cards[site] = card["version"]
        folder = models / site / card["version"]
        predictions, clipping[site] = predict_model(load_model(folder), card["config"], part)
        states = read_json(folder / "baselines.json")
        features = prepare_features(part, card["config"]["feature_set"])
        baselines = {name: bounded_predictions(baseline_prediction(name, states[name], features))[0] for name in BASELINES}
        for index, (_, item) in enumerate(part.iterrows()):
            valid = bool(item.complete == True and pd.notna(item.actual) and 0 <= item.actual <= 1)
            rows.append({"run_id": "january-" + digest({"site": site, "origin": iso(item.forecast_origin), "model": card["version"]})[:24],
                         "site_id": site, "issued_at": iso(item.forecast_origin), "target_time": iso(item.target_time),
                         "horizon_step": int(item.lead_hours), "prediction": float(predictions[index]),
                         "actual": float(item.actual) if valid else None, "actual_valid": valid, "prediction_valid": True,
                         "actual_reason": "complete" if valid else str(item.quality_reason),
                         "model_version": card["version"], "training_cutoff": card["training_cutoff"],
                         "baselines": {name: float(values[index]) for name, values in baselines.items()}})
    payload = {"schema_version": 1, "kind": "january_backtest", "data_mode": "real_saved", "timezone": "Asia/Almaty",
               "methodology": "Separate models per turbine. Configuration selected using October-November training and December-only eval_set (labels available by December 31 22:00 local). Fixed iterations refitted on October-December labels available by December 31 22:00. Daily origins December 31–January 30 at 23:00 local; January target hours only; overlapping releases are distinct equally weighted pairs. Six unique valid SCADA interval labels per hour; unconfirmed interval/delay assumptions in provenance. No January tuning, no February actuals. Same finite [0,1] clip everywhere; missing labels remain null. Baselines refitted on the same training pairs.",
               "provenance": {"data_source": "data/scada/turbine_1.csv; data/scada/turbine_2.csv",
                              "input_checksums": prepared["inputs"], "label_policy": prepared["label_policy"],
                              "selection_sha256": digest(selection), "models": cards}, "rows": rows}
    from windops.ui.quality import load_backtest, evaluate_backtest
    validated = load_backtest(payload)
    metrics = comparison_metrics(rows)
    atomic_json(root / "ml" / "january_backtest.json", payload)
    atomic_json(root / "ml" / "january_metrics.json", {"evaluation_models": cards, "metrics": metrics,
                "clipped_predictions": clipping, "split": split, "independence": "No January configuration changes"})
    write_csv(root / "ml" / "january_ui_metrics.csv", evaluate_backtest(validated))
    return {"models": cards, "metrics": metrics, "path": str(root / "ml" / "january_backtest.json")}


def train_final(root, models=None):
    frame, prepared = load_prepared(root)
    selection = load_selection(root, prepared)
    train = training_rows(frame, "final")
    save_split(root, "final", train)
    cards = [train_artifact(train.loc[train.site_id == site], selection["selected"][site], site_id=site,
                            purpose="february_final", cutoff=CUTOFFS["final"], prepared=prepared,
                            models=models, selection_digest=digest(selection)) for site in SITES]
    result = {c["site_id"]: {"version": c["version"], "config": c["config"], "training_summary": c["training_summary"]} for c in cards}
    atomic_json(root / "ml" / "final_models.json", result)
    return result
