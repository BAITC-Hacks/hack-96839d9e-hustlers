"""Build geometry v2 artifacts only after the fixed, cutoff-aware promotion gate.

Uses the existing training, feature, model and backend contracts. Original
models/data/ml remain intact; activation is via existing environment variables.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from windops.core import ROOT, SITES, atomic_json, data_root, digest, iso, read_json
from windops.ml.data import CUTOFFS, assert_disjoint, evaluation_rows, load_prepared, training_rows
from windops.ml.features import prepare_features
from windops.ml.models import load_model, predict_model, read_passport, train_artifact
from windops.ui.quality import load_backtest
from improve_model import MIN_JANUARY_GAIN, gate, locked_read, metrics, unchanged
from improve_weather_features import features as experimental_features

REPORTS = ROOT / "data/experiments/ml_weather_geometry_v1"
MODELS = ROOT / "models/geometry_v2"
RUNTIME = REPORTS / "runtime_data"


def copy_exact(source, destination):
    if destination.exists():
        assert digest(source.read_bytes()) == digest(destination.read_bytes()), destination
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main():
    root = data_root()
    assert root.resolve() == (ROOT / "data").resolve(), "Build from original data, before activating runtime_data"
    plan = read_json(REPORTS / "protocol.json")
    unchanged(plan)
    selection = locked_read(REPORTS / "selection.json")
    assert selection["protocol_sha256"] == plan["protocol_sha256"]
    frame, prepared = load_prepared(root)
    january_train, final_train = training_rows(frame, "january"), training_rows(frame, "final")
    january = evaluation_rows(frame, "january")
    assert_disjoint(january_train, january)
    source_backtest = read_json(root / "ml/january_backtest.json")
    old = pd.DataFrame(source_backtest["rows"])
    for name in ("issued_at", "target_time"):
        old[name] = pd.to_datetime(old[name], utc=True)
    accepted, checks = {}, {}
    # Recheck final model choice using ONLY labels available by the final cutoff.
    for site, selected in selection["selected"].items():
        assert selected is not None and selected["gate"]["passed"], site
        expected = pd.read_csv(REPORTS / "january_predictions" / f"{site}.csv")
        for name in ("forecast_origin", "target_time"):
            expected[name] = pd.to_datetime(expected[name], utc=True)
        v = january.loc[january.site_id.eq(site)].merge(expected, on=["site_id", "forecast_origin", "target_time"],
                            validate="one_to_one", suffixes=("", "_experiment"))
        assert len(v) == 1464
        np.testing.assert_allclose(v.actual, v.actual_experiment, atol=1e-12, rtol=0)
        allowed = v.loc[v.label_available_at <= CUTOFFS["final"]]
        previous, challenger = metrics(allowed, allowed.incumbent_prediction), metrics(allowed, allowed.candidate_prediction)
        decision = gate(challenger, previous, MIN_JANUARY_GAIN)
        checks[site] = {"n_used": len(allowed), "n_excluded_after_cutoff": len(v) - len(allowed),
                        "max_label_available_at": iso(allowed.label_available_at.max()),
                        "incumbent": previous, "candidate": challenger, "gate": decision}
        assert decision["passed"], f"Do not replace {site}: {decision}"
        accepted[site] = (selected, v)
    context = {"experiment": "ml_weather_geometry_v1", "january_independent": False,
        "selection": "Configuration and tree count frozen on December; promotion compared on previously viewed January labels available by final cutoff",
        "promotion_labels_cutoff": iso(CUTOFFS["final"]), "promotion_checks_sha256": digest(checks),
        "december_selection_sha256": digest(selection), "original_independent_backtest": "data/ml/january_backtest.json",
        "build_script_sha256": digest(Path(__file__).read_bytes())}
    finals, evaluation_models, output_rows = {}, {}, []
    for site, (selected, v) in accepted.items():
        base = selected["config"]["base_features"]
        config = {"kind": "catboost", "feature_set": base + "_geometry", "params": selected["config"]["params"]}
        # Verify training and runtime share exactly the experimental feature order/values.
        probe = frame.loc[frame.site_id.eq(site)]
        pd.testing.assert_frame_equal(prepare_features(probe, config["feature_set"]), experimental_features(probe, base))
        for purpose, cohort, cutoff, directory in (
            ("january_development", january_train, CUTOFFS["january"], MODELS / "evaluation"),
            ("february_final", final_train, CUTOFFS["final"], MODELS / "final")):
            train = cohort.loc[cohort.site_id.eq(site)]
            card = train_artifact(train, config, site_id=site, purpose=purpose, cutoff=cutoff,
                                  prepared=prepared, models=directory, selection_digest=digest(selection), experiment_context=context)
            if purpose == "january_development":
                prediction, _ = predict_model(load_model(directory / site / card["version"]), config, v)
                np.testing.assert_allclose(prediction, v.candidate_prediction, atol=1e-12, rtol=0)
                evaluation_models[site] = card["version"]
                actual_map = {(r.issued_at, r.target_time): r for r in old.loc[old.site_id.eq(site)].itertuples()}
                for position, row in enumerate(v.itertuples()):
                    previous = actual_map[(row.forecast_origin, row.target_time)]
                    output_rows.append({"run_id": "jan-development-" + digest({"site": site, "origin": iso(row.forecast_origin), "model": card["version"]})[:24],
                        "site_id": site, "issued_at": iso(row.forecast_origin), "target_time": iso(row.target_time),
                        "horizon_step": int(row.lead_hours), "prediction": float(prediction[position]), "actual": float(row.actual),
                        "actual_valid": True, "prediction_valid": True, "model_version": card["version"],
                        "training_cutoff": card["training_cutoff"], "baselines": {**previous.baselines, "previous_model": float(previous.prediction)}})
            else:
                finals[site] = {"version": card["version"], "config": card["config"], "training_summary": card["training_summary"]}
                destination = MODELS / "active" / site / card["version"]
                for source in (directory / site / card["version"]).iterdir():
                    if source.is_file():
                        copy_exact(source, destination / source.name)
                # Keep the original historical November/December artifacts available.
                for passport in (ROOT / "models" / site).glob("ml-*/passport.json"):
                    old_card = read_passport(passport.parent)
                    if old_card["purpose"] in ("baseline_smoke", "january_evaluation"):
                        for source in passport.parent.iterdir():
                            if source.is_file():
                                copy_exact(source, MODELS / "active" / site / old_card["version"] / source.name)
        print(site, "final:", finals[site]["version"], "January development:", evaluation_models[site], flush=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for name in ("weather", "scada"):
        link = RUNTIME / name
        if link.exists():
            assert link.resolve() == (root / name).resolve()
        else:
            link.symlink_to((root / name).resolve(), target_is_directory=True)
    # Preserve the original backtest separately and label the new one accurately.
    copy_exact(root / "ml/january_backtest.json", RUNTIME / "ml/original_independent_january_backtest.json")
    payload = {"schema_version": 1, "kind": "january_backtest", "data_mode": "real_saved", "timezone": "Asia/Almaty",
        "methodology": "Development comparison, NOT a new independent evaluation: January was previously observed. Six geometry candidates per turbine selected using October-November training and December-only early stopping; frozen counts refitted on October-December labels available by December 31 22:00 local. Promotion uses January labels available by January 31 22:00 only (1460 pairs per turbine); full-month diagnostics retain all 1464. Forecast pair unit and SCADA assumptions are unchanged. No February actuals.",
        "provenance": {"data_source": "data/scada/turbine_1.csv; data/scada/turbine_2.csv", "evaluation_independent": False,
                       "label_policy": prepared["label_policy"], "input_checksums": prepared["inputs"],
                       "models": evaluation_models, "experiment_context": context}, "rows": output_rows}
    assert len(load_backtest(payload).rows) == 2928
    atomic_json(RUNTIME / "ml/january_backtest.json", payload)
    atomic_json(RUNTIME / "ml/final_models.json", finals)
    atomic_json(REPORTS / "build.json", {"context": context, "promotion_checks": checks,
        "final_models": finals, "evaluation_models": evaluation_models,
        "model_directory": str(MODELS / "active"), "data_directory": str(RUNTIME),
        "status": "built_not_yet_activated", "feature_equivalence_verified": True, "original_artifacts_unchanged": True})
    unchanged(plan)
    print("Built separate runtime:", RUNTIME)


if __name__ == "__main__":
    main()
