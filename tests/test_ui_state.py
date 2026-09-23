from windops.ui.state import begin_run, fail_run, finish_run, initialize_state, selected_bundle, select_bundle


def test_run_guard_and_failed_request_do_not_masquerade_as_success(make_bundle):
    state = {}
    initialize_state(state)
    assert state["status"] == "idle"
    assert selected_bundle(state) is None
    params = {"site_ids": ["site-a"], "forecast_origin": "2026-01-31T23:00:00Z", "horizon_hours": 24}
    assert begin_run(state, params)
    assert not begin_run(state, params)
    assert state["status"] == "running"
    first = make_bundle(sites=("site-a",), horizon=24)
    finish_run(state, [first])
    assert state["status"] == "success"
    assert selected_bundle(state).forecast_id == first.forecast_id
    assert begin_run(state, dict(params, horizon_hours=48))
    fail_run(state, "Новый запуск не выполнен")
    assert state["status"] == "error"
    assert state["error"] == "Новый запуск не выполнен"
    assert selected_bundle(state) is None or state["previous_success"]
    assert first.forecast_id in state["history"]


def test_two_versions_remain_independently_selectable(make_bundle):
    state = {}
    initialize_state(state)
    first = make_bundle(forecast_id="old")
    second = make_bundle(forecast_id="new")
    finish_run(state, [first])
    finish_run(state, [second])
    assert set(state["history"]) == {"old", "new"}
    select_bundle(state, "old")
    assert selected_bundle(state).forecast_id == "old"
    select_bundle(state, "new")
    assert selected_bundle(state).forecast_id == "new"
    initialize_state(state)
    assert len(state["history"]) == 2


def test_missing_hours_mark_result_partial(make_bundle):
    bundle = make_bundle()
    bundle.rows = bundle.rows.iloc[1:]
    state = {}
    initialize_state(state)
    finish_run(state, [bundle])
    assert state["status"] == "partial"
