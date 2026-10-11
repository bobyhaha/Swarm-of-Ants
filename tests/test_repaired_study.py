"""The rerun preserves prior spending and isolates nonfinancial failures."""

import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_programbench_repaired.py"
    spec = importlib.util.spec_from_file_location("repaired_study", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fresh_conditions_fit_remaining_original_authorization():
    episodes = module().episode_plan("19.896917925")
    assert sum(e["cap"] for e in episodes) == 78
    assert len({e["id"] for e in episodes}) == 9
    assert {(e["task"], e["solo_model"]) for e in episodes if not e["pilot"]} == {
        (task, model) for task in ("xsv", "hexyl") for model in (None, "astra", "sol", "luna")}
    with pytest.raises(ValueError, match="100"):
        module().episode_plan(23)


def test_missing_native_score_does_not_trigger_global_financial_stop():
    account = {"blocked": False, "pending": 0}
    check = {"accounting_valid": True, "complete_evaluation": False}
    events = [{"event": "campaign_reserved", "reasoning_effort": "high"},
              {"event": "runner_error", "reason": "native evaluation failed"}]
    assert not module().financial_failure(account, check, events)
    assert module().financial_failure({**account, "blocked": True}, check, events)
    assert module().financial_failure(account, {**check, "accounting_valid": False}, events)
    events[0]["reasoning_effort"] = "low"
    assert module().financial_failure(account, check, events)
