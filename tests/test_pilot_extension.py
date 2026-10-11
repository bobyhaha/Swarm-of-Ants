"""A new pilot preserves prior charges and cannot replay dispatched jobs."""

import importlib.util
from pathlib import Path
import sys

import pytest


def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("pilot_extension", scripts / "extend_programbench_pilot.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_replacement_counts_existing_pilot_spend_and_preserves_main(monkeypatch):
    mod = module(monkeypatch)
    plan = {"episodes": [
        {"id": "v2-pilot-xsv", "task": "xsv", "cap": 4, "pilot": True, "solo_model": None},
        {"id": "v2-colony-xsv", "task": "xsv", "cap": 15, "pilot": False, "solo_model": None},
    ]}
    pilot, schedule = mod.amended_schedule(plan, "80.25", {"v2-pilot-xsv"}, 10)
    assert pilot["id"] == "v2-pilot-xsv"
    assert schedule[0]["id"] == "v2-pilot10-xsv"
    assert schedule[0]["max_rounds"] == 10
    assert schedule[0]["cap"] == 4
    assert schedule[1] == plan["episodes"][1]
    assert len(plan["episodes"]) == 2
    with pytest.raises(ValueError, match="100"):
        mod.amended_schedule(plan, "81.01", {pilot["id"]}, 10)
    with pytest.raises(ValueError, match="Main episodes"):
        mod.amended_schedule(plan, 1, {"v2-colony-xsv"}, 10)
    with pytest.raises(ValueError, match="no paid replay"):
        mod.amended_schedule(plan, 1, {"v2-pilot10-xsv"}, 10)
    with pytest.raises(ValueError, match="already has"):
        mod.amended_schedule({**plan, "pilot_amendment": {"requested_rounds": 10}}, 1, set(), 10)


def test_waits_for_generation_to_end_not_merely_final_round(monkeypatch):
    mod = module(monkeypatch)
    assert not mod.generation_finished([{"event": "call", "round": 6}])
    assert not mod.generation_finished([{"event": "submission"}])
    assert mod.generation_finished([{"event": "native_evaluation_waiting"}])
    assert mod.generation_finished([{"event": "finished"}])
