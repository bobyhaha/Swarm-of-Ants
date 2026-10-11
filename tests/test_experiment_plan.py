import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from decimal import Decimal

import pytest


def test_finite_experiment_allocations_cap_total_and_cannot_replay(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "scripts/run_board_experiments.py"
    spec = importlib.util.spec_from_file_location("board_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-credential-no-network")
    monkeypatch.setattr(module.sys, "argv", [str(path), "--out", str(tmp_path / "study")])
    assert module.main() == 0
    plan = json.loads((tmp_path / "study/plan.json").read_text())
    assert sum(Decimal(j["cap_usd"]) for j in plan["jobs"]) == Decimal("12") <= Decimal("50")
    assert len(calls) == 6
    assert all("--parallel" in c and c[c.index("--max-usd") + 1] == "2" for c in calls)
    a = json.loads((tmp_path / "study/board.json").read_text())
    b = json.loads((tmp_path / "study/no-board.json").read_text())
    assert a.pop("bank_board_size") == 6 and b.pop("bank_board_size") == 0
    assert a == b
    with pytest.raises(FileExistsError):
        module.main()
    assert len(calls) == 6
