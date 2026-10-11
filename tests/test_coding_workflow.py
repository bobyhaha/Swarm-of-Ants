from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from ant_colony.config import ColonyConfig
from ant_colony.engine import ColonyMAS
from ant_colony.environment import Task
from ant_colony.providers import BudgetStop
from ant_colony.sandbox import copy_source
from test_commit0 import fixture_task
from test_mechanism import Policy


def write(bench, agent, path="pkg/a.py", text="value = 42\n"):
    current = bench.execute(agent, {"kind": "read", "path": path})
    return bench.execute(
        agent, {"kind": "write", "path": path, "base_sha256": current["sha256"], "content": text}
    )


def checks(bench):
    calls = []
    bench.check = lambda source, label, **kw: (
        calls.append((source, label)) or {"status": "scored", "score": 0.5, "passed": 1, "total": 2}
    )
    return calls


def test_disjoint_teams_accumulate_code_and_new_teams_inherit_it(tmp_path):
    bench, a = fixture_task(tmp_path)
    b = SimpleNamespace(name="b", team="t2")
    bench.observation(b)
    bench.execute(b, {"kind": "write", "path": "pkg/b.py", "base_sha256": "new", "content": "b = 3\n"})
    write(bench, a)
    calls = checks(bench)
    bench.finish_round(1, limit=2)
    assert (bench.public / "pkg/a.py").read_text() == "value = 42\n"
    assert (bench.source(a.team) / "pkg/b.py").read_text() == "b = 3\n"
    assert (bench.source(b.team) / "pkg/a.py").read_text() == "value = 42\n"
    assert (bench.source("new-team") / "pkg/b.py").read_text() == "b = 3\n"
    assert len(calls) == 1
    bench.finish_round(2)
    assert len(calls) == 1  # Unchanged checkpoints do not repeat tests.


def test_conflict_does_not_overwrite_local_or_public_work(tmp_path):
    bench, a = fixture_task(tmp_path)
    b = SimpleNamespace(name="b", team="t2")
    bench.observation(b)
    write(bench, a, text="value = 1\n")
    write(bench, b, text="value = 2\n")
    checks(bench)
    bench.finish_round(1)
    assert (bench.public / "pkg/a.py").read_text() == "value = 1\n"
    assert (bench.source(b.team) / "pkg/a.py").read_text() == "value = 2\n"
    assert bench.summary(b.team)["conflicts"] == ["pkg/a.py"]


def test_checkpoint_admission_is_bounded_and_waiting_changes_survive(tmp_path):
    bench, first = fixture_task(tmp_path)
    for i in range(5):
        agent = SimpleNamespace(name=f"a{i}", team=f"t{i + 1}")
        bench.observation(agent)
        bench.execute(
            agent, {"kind": "write", "path": f"pkg/new{i}.py", "base_sha256": "new", "content": f"x={i}\n"}
        )
    checks(bench)
    bench.finish_round(1, limit=2)
    assert bench.public_version == 2
    assert len(list((bench.public / "pkg").glob("new*.py"))) == 2
    bench.finish_round(2, limit=2)
    bench.finish_round(3, limit=2)
    assert len(list((bench.public / "pkg").glob("new*.py"))) == 5


def test_patch_requires_unique_text_and_fresh_hash(tmp_path):
    bench, agent = fixture_task(tmp_path)
    current = bench.execute(agent, {"kind": "read", "path": "pkg/a.py"})
    action = {
        "kind": "patch",
        "path": "pkg/a.py",
        "base_sha256": current["sha256"],
        "old": "    pass",
        "new": "    return 42",
    }
    bench.execute(agent, action)
    assert "return 42" in (bench.source(agent.team) / "pkg/a.py").read_text()
    with pytest.raises(ValueError, match="Stale"):
        bench.execute(agent, action)
    assert bench.summary(agent.team)["writes"] == 1


def test_observation_supplies_current_write_hash_without_extra_inspection(tmp_path):
    bench, agent = fixture_task(tmp_path)
    version = bench.observation(agent)["file_versions"]["pkg/a.py"]
    action = {"kind": "write", "path": "pkg/a.py", "base_sha256": version, "content": "value = 9\n"}
    bench.execute(agent, action)
    assert bench.observation(agent)["file_versions"]["pkg/a.py"] != version
    with pytest.raises(ValueError, match="Stale"):
        bench.execute(agent, dict(action, content="value = 10\n"))
    assert (bench.source(agent.team) / "pkg/a.py").read_text() == "value = 9\n"


def test_visible_write_versions_have_constant_count(tmp_path):
    bench, agent = fixture_task(tmp_path)
    source = bench.source(agent.team)
    for i in range(40):
        (source / f"pkg/new{i:02}.py").write_text("value = 1\n")
    versions = bench.observation(agent)["file_versions"]
    assert len(versions) == 16
    assert all(len(name.encode()) <= 240 and len(version) == 64 for name, version in versions.items())


def test_verified_test_status_becomes_stale_after_an_edit(tmp_path):
    bench, agent = fixture_task(tmp_path)
    checks(bench)
    bench.execute(agent, {"kind": "test"})
    assert bench.summary(agent.team)["tests_current"]
    write(bench, agent)
    assert not bench.summary(agent.team)["tests_current"]
    assert len(bench.recent[agent.name]) == 2


def test_mock_container_only_imports_implementation_and_notes(tmp_path, monkeypatch):
    bench, agent = fixture_task(tmp_path)
    bench.manifest["protected_paths"] = ["tests"]

    class Box:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def load(self, path):
            self.source = path

        def put(self, *a, **kw):
            pass

        def execute(self, command, **kw):
            return {"status": "ok", "exit_code": 0, "output": "done"}

        def snapshot(self, target):
            copy_source(self.source, target)
            (target / "pkg/a.py").write_text("value = 42\n")
            (target / "PROJECT_NOTES.md").write_text("Implemented a.py; next run test_a.py.")
            (target / "ignored.sh").write_text("arbitrary file")

    monkeypatch.setattr("ant_colony.coding_workspace.DockerSandbox", Box)
    result = bench.execute(agent, {"kind": "exec", "command": "placeholder"})
    assert result["changed"] == ["pkg/a.py"]
    obs = bench.observation(agent)
    assert "next run" in obs["project_notes"]
    assert not (bench.source(agent.team) / "ignored.sh").exists()


def test_deadline_grades_public_checkpoint_even_after_manager_closes_team(tmp_path):
    bench, _ = fixture_task(tmp_path)
    # Discard the fixture's unrelated workspace so only this colony contributes.
    bench.sources.clear()
    bench.bases.clear()
    bench.progress.clear()
    calls = checks(bench)

    def policy(obs):
        if obs["phase"] == "application":
            return {"proposal": {"plan": "Implement a.py and verify", "amount": 12000}}
        if obs["phase"] == "bank":
            return {
                "decisions": [
                    {
                        "agent": obs["proposals"][0]["agent"],
                        "decision": "approve",
                        "amount": 12000,
                        "feedback": "Implement and verify",
                    }
                ]
            }
        if obs["phase"] == "formation":
            return {"close": True}
        if obs["phase"] == "work":
            if obs["coding"]["result"] is None:
                return {"tool": {"kind": "read", "path": "pkg/a.py"}}
            return {
                "tool": {
                    "kind": "patch",
                    "path": "pkg/a.py",
                    "base_sha256": obs["coding"]["result"]["sha256"],
                    "old": "    pass",
                    "new": "    return 42",
                }
            }
        return {}

    cfg = replace(ColonyConfig(), initial_agents=1, max_rounds=2, work_turns=2)
    engine = ColonyMAS(
        cfg, Task("fixture", "Implement a.py", "tests", "commit0"), Policy(policy), workbench=bench
    )
    report = engine.run()
    assert report["status"] == "completed"
    assert report["submission_mode"] == "automatic_checkpoint"
    assert report["score"] == 0.5 and report["rewarded"] == cfg.reward // 2
    assert "return 42" in (bench.directory / "submission/pkg/a.py").read_text()
    assert (bench.directory / "checkpoint-export.json").exists()
    assert [label for _, label in calls] == ["checkpoint-0001", "final"]


def test_budget_stop_exports_checkpoint_without_paid_finalization(tmp_path):
    bench, _ = fixture_task(tmp_path)

    class Stop(Policy):
        def complete(self, *args):
            raise BudgetStop("test stop")

    engine = ColonyMAS(
        replace(ColonyConfig(), initial_agents=1),
        Task("f", "Implement", "tests", "commit0"),
        Stop(),
        workbench=bench,
    )
    report = engine.run()
    assert report["status"] == "budget_stopped"
    assert report["score"] is None
    assert (bench.directory / "checkpoint-export/pkg/a.py").exists()
    assert json.loads((bench.directory / "checkpoint-export.json").read_text())["reason"] == "budget_stopped"


def test_combined_checkpoint_cannot_exceed_artifact_cap(tmp_path):
    bench, a = fixture_task(tmp_path)
    bench.max_files = 2
    b = SimpleNamespace(name="b", team="t2")
    bench.observation(b)
    for agent, path in [(a, "pkg/b.py"), (b, "pkg/c.py")]:
        bench.execute(agent, {"kind": "write", "path": path, "base_sha256": "new", "content": "x=1\n"})
    checks(bench)
    bench.finish_round(1)
    assert len(bench.code_files(bench.public)) == 2
    assert bench.public_version == 1
    assert (bench.source(b.team) / "pkg/c.py").exists()
    assert len(bench.code_files(bench.source(b.team))) == 2
    assert "sync_blocked" in bench.summary(b.team)
    assert "pkg/b.py" not in bench.bases[b.team]
    (bench.source(b.team) / "pkg/c.py").unlink()
    bench.synchronize(b.team)
    assert "sync_blocked" not in bench.summary(b.team)
    assert (bench.source(b.team) / "pkg/b.py").exists()


def test_bounded_receipt_does_not_loop_on_oversized_metadata(tmp_path):
    bench, _ = fixture_task(tmp_path)
    result = bench.bounded_result({"output": "", "changed": ["x" * 10000]}, budget=1000)
    assert len(json.dumps(result).encode()) <= 1000
    assert result["truncated"]


def test_syntax_invalid_update_is_rejected_atomically_with_repair_feedback(tmp_path):
    bench, a = fixture_task(tmp_path)
    original = bench.hashes(bench.public)
    source = bench.source(a.team)
    (source / "pkg/a.py").write_text("def value():\n    return 42\n")
    (source / "pkg/b.py").write_text("def incomplete(:\n")
    calls = checks(bench)
    bench.finish_round(1)
    assert bench.hashes(bench.public) == original
    assert bench.public_version == 0
    assert not calls
    assert "pkg/b.py:1" in bench.summary(a.team)["integration_feedback"]
    assert "return 42" in (source / "pkg/a.py").read_text()
    (source / "pkg/b.py").write_text("x = 1\n")
    bench.finish_round(2)
    assert bench.public_version == 1
    assert "integration_feedback" not in bench.summary(a.team)
    assert len(calls) == 1


def test_encoded_read_clipping_preserves_continuation(tmp_path):
    bench, _ = fixture_task(tmp_path)
    result = bench.bounded_result(
        {
            "path": "pkg/a.py",
            "text": "\n".join(f"{i}: " + '\\"' * 100 for i in range(1, 31)),
            "total_lines": 30,
            "next_start": None,
        },
        budget=1000,
    )
    assert len(json.dumps(result).encode()) <= 1000
    assert result["next_start"] == len(result["text"].splitlines()) + 1


def test_receipt_packing_preserves_nearly_full_output_instead_of_halving(tmp_path):
    bench, _ = fixture_task(tmp_path)
    result = bench.bounded_result({"output": "x" * 6000, "status": "ok"})
    assert len(json.dumps(result).encode()) <= 6000
    assert len(result["output"]) > 5800
    assert result["output_truncated"] and result["shortened_for_context"]


def test_receipt_packing_preserves_complete_unicode_and_bounds_multiple_fields(tmp_path):
    bench, _ = fixture_task(tmp_path)
    result = bench.bounded_result({"output": "🤖" * 3000, "diagnostic": '\\"' * 3000})
    assert len(json.dumps(result, ensure_ascii=False).encode()) <= 6000
    assert result["output_truncated"]


@pytest.mark.parametrize("context_bytes,output_bytes", [(24000, 6000), (64000, 16000)])
def test_worker_context_remains_bounded_with_large_receipts_and_notes(tmp_path, context_bytes, output_bytes):
    bench, _ = fixture_task(tmp_path)
    bench.output_bytes = output_bytes
    engine = ColonyMAS(
        replace(
            ColonyConfig(), initial_agents=1, context_bytes=context_bytes, coding_output_bytes=output_bytes
        ),
        Task("f", "task " * 1100, "tests", "commit0"),
        Policy(),
        workbench=bench,
    )
    a = engine.agents["a0001"]
    engine.grant(a, 12000)
    bench.observation(a)
    bench.results[a.name] = {"output": '\\"' * output_bytes, "status": "ok"}
    (bench.source(a.team) / "PROJECT_NOTES.md").write_text('\\"' * 8000)
    bench.remember_tool(a, {"kind": "exec", "command": "inspect"}, bench.results[a.name])
    bench.remember_tool(a, {"kind": "exec", "command": "inspect again"}, bench.results[a.name])
    messages, dropped = engine.observation(a, "work")
    from ant_colony.context import byte_size

    assert byte_size(messages) <= engine.config.context_bytes
    progress = json.loads(messages[-1]["content"])["coding"]["progress"]
    assert progress["executions"] == 2
    assert progress["reads"] == 0  # A shell command is not necessarily an inspection.


def test_workers_receive_actual_remaining_work_calls(tmp_path):
    bench, _ = fixture_task(tmp_path)
    policy = Policy()
    engine = ColonyMAS(
        replace(ColonyConfig(), initial_agents=1, work_turns=3),
        Task("f", "Implement", "tests", "commit0"),
        policy,
        workbench=bench,
    )
    engine.round = 1
    engine.grant(engine.agents["a0001"], 12000)
    engine.work()
    assert [o["work_calls_left_this_round"] for o in policy.observations] == [3, 2, 1]
