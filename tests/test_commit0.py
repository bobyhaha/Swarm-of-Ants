import hashlib
import json
from types import SimpleNamespace

import pytest

from ant_colony.commit0 import Commit0Workbench, parse_junit


def fixture_task(tmp_path):
    root = tmp_path / "task"
    (root / "seed/pkg").mkdir(parents=True)
    (root / "seed/tests").mkdir()
    (root / "seed/pkg/a.py").write_text("def value():\n    pass\n")
    (root / "seed/tests/test_a.py").write_text("assert True\n")
    spec = b"Public specification only."
    (root / "SPEC.md").write_bytes(spec)
    (root / "dataset-row.json").write_text(json.dumps({"src_dir": "pkg/", "base_commit": "abc"}))
    task = {
        "benchmark": "commit0",
        "id": "fixture",
        "source_revision": "abc",
        "image": "unused",
        "spec_sha256": hashlib.sha256(spec).hexdigest(),
    }
    (root / "task.json").write_text(json.dumps(task))
    bench = Commit0Workbench(root / "task.json", tmp_path / "coding")
    agent = SimpleNamespace(name="a1", team="t1")
    bench.observation(agent)
    return bench, agent


def test_team_edits_are_private_and_protected_paths_cannot_be_written(tmp_path):
    bench, agent = fixture_task(tmp_path)
    initial = bench.execute(agent, {"kind": "read", "path": "pkg/a.py"})
    bench.execute(
        agent,
        {
            "kind": "write",
            "path": "pkg/a.py",
            "base_sha256": initial["sha256"],
            "content": "def value():\n    return 42\n",
        },
    )
    other = SimpleNamespace(name="a2", team="t2")
    bench.observation(other)
    assert "pass" in bench.execute(other, {"kind": "read", "path": "pkg/a.py"})["text"]
    for path in ("../outside.py", "/tmp/x.py", "tests/test_a.py", "pyproject.toml", "SPEC.md", ".git/config"):
        with pytest.raises(ValueError):
            bench.execute(agent, {"kind": "write", "path": path, "base_sha256": "new", "content": "x"})


def test_concurrent_edits_require_current_hash(tmp_path):
    bench, agent = fixture_task(tmp_path)
    initial = bench.execute(agent, {"kind": "read", "path": "pkg/a.py"})
    edit = {"kind": "write", "path": "pkg/a.py", "base_sha256": initial["sha256"], "content": "a = 1\n"}
    bench.execute(agent, edit)
    with pytest.raises(ValueError, match="Stale"):
        bench.execute(agent, edit)


def test_paging_does_not_skip_lines_when_byte_cap_truncates_page(tmp_path):
    bench, agent = fixture_task(tmp_path)
    bench.output_bytes = 32
    (bench.source(agent.team) / "pkg/a.py").write_text("abcdefghij\n" * 10)
    page = bench.execute(agent, {"kind": "read", "path": "pkg/a.py", "lines": 100})
    assert page["next_start"] == 3
    following = bench.execute(agent, {"kind": "read", "path": "pkg/a.py", "start": page["next_start"]})
    assert following["text"].startswith("3:")


def test_missing_file_is_an_action_error_not_a_host_exception(tmp_path):
    bench, agent = fixture_task(tmp_path)
    with pytest.raises(ValueError, match="unavailable"):
        bench.execute(agent, {"kind": "read", "path": "pkg/missing.py"})


def test_oversized_read_is_bounded_and_continues_without_skipping(tmp_path):
    bench, agent = fixture_task(tmp_path)
    (bench.source(agent.team) / "pkg/a.py").write_text("x\n" * 150)
    page = bench.execute(agent, {"kind": "read", "path": "pkg/a.py", "lines": 10000})
    assert len(page["text"].splitlines()) == 100
    assert page["next_start"] == 101
    following = bench.execute(
        agent, {"kind": "read", "path": "pkg/a.py", "start": page["next_start"], "lines": 10000}
    )
    assert following["text"].startswith("101:")
    assert len(following["text"].splitlines()) == 50
    assert following["next_start"] is None


@pytest.mark.parametrize("field,value", [("start", "1"), ("lines", "100"), ("lines", 0), ("lines", True)])
def test_invalid_read_pages_give_actionable_feedback(tmp_path, field, value):
    bench, agent = fixture_task(tmp_path)
    with pytest.raises(ValueError, match="positive JSON integers"):
        bench.execute(agent, {"kind": "read", "path": "pkg/a.py", field: value})


def test_symlink_read_and_write_rejected(tmp_path):
    bench, agent = fixture_task(tmp_path)
    outside = tmp_path / "secret.py"
    outside.write_text("secret")
    (bench.source(agent.team) / "pkg/link.py").symlink_to(outside)
    with pytest.raises(ValueError):
        bench.execute(agent, {"kind": "read", "path": "pkg/link.py"})


def test_junit_missing_collection_errors_and_fixture_failures_are_distinct():
    assert parse_junit("", 1)["score"] is None
    assert (
        parse_junit('<testsuite><testcase name="collect"><error/></testcase></testsuite>', 2)["score"] is None
    )
    xml = '<testsuite><testcase name="a"/><testcase name="b"><failure/></testcase><testcase name="c"><error/></testcase></testsuite>'
    result = parse_junit(xml, 1)
    assert result["total"] == 3 and result["passed"] == 1 and result["score"] == 1 / 3
    assert parse_junit('<testsuite><testcase name="a"/><testcase name="a"/></testsuite>', 0)["score"] is None


def test_final_selects_submitting_team_before_grading_and_only_once(tmp_path):
    bench, agent = fixture_task(tmp_path)
    calls = []
    bench.check = lambda source, label: calls.append((source, label)) or {"score": 0.5, "status": "scored"}
    action = SimpleNamespace(author=agent.name)
    assert bench.apply(action) == 0.5
    assert calls == [(bench.source(agent.team), "final")]
    assert (bench.directory / "submission/pkg/a.py").exists()
    with pytest.raises(RuntimeError):
        bench.apply(action)


def test_junit_feedback_deduplicates_fixture_errors_and_bounds_details():
    xml = (
        "<testsuite>"
        + "".join(
            f'<testcase name="case{i}"><error message="storage unavailable"/></testcase>' for i in range(8)
        )
        + '<testcase name="distinct"><failure message="expected 42"/></testcase></testsuite>'
    )
    result = parse_junit(xml, 1)
    assert result["errors"] == 8
    assert [f["message"] for f in result["failures"]] == ["storage unavailable", "expected 42"]
    xml = (
        "<testsuite>"
        + "".join(
            f'<testcase name="case{i}"><failure message="{i}{"x" * 1000}"/></testcase>' for i in range(8)
        )
        + "</testsuite>"
    )
    failures = parse_junit(xml, 1)["failures"]
    assert len(failures) == 4
    assert all(len(f["message"].encode()) <= 400 for f in failures)


def test_final_missing_score_does_not_become_zero(tmp_path):
    bench, agent = fixture_task(tmp_path)
    bench.check = lambda *args: {"score": None, "status": "build_failed"}
    with pytest.raises(RuntimeError, match="unscored"):
        bench.apply(SimpleNamespace(author=agent.name))
    assert bench.score is None


def test_tool_examples_use_json_integers_and_list_respects_directory(tmp_path):
    bench, agent = fixture_task(tmp_path)
    assert type(bench.schema["start"]) is int and type(bench.schema["lines"]) is int
    result = bench.execute(agent, {"kind": "list", "path": "pkg", "start": 1})
    assert result["files"] == ["pkg/a.py"]
    with pytest.raises(ValueError):
        bench.execute(agent, {"kind": "list", "start": "1"})


def test_changing_team_clears_previous_private_tool_result(tmp_path):
    bench, agent = fixture_task(tmp_path)
    bench.execute(agent, {"kind": "read", "path": "pkg/a.py"})
    agent.team = "t2"
    assert bench.observation(agent)["result"] is None


def test_subset_test_selector_rejects_nonoriginal_or_implementation_paths(tmp_path):
    bench, agent = fixture_task(tmp_path)
    for path in ("pkg/a.py", "tests/test_injected.py", "../x.py", "tests/test_a.py; echo bad"):
        with pytest.raises(ValueError):
            bench.check(bench.source(agent.team), "bad", test_path=path)


def test_grader_preserves_source_deletions_but_restores_original_tests(tmp_path, monkeypatch):
    bench, agent = fixture_task(tmp_path)
    source = bench.source(agent.team)
    (source / "pkg/a.py").unlink()
    (source / "tests/test_a.py").write_text("tampered")
    bench.manifest.update(
        protected_paths=["tests"], build_command="build", visible_test_command="pytest tests"
    )
    loaded = []

    class Box:
        name = "fixture"

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def load(self, directory):
            loaded.append(directory)
            assert not (directory / "pkg/a.py").exists()
            assert (directory / "tests/test_a.py").read_text() == "assert True\n"

        def put(self, *args, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            return {"exit_code": 0, "output": ""}

    def copy_report(argv, **kwargs):
        from pathlib import Path

        Path(argv[-1]).write_text('<testsuite><testcase name="a"/></testsuite>')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("ant_colony.commit0.DockerSandbox", Box)
    monkeypatch.setattr("ant_colony.commit0.run", copy_report)
    assert bench.check(source, "deleted")["passed"] == 1
    assert len(loaded) == 1
