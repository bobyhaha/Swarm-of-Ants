import json
import subprocess
import sys

import pytest

from ant_colony.cli import main
from ant_colony.replay import write_replay
from ant_colony.storage import RunStore


def test_cli_writes_reviewable_run_and_refuses_overwrite(tmp_path, capsys):
    run = tmp_path / "demo"
    assert main(["demo", "--out", str(run)]) == 0
    for name in ("events.jsonl", "report.json", "audit.json", "metrics.json", "replay.html"):
        assert (run / name).is_file()
    assert json.loads((run / "report.json").read_text())["scripted_demo"]
    assert main(["audit", str(run)]) == 0
    assert main(["replay", str(run), "--round", "3"]) == 0
    assert main(["replay", str(run), "--html"]) == 0
    with pytest.raises(FileExistsError):
        main(["demo", "--out", str(run)])
    assert "budget manager" in (run / "replay.html").read_text()


def test_replay_does_not_embed_model_html_as_executable_markup(tmp_path):
    store = RunStore(tmp_path / "unsafe-text")
    store.emit("finished", report={"final_answer": "</script><script>BAD_CODE()</script>"})
    store.close()
    rendered = write_replay(store.directory).read_text()
    assert "</script><script>BAD_CODE()" not in rendered
    assert "\\u003c/script>\\u003cscript>BAD_CODE()" in rendered
    assert "textContent=report.final_answer" in rendered


def test_installed_command_works_outside_source_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "ant_colony", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "demo,run,commit0,audit,replay" in result.stdout
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from hayekmas import BaseAgent, BaseEnv, HayekMAS, Trainer; print('compatible')",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "compatible"
