"""Grade an already selected artifact after an interrupted evaluator; no model calls."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--evaluator-python", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    episode = args.episode.resolve()
    coding = episode / "coding"
    manifest = json.loads(args.task.read_text())
    root = coding / "submission"
    archive = root / manifest["id"] / "submission.tar.gz"
    expected = json.loads((coding / "selected-artifact.json").read_text())["hashes"]
    with tarfile.open(archive) as tf:
        actual = {m.name: hashlib.sha256(tf.extractfile(m).read()).hexdigest()
                  for m in tf.getmembers() if m.isfile()}
    if actual != expected:
        raise ValueError("Saved submission differs from the original selected artifact")
    status_file = episode / "operator-status.json"
    status = {"status": "interrupted_during_evaluation", "operator_interventions": 1,
              "reason": "Scheduler handoff terminated the old process group during native grading; all six generation rounds and their costs are preserved.",
              "grading_recovery": "running", "new_model_calls": 0,
              "submission_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}

    def save():
        tmp = status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2) + "\n")
        tmp.replace(status_file)

    save()
    environment = {k: v for k, v in os.environ.items() if k not in {"OPENROUTER_API_KEY", "OPENAI_API_KEY"}}
    native = [str(args.evaluator_python.absolute()), "-c", "from programbench.cli.main import app; app()"]
    try:
        with args.lock.open("a+b") as lock, (root / "native-eval-recovery.log").open("wb") as log:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = subprocess.run([*native, "eval", str(root), "--workers", "1", "--branch-workers", "1",
                                     "--docker-cpus", "1"], env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=21600, start_new_session=True)
        if result.returncode:
            raise RuntimeError(f"Native evaluator exited {result.returncode}")
        scorer = episode.parent / "runtime/ant_colony/programbench_score.py"
        script = ("import importlib.util,json; s=importlib.util.spec_from_file_location('scorer',"
                  + repr(str(scorer)) + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                  "print(json.dumps(m.summarize(" + repr(str(root)) + "," + repr(manifest["id"]) + ")))")
        scored = subprocess.run([native[0], "-c", script], env=environment, capture_output=True,
                                text=True, timeout=120, check=True)
        final = json.loads(scored.stdout)
        status.update(grading_recovery="completed", recovered_native_result=final)
        (coding / "recovered-final-result.json").write_text(json.dumps(final, indent=2) + "\n")
    except BaseException as exc:
        status.update(grading_recovery="failed", error_type=type(exc).__name__, error=str(exc)[:500])
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
