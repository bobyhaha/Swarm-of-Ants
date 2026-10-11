"""Replace a pilot at an explicit user request, preserving its work and original ledger.

This starts a fresh episode. It never changes a live engine, replays paid calls,
reuses a graded solution, or raises the campaign ceiling.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ant_colony.campaign_budget import Ledger
from ant_colony.storage import audit
from run_programbench_repaired import financial_failure, write


def amended_schedule(plan, accounted, registered, rounds):
    """Only replace an initial pilot; never rerun already dispatched main jobs."""
    if type(rounds) is not int or not 1 <= rounds <= 100:
        raise ValueError("Pilot rounds must be between 1 and 100")
    if plan.get("pilot_amendment"):
        raise ValueError("This study already has a pilot amendment")
    pilot = next(e for e in plan["episodes"] if e["pilot"])
    main = [e for e in plan["episodes"] if not e["pilot"]]
    if any(e["id"] in registered for e in main):
        raise ValueError("Main episodes have started; do not replace their admission gate")
    replacement = {**pilot, "id": f"v2-pilot{rounds}-{pilot['task']}", "max_rounds": rounds}
    if replacement["id"] in registered:
        raise ValueError("Replacement episode already registered; no paid replay")
    schedule = [replacement, *main]
    if Decimal(str(accounted)) + sum(Decimal(str(e["cap"])) for e in schedule) > 100:
        raise ValueError("Replacement and all queued ceilings would exceed the original $100 cap")
    return pilot, schedule


def generation_finished(events):
    # These events occur after final artifact selection; the runner makes no
    # more model calls while doing native evaluation or terminal cleanup.
    return any(e["event"] in {"native_evaluation_waiting", "native_evaluation_started", "finished"}
               for e in events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--prompt-key", action="store_true")
    args = parser.parse_args()
    study = args.study.resolve()
    plan = json.loads((study / "study.json").read_text())
    ledger_path = Path(plan["ledger"])
    if not ledger_path.is_file():
        raise ValueError("Original ledger is required")
    lock = (ledger_path.parent / "repair-controller.lock").open("a+b")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = ledger_path.parent / "STOP"
    if stop.exists():
        raise RuntimeError("Original campaign STOP remains in force")
    ledger = Ledger(ledger_path, parallel=True)
    old_pilot = next(e for e in plan["episodes"] if e["pilot"])
    events_path = study / old_pilot["id"] / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    if not generation_finished(events):
        raise RuntimeError("Prior pilot is still generating; leave it running before starting a replacement")
    before = ledger.summary()
    if before["blocked"] or before["pending"]:
        raise RuntimeError("Unresolved accounting prevents any replacement requests")
    if any(e.get("reasoning_effort") != "high" for e in events if e["event"] == "campaign_reserved"):
        raise RuntimeError("Prior pilot violated the high reasoning contract")
    if (events_path.parent / "report.json").exists() and not audit(events_path)["accounting_valid"]:
        raise RuntimeError("Prior pilot accounting audit failed")
    with ledger.connect() as db:
        registered = {row["id"] for row in db.execute("SELECT id FROM runs")}
    old_pilot, schedule = amended_schedule(plan, before["accounted_usd"], registered, args.rounds)
    runtime = study / "runtime"
    hashes = json.loads((study / "source-hashes.json").read_text())

    def verify_runtime():
        if any(hashlib.sha256((runtime / p).read_bytes()).hexdigest() != digest for p, digest in hashes.items()):
            raise RuntimeError("Original frozen runtime changed")

    verify_runtime()
    pilot_config = study / f"pilot-{args.rounds}-config.json"
    if pilot_config.exists() or (study / schedule[0]["id"]).exists():
        raise RuntimeError("Replacement artifacts already exist; no implicit retry")
    cfg = json.loads((study / "pilot-config.json").read_text())
    if args.rounds <= cfg["max_rounds"]:
        raise ValueError("Replacement must extend the original pilot round limit")
    if args.prompt_key and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = getpass.getpass("OpenRouter API key (not saved): ")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required")
    write(study / "study-before-pilot-amendment.json", plan)
    write(pilot_config, {**cfg, "max_rounds": args.rounds})
    snapshot = study / "pilot-amendment-controller.py"
    shutil.copyfile(Path(__file__).resolve(), snapshot)
    old_spend = Decimal(str(ledger.summary(old_pilot["id"])["accounted_usd"]))
    future_caps = sum(Decimal(str(e["cap"])) for e in schedule)
    plan["episodes"] = [{**old_pilot, "historical": True}, *schedule]
    plan["allocated_maximum_usd"] = float(old_spend + future_caps)
    plan["maximum_prior_plus_new_usd"] = str(Decimal(str(before["accounted_usd"])) + future_caps)
    plan["pilot_rounds"] = args.rounds
    plan["pilot_amendment"] = {
        "reason": f"User requested {args.rounds} pilot rounds after the {cfg['max_rounds']}-round engine was already running.",
        "requested_rounds": args.rounds, "replacement": schedule[0]["id"],
        "prior_pilot": old_pilot["id"], "prior_pilot_generation_spend_usd": str(old_spend),
        "accounted_before_amendment_usd": str(before["accounted_usd"]),
        "future_episode_ceilings_usd": str(future_caps),
        "fresh_seed": True, "previous_solutions_and_hidden_feedback_reused": False,
        "controller_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
    }
    code = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(runtime),
                       ANT_COLONY_NATIVE_LOCK=str(ledger_path.parent / "native-evaluation.lock"))
    evaluator = code.parent / "code-testing-v1/.venv/bin/python"
    task_root = code.parent / "code-testing-v1/datasets"

    def execute(row):
        if ledger.summary()["blocked"] or stop.exists():
            return {"id": row["id"], "status": "not_started_financial_stop"}
        verify_runtime()
        config = pilot_config if row["pilot"] else study / "main-config.json"
        command = [sys.executable, str(runtime / "scripts/run_programbench_campaign.py"),
                   "--task", str(task_root / ("programbench-" + row["task"]) / "task.json"),
                   "--config", str(config), "--models", str(study / "models.json"),
                   "--out", str(study / row["id"]), "--ledger", str(ledger_path),
                   "--evaluator-python", str(evaluator), "--max-usd", str(row["cap"])]
        if row["solo_model"]:
            command += ["--solo-model", row["solo_model"]]
        print(f"Starting {row['id']}; ceiling ${row['cap']}", flush=True)
        started = time.time()
        with (study / (row["id"] + ".log")).open("x") as log:
            result = subprocess.run(command, cwd=runtime, env=environment, stdout=log, stderr=subprocess.STDOUT)
        directory = study / row["id"]
        event_file, report_file = directory / "events.jsonl", directory / "report.json"
        event_rows = [json.loads(line) for line in event_file.read_text().splitlines()] if event_file.exists() else []
        check = audit(event_file) if report_file.exists() else {}
        report = json.loads(report_file.read_text()) if report_file.exists() else {}
        accounting = ledger.summary(row["id"])
        unsafe = financial_failure(accounting, check, event_rows)
        healthy = (result.returncode == 0 and report.get("status") == "completed"
                   and check.get("complete_evaluation") and not unsafe)
        outcome = {"id": row["id"], "status": "completed" if healthy else "failed_episode",
                   "elapsed_seconds": time.time() - started, "returncode": result.returncode,
                   "report": report, "audit": check, "accounting": accounting}
        write(study / (row["id"] + "-outcome.json"), outcome)
        if unsafe:
            stop.write_text("Financial/inference-contract failure in " + row["id"] + ".\n")
        print(row["id"] + ": " + outcome["status"], flush=True)
        return outcome

    def episode(row):
        try:
            return execute(row)
        except BaseException:
            stop.write_text("Amended controller exception; no queued requests authorized.\n")
            raise

    outcomes = []
    try:
        plan["status"] = "pilot_running"
        write(study / "study.json", plan)
        outcomes.append(episode(schedule[0]))
        write(study / "outcomes.json", outcomes)
        if outcomes[0]["status"] == "completed":
            plan["status"] = "main_running"
            write(study / "study.json", plan)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(episode, row) for row in schedule[1:]]
                for future in as_completed(futures):
                    outcomes.append(future.result())
                    write(study / "outcomes.json", outcomes)
        plan["status"] = ("completed" if len(outcomes) == len(schedule)
                          and all(o["status"] == "completed" for o in outcomes)
                          else "finished_with_failures" if len(outcomes) == len(schedule) else "pilot_failed")
    except BaseException as exc:
        stop.write_text("Amended controller exception; no queued requests authorized.\n")
        plan["status"], plan["error_type"] = "controller_failed", type(exc).__name__
        raise
    finally:
        write(study / "study.json", plan)
        write(study / "outcomes.json", outcomes)
        write(study / "budget.json", ledger.summary())
        ledger.close()
        lock.close()
    return int(plan["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
