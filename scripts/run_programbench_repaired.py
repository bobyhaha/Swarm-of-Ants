"""Fresh repaired conditions charged to the ORIGINAL $100 ProgramBench ledger."""

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


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def episode_plan(accounted):
    episodes = [{"id": "v2-pilot-xsv", "task": "xsv", "cap": 4, "solo": False,
                 "solo_model": None, "pilot": True}]
    for task in ("xsv", "hexyl"):
        for model, cap in ((None, 15), ("astra", 15), ("sol", 5), ("luna", 2)):
            episodes.append({"id": "v2-" + ("colony" if model is None else "solo-" + model) + "-" + task,
                             "task": task, "cap": cap, "solo": model is not None,
                             "solo_model": model, "pilot": False})
    allocated = sum(e["cap"] for e in episodes)
    if Decimal(str(accounted)) + allocated > Decimal(100):
        raise ValueError("The fresh run plan would exceed the existing $100 authorization")
    return episodes


def financial_failure(accounting, check, events):
    dispatched = [e for e in events if e["event"] == "campaign_reserved"]
    return (accounting["blocked"] or accounting["pending"] > 0
            or (bool(dispatched) and not check.get("accounting_valid"))
            or any(e["event"] == "api_limit_violation" for e in events)
            or any(e.get("reasoning_effort") != "high" for e in dispatched))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior-study", type=Path, required=True)
    parser.add_argument("--pilot-rounds", type=int, default=10)
    parser.add_argument("--prompt-key", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.pilot_rounds <= 100:
        parser.error("--pilot-rounds must be between 1 and 100")
    code, prior, study = Path(__file__).resolve().parents[1], args.prior_study.resolve(), args.out.resolve()
    ledger_path = prior / "campaign.sqlite"
    if not ledger_path.is_file():
        raise ValueError("Original campaign ledger must exist; never create a replacement budget")
    # One recovery controller at a time, even between HTTP requests.
    lock = (prior / "repair-controller.lock").open("a+b")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger = Ledger(ledger_path, parallel=True)
    before = ledger.summary()
    if before["blocked"] or before["pending"]:
        raise RuntimeError("Original ledger has unresolved requests; recovery cannot send paid calls")
    if json.loads((prior / "study.json").read_text())["status"] != "stopped_for_review":
        raise RuntimeError("Original study must be terminal before recovery")
    prior_audits = {}
    for path in prior.glob("*/report.json"):
        prior_audits[path.parent.name] = audit(path.parent / "events.jsonl")
        if not prior_audits[path.parent.name]["accounting_valid"]:
            raise RuntimeError("Prior accounting audit failed")
    episodes = episode_plan(before["accounted_usd"])
    study.mkdir(parents=True, exist_ok=False)
    runtime = study / "runtime"
    for name in ("ant_colony", "hayekmas"):
        shutil.copytree(code / name, runtime / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (runtime / "scripts").mkdir()
    for name in ("run_programbench_campaign.py", Path(__file__).name):
        shutil.copyfile(code / "scripts" / name, runtime / "scripts" / name)
    cfg = json.loads((code / "configs/programbench-mixed-high.json").read_text())
    cfg.update(funding_recovery=True, persistent_teams=True, bank_finalization=True)
    write(study / "main-config.json", cfg)
    write(study / "pilot-config.json", {**cfg, "max_rounds": args.pilot_rounds})
    shutil.copyfile(code / "configs/programbench-high-models.json", study / "models.json")
    hashes = {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in runtime.rglob("*") if p.is_file()}
    write(study / "source-hashes.json", hashes)
    plan = {"status": "prepared", "benchmark": "ProgramBench", "reasoning_effort": "high",
            "ledger": str(ledger_path), "prior_study": str(prior),
            "prior_accounted_usd": str(before["accounted_usd"]),
            "global_ceiling_usd": 100, "allocated_maximum_usd": sum(e["cap"] for e in episodes),
            "maximum_prior_plus_new_usd": str(Decimal(str(before["accounted_usd"])) + sum(e["cap"] for e in episodes)),
            "initial_society": {"astra": 2, "sol": 8, "luna": 10, "bank_included": True},
            "max_population_including_bank": 32, "max_direct_children": 2,
            "context_bytes_per_call": cfg["context_bytes"], "max_parallel_episodes": 2,
            "native_evaluations_at_once": 1,
            "pilot_rounds": args.pilot_rounds,
            "pilot_gate": "Completed native evaluation, valid accounting, all high; no hidden-score threshold",
            "selection": "One final artifact per fresh episode, selected before protected tests; no prior solutions or hidden feedback",
            "comparison": "Fresh colony and Astra ceilings $15 per task; Sol $5 and Luna $2. Same tools, 30 rounds, four work turns. Prior runs remain historical, interrupted comparisons.",
            "episodes": episodes}
    write(study / "study.json", plan)
    write(study / "prior-accounting-audits.json", prior_audits)
    if args.prompt_key and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = getpass.getpass("OpenRouter API key (not saved): ")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required")
    environment = dict(os.environ, PYTHONPATH=str(runtime),
                       ANT_COLONY_NATIVE_LOCK=str(prior / "native-evaluation.lock"))
    # Preserve the exact old stop reason. Billing failures can never take this path.
    stop = prior / "STOP"
    if stop.exists():
        reason = stop.read_text()
        if reason != "Episode colony-hexyl failed the engineering/accounting gate.\n":
            raise RuntimeError("Unexpected stop reason; do not automatically bypass it")
        stop.rename(prior / ("STOP.recovered-" + study.name))
        write(study / "recovery.json", {"previous_stop": reason, "accounting": before,
              "reason": "Audited Docker grading startup failure; no unresolved billing. Fresh runs share the original ledger.",
              "old_queued_episodes": "Superseded by fresh v2 conditions; old source/results unchanged"})
    evaluator = code.parent / "code-testing-v1/.venv/bin/python"
    task_root = code.parent / "code-testing-v1/datasets"

    def execute_episode(row):
        if ledger.summary()["blocked"] or stop.exists():
            return {"id": row["id"], "status": "not_started_financial_stop"}
        if any(hashlib.sha256((runtime / p).read_bytes()).hexdigest() != digest for p, digest in hashes.items()):
            raise RuntimeError("Frozen runtime changed during study")
        command = [sys.executable, str(runtime / "scripts/run_programbench_campaign.py"),
                   "--task", str(task_root / ("programbench-" + row["task"]) / "task.json"),
                   "--config", str(study / ("pilot-config.json" if row["pilot"] else "main-config.json")),
                   "--models", str(study / "models.json"), "--out", str(study / row["id"]),
                   "--ledger", str(ledger_path), "--evaluator-python", str(evaluator), "--max-usd", str(row["cap"])]
        if row["solo_model"]:
            command += ["--solo-model", row["solo_model"]]
        print("Starting " + row["id"] + "; ceiling $" + str(row["cap"]), flush=True)
        started = time.time()
        with (study / (row["id"] + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=runtime, env=environment, stdout=log, stderr=subprocess.STDOUT)
        directory = study / row["id"]
        events_path = directory / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()] if events_path.exists() else []
        check = audit(events_path) if (directory / "report.json").exists() else {}
        report = json.loads((directory / "report.json").read_text()) if (directory / "report.json").exists() else {}
        accounting = ledger.summary(row["id"])
        unsafe = financial_failure(accounting, check, events)
        healthy = (result.returncode == 0 and report.get("status") == "completed"
                   and check.get("complete_evaluation") and not unsafe)
        outcome = {"id": row["id"], "status": "completed" if healthy else "failed_episode",
                   "elapsed_seconds": time.time() - started, "returncode": result.returncode,
                   "report": report, "audit": check, "accounting": accounting}
        write(study / (row["id"] + "-outcome.json"), outcome)
        if unsafe:
            stop.write_text("Financial/inference-contract failure in " + row["id"] + ".\n")
        # A Docker/native evaluator failure ends this episode, not healthy peers.
        print(row["id"] + ": " + outcome["status"] + "; cumulative " + json.dumps(ledger.summary()), flush=True)
        return outcome

    def episode(row):
        try:
            return execute_episode(row)
        except BaseException:
            # Stop before the executor can dispatch a queued episode, not only
            # after its context manager has waited for every future.
            stop.write_text("Repair controller exception; no queued requests authorized.\n")
            raise

    outcomes = []
    try:
        plan["status"] = "pilot_running"
        write(study / "study.json", plan)
        outcomes.append(episode(episodes[0]))
        write(study / "outcomes.json", outcomes)
        if outcomes[0]["status"] == "completed":
            plan["status"] = "main_running"
            write(study / "study.json", plan)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(episode, row) for row in episodes[1:]]
                for future in as_completed(futures):
                    outcomes.append(future.result())
                    write(study / "outcomes.json", outcomes)
        plan["status"] = ("completed" if len(outcomes) == len(episodes)
                          and all(o["status"] == "completed" for o in outcomes)
                          else "finished_with_failures" if len(outcomes) == len(episodes) else "pilot_failed")
    except BaseException as exc:
        stop.write_text("Repair controller exception; no queued requests authorized.\n")
        plan["status"] = "controller_failed"
        plan["error_type"] = type(exc).__name__
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
