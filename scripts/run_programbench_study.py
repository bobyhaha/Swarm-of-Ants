"""Finite high-reasoning study: $5 pilot, four $20 matched episodes, $100 global ceiling."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def write(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--prompt-key",action="store_true")
    args=parser.parse_args()
    code=Path(__file__).resolve().parents[1]
    study=args.out.resolve()
    study.mkdir(parents=True,exist_ok=False)
    runtime=study / "runtime"
    for name in ("ant_colony","hayekmas"):
        shutil.copytree(code / name,runtime / name,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    (runtime / "scripts").mkdir()
    shutil.copyfile(code / "scripts/run_programbench_campaign.py",runtime / "scripts/run_programbench_campaign.py")
    shutil.copyfile(__file__,runtime / "scripts/run_programbench_study.py")
    cfg=json.loads((code / "configs/programbench-mixed-high.json").read_text())
    models=json.loads((code / "configs/programbench-high-models.json").read_text())
    write(study / "models.json",models)
    write(study / "main-config.json",cfg)
    write(study / "pilot-config.json",{**cfg,"max_rounds":3})
    files={str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in runtime.rglob("*") if p.is_file()}
    write(study / "source-hashes.json",files)
    episodes=[{"id":"pilot-xsv","task":"xsv","cap":5,"solo":False,"pilot":True}]
    episodes += [{"id":condition+"-"+task,"task":task,"cap":20,"solo":condition=="solo","pilot":False}
                 for task in ("xsv","hexyl") for condition in ("colony","solo")]
    plan={"benchmark":"ProgramBench", "reasoning_effort":"high","global_ceiling_usd":100,
          "allocated_maximum_usd":85,"unallocated_reserve_usd":15,
          "initial_society":{"astra":2,"sol":8,"luna":10,"bank_included":True},
          "max_population_including_bank":32,"max_direct_children":2,
          "context_bytes_per_call":cfg["context_bytes"],"max_parallel_episodes":2,
          "condition":"Astra bank and dynamic worker roles versus one Astra working directly",
          "selection":"One final artifact per episode, prior to protected native evaluation; no score-based selection or automatic retries",
          "pilot_gate":"Healthy native evaluator, conserved accounting, all calls high, no unresolved receipts; no hidden-score threshold",
          "status":"prepared","episodes":episodes}
    write(study / "study.json",plan)
    if args.prompt_key and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"]=getpass.getpass("OpenRouter API key (not saved): ")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required; prepared study was not executed")
    environment=dict(os.environ,PYTHONPATH=str(runtime))
    ledger=Ledger(study / "campaign.sqlite",parallel=True)
    evaluator=code.parent / "code-testing-v1/.venv/bin/python"
    task_root=code.parent / "code-testing-v1/datasets"

    def episode(row):
        if ledger.summary()["blocked"] or (study / "STOP").exists():
            return {"id":row["id"],"status":"not_started_blocked"}
        command=[sys.executable,str(runtime / "scripts/run_programbench_campaign.py"),
                 "--task",str(task_root / ("programbench-"+row["task"]) / "task.json"),
                 "--config",str(study / ("pilot-config.json" if row["pilot"] else "main-config.json")),
                 "--models",str(study / "models.json"),"--out",str(study / row["id"]),
                 "--ledger",str(study / "campaign.sqlite"),"--evaluator-python",str(evaluator),
                 "--max-usd",str(row["cap"])]
        if row["solo"]:
            command.append("--solo")
        started=time.time()
        print("Starting "+row["id"]+"; ceiling $"+str(row["cap"]),flush=True)
        with (study / (row["id"]+".log")).open("w") as log:
            result=subprocess.run(command,cwd=runtime,env=environment,stdout=log,stderr=subprocess.STDOUT)
        directory=study / row["id"]
        audit_path=directory / "audit.json"
        report_path=directory / "report.json"
        audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
        report=json.loads(report_path.read_text()) if report_path.exists() else {}
        events=[json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()] if (directory / "events.jsonl").exists() else []
        payloads=[e for e in events if e["event"]=="campaign_reserved"]
        healthy=(result.returncode==0 and report.get("status")=="completed"
                 and audit.get("accounting_valid") and bool(payloads)
                 and all(e["reasoning_effort"]=="high" for e in payloads)
                 and not ledger.summary()["blocked"])
        outcome={"id":row["id"],"status":"completed" if healthy else "failed_gate",
                 "elapsed_seconds":time.time()-started,"returncode":result.returncode,
                 "report":report,"audit":audit,"accounting":ledger.summary(row["id"])}
        write(study / (row["id"]+"-outcome.json"),outcome)
        if not healthy:
            (study / "STOP").write_text("Episode " + row["id"] + " failed the engineering/accounting gate.\n")
        print(row["id"]+": "+outcome["status"]+"; study "+json.dumps(ledger.summary()),flush=True)
        return outcome

    outcomes=[]
    plan["status"]="pilot_running"
    write(study / "study.json",plan)
    outcomes.append(episode(episodes[0]))
    if outcomes[0]["status"]=="completed":
        plan["status"]="main_running"
        write(study / "study.json",plan)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(episode,row) for row in episodes[1:]]
            for future in as_completed(futures):
                outcomes.append(future.result())
                write(study / "outcomes.json",outcomes)
    else:
        (study / "STOP").write_text("Pilot failed the engineering gate; main episodes were not started.\n")
    plan["status"]="completed" if len(outcomes)==5 and all(r["status"]=="completed" for r in outcomes) else "stopped_for_review"
    write(study / "study.json",plan)
    write(study / "outcomes.json",outcomes)
    write(study / "budget.json",ledger.summary())
    ledger.close()
    return int(plan["status"]!="completed")


if __name__=="__main__":
    raise SystemExit(main())
