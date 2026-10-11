"""Append single-Sol/Luna controls to a live study without changing its frozen episodes."""

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
from ant_colony.storage import audit


def extension_plan(primary):
    episodes=[{"id":f"solo-{model}-{task}","task":task,"cap":cap,
               "solo":True,"solo_model":model,"pilot":False}
              for task in ("xsv","hexyl") for model,cap in (("sol",5),("luna",2))]
    total=primary["allocated_maximum_usd"]+sum(row["cap"] for row in episodes)
    if total>100 or primary["global_ceiling_usd"]!=100:
        raise ValueError("Extended allocation would exceed the existing $100 ceiling")
    return {"status":"waiting_for_primary","reasoning_effort":"high","episodes":episodes,
            "allocated_maximum_usd":total,"unallocated_reserve_usd":100-total,
            "gate":"Primary study completes with valid audits and no unresolved API charges",
            "comparison":"Same tools, high reasoning, 64 KB contexts, 30 rounds and four work calls per round. "
                         "Astra/colony ceilings remain $20; supplemental Sol/Luna ceilings are $5/$2. "
                         "Report actual costs and budget stops; supplemental controls are not equal-dollar comparisons.",
            "population":"Exactly one actor per control; no hiring, spawning, application calls or paid bank",
            "max_parallel_episodes":2,"runtime":"baseline-runtime",
            "preserved_primary_runtime":"runtime","shared_ledger":"campaign.sqlite"}


def write(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path)
    parser.add_argument("--prompt-key",action="store_true")
    args=parser.parse_args()
    study=args.study.resolve()
    code=Path(__file__).resolve().parents[1]
    primary=json.loads((study / "study.json").read_text())
    plan=extension_plan(primary)
    manifest=study / "baseline-extension.json"
    if manifest.exists() or not (study / "campaign.sqlite").exists():
        raise ValueError("Need an existing ledger and a fresh baseline extension; no automatic replay")
    expected=json.loads((study / "source-hashes.json").read_text())
    if any(hashlib.sha256((study / "runtime" / p).read_bytes()).hexdigest()!=h for p,h in expected.items()):
        raise ValueError("Primary frozen runtime changed; inspect before extending")
    runtime=study / "baseline-runtime"
    shutil.copytree(study / "runtime",runtime,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for rel in ("ant_colony/baselines.py","scripts/run_programbench_campaign.py"):
        shutil.copyfile(code / rel,runtime / rel)
    shutil.copyfile(__file__,runtime / "scripts/add_programbench_baselines.py")
    hashes={str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in runtime.rglob("*") if p.is_file()}
    write(study / "baseline-source-hashes.json",hashes)
    write(manifest,plan)
    if args.prompt_key and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"]=getpass.getpass("OpenRouter API key (not saved): ")
    if not os.environ.get("OPENROUTER_API_KEY"):
        plan["status"]="credentials_missing"
        write(manifest,plan)
        raise RuntimeError("Key required in process environment; no requests were sent")
    ledger=Ledger(study / "campaign.sqlite",parallel=True)
    environment=dict(os.environ,PYTHONPATH=str(runtime))
    try:
        print("Supplemental baselines queued behind the primary study; combined allocation $99 / $100.",flush=True)
        while True:
            state=json.loads((study / "study.json").read_text())["status"]
            if (study / "STOP").exists() or ledger.summary()["blocked"] or state=="stopped_for_review":
                plan["status"]="blocked_before_start"
                write(manifest,plan)
                print("Baselines not started: primary study needs review.",flush=True)
                return 1
            if state=="completed":
                break
            time.sleep(15)
        # Recheck independently, without inspecting protected test details.
        for row in primary["episodes"]:
            directory=study / row["id"]
            report=json.loads((directory / "report.json").read_text())
            if report["status"]!="completed" or not report["usage_verified"] or not audit(directory / "events.jsonl")["accounting_valid"]:
                raise RuntimeError("Primary episode did not pass the accounting gate")

        def episode(row):
            if (study / "STOP").exists() or ledger.summary()["blocked"]:
                return {"id":row["id"],"status":"not_started_blocked"}
            command=[sys.executable,str(runtime / "scripts/run_programbench_campaign.py"),
                     "--task",str(code.parent / "code-testing-v1/datasets" / ("programbench-"+row["task"]) / "task.json"),
                     "--config",str(study / "main-config.json"),"--models",str(study / "models.json"),
                     "--out",str(study / row["id"]),"--ledger",str(study / "campaign.sqlite"),
                     "--evaluator-python",str(code.parent / "code-testing-v1/.venv/bin/python"),
                     "--max-usd",str(row["cap"]),"--solo-model",row["solo_model"]]
            print("Starting "+row["id"],flush=True)
            with (study / (row["id"]+".log")).open("x") as log:
                run=subprocess.run(command,cwd=runtime,env=environment,stdout=log,stderr=subprocess.STDOUT)
            directory=study / row["id"]
            report=json.loads((directory / "report.json").read_text()) if (directory / "report.json").exists() else {}
            check=audit(directory / "events.jsonl") if (directory / "audit.json").exists() else {}
            events=[json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()] if (directory / "events.jsonl").exists() else []
            calls=[e for e in events if e["event"]=="call"]
            dispatched=[e for e in events if e["event"]=="campaign_reserved"]
            single_high=(report.get("total_births")==1 and bool(calls) and bool(dispatched)
                         and all(e["kind"]=="work" and e["model"]==row["solo_model"] for e in calls)
                         and all(e["reasoning_effort"]=="high" for e in dispatched))
            healthy=run.returncode==0 and single_high and report.get("status")=="completed" and report.get("usage_verified") and check.get("accounting_valid") and not ledger.summary()["blocked"]
            outcome={"id":row["id"],"status":"completed" if healthy else "failed_gate",
                     "returncode":run.returncode,"report":report,"audit":check,"accounting":ledger.summary(row["id"])}
            write(study / (row["id"]+"-outcome.json"),outcome)
            if not healthy:
                (study / "STOP").write_text("Supplemental baseline failed its engineering/accounting gate.\n")
            return outcome

        def guarded_episode(row):
            try:
                return episode(row)
            except BaseException:
                (study / "STOP").write_text("Baseline episode controller failed; queued calls are stopped.\n")
                raise

        plan["status"]="running"
        write(manifest,plan)
        outcomes=[]
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(guarded_episode,row) for row in plan["episodes"]]
            for future in as_completed(futures):
                outcomes.append(future.result())
                write(study / "baseline-outcomes.json",outcomes)
        plan["status"]="completed" if all(r["status"]=="completed" for r in outcomes) else "stopped_for_review"
        write(manifest,plan)
        write(study / "budget.json",ledger.summary())
        print(plan["status"]+": "+json.dumps(ledger.summary()),flush=True)
        return int(plan["status"]!="completed")
    except Exception as exc:
        plan["status"]="stopped_for_review"
        plan["error_type"]=type(exc).__name__
        write(manifest,plan)
        (study / "STOP").write_text("Baseline controller failed; inspect its log before further spending.\n")
        raise
    finally:
        ledger.close()


if __name__=="__main__":
    raise SystemExit(main())
