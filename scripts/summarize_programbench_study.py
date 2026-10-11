"""Account for every ProgramBench episode, including failures and retained charges."""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from ant_colony.campaign_budget import Ledger
from ant_colony.storage import audit
from summarize_board_experiments import summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path)
    args=parser.parse_args()
    study=args.study.resolve()
    plan=json.loads((study / "study.json").read_text())
    extension_path=study / "baseline-extension.json"
    extension=json.loads(extension_path.read_text()) if extension_path.exists() else None
    episodes=plan["episodes"]+(extension["episodes"] if extension else [])
    rows=[]
    for episode in episodes:
        directory=study / episode["id"]
        if not (directory / "events.jsonl").exists():
            rows.append({**episode,"status":"not_started"})
            continue
        events=[json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        row={**episode,**summarize(directory)}
        final=next((e for e in reversed(events) if e["event"]=="programbench_final"),{})
        row.update(final_passed=final.get("passed"),final_total=final.get("total"),native_status=final.get("status"))
        roster={e["agent"]:e["model"] for e in events if e["event"]=="spawn"}
        row["initial_workers_by_model"]=dict(Counter(e["model"] for e in events if e["event"]=="spawn" and e.get("round")==0))
        row["active_working_agents_by_model"]=dict(Counter(roster[name] for name in {e["actor"] for e in events if e["event"]=="call" and e["kind"]=="work"}))
        cost_by_model=defaultdict(Decimal)
        for e in events:
            if e["event"]=="call":
                cost_by_model[e["model"]]+=Decimal(e["cost_usd"])
        row["cost_by_model_usd"]={m:str(v) for m,v in cost_by_model.items()}
        dispatched=[e for e in events if e["event"]=="campaign_reserved"]
        row["all_dispatched_calls_high"]=bool(dispatched) and all(e.get("reasoning_effort")=="high" for e in dispatched)
        row["public_probe_count"]=sum(e["event"]=="probe_discovered" for e in events)
        row["funding_requests"]=sum(e["event"]=="funding_needed" for e in events)
        row["recruitment_funding_requests"]=sum(e["event"]=="recruitment_funding_requested" for e in events)
        row["recruitment_funding_reviews"]=dict(Counter(e["status"] for e in events if e["event"]=="recruitment_funding_reviewed"))
        row["recruitment_executed_by_funding_status"]=dict(Counter(e["funding_status"] for e in events if e["event"]=="funded_recruitment_executed"))
        row["team_readiness_reports"]=sum(e["event"]=="team_ready" for e in events)
        row["incomplete_applications"]=sum(e["event"]=="application_incomplete" for e in events)
        row["actual_declines"]=sum(e["event"]=="application_declined" for e in events)
        row["standing_grants_used"]=sum(e["event"]=="standing_grant_used" for e in events)
        row["standing_grants_authorized"]=sum(e["event"]=="standing_grant_authorized" for e in events)
        row["bank_reviews_skipped"]=sum(e["event"]=="bank_review_skipped" for e in events)
        # Reconstruct local relationships as events occur, including before the
        # next round snapshot. Never infer useful depth from spawn count alone.
        parents, productive_nested = {}, set()
        for e in events:
            if e["event"] in {"spawn", "hire"}:
                name=e.get("agent", e.get("child"))
                if name:
                    parents[name]=e.get("parent")
            elif e["event"] in {"leave", "retire"}:
                parents[e["agent"]]=None
            elif e["event"]=="team_created":
                parents[e["manager"]]=None
            elif e["event"]=="snapshot":
                parents={a["id"]:a["parent"] for a in e["state"]["agents"]}
            elif e["event"]=="productive_step":
                parent=parents.get(e["agent"])
                if parent and parents.get(parent):
                    productive_nested.add(e["agent"])
        row["productive_workers_at_depth_2_or_more"]=len(productive_nested)
        by_kind=defaultdict(Decimal)
        for e in events:
            if e["event"]=="call":
                by_kind[e["kind"]]+=Decimal(e["cost_usd"])
        row["cost_by_kind_usd"]={k:str(v) for k,v in by_kind.items()}
        row["operator_interventions"]=0
        operator_file=directory / "operator-status.json"
        if operator_file.exists():
            operator=json.loads(operator_file.read_text())
            row.update(status=operator["status"], operator_interventions=operator["operator_interventions"],
                       grading_recovery=operator.get("grading_recovery"))
            recovered=operator.get("recovered_native_result")
            if recovered:
                row.update(final_passed=recovered.get("passed"), final_total=recovered.get("total"),
                           final_score=recovered.get("score"), native_status=recovered.get("status"))
        if (directory / "report.json").exists():
            row["audit"]=audit(directory / "events.jsonl")
        rows.append(row)
    with Ledger(Path(plan.get("ledger", study / "campaign.sqlite"))).connect() as db:
        receipts=[dict(row) for row in db.execute("SELECT run,actual,reserved,status FROM calls")]
        ceiling=Decimal(str(db.execute("SELECT value FROM campaign_settings WHERE key='ceiling_usd'").fetchone()[0]))
    spent=sum((Decimal(str(r["actual"])) for r in receipts if r["actual"] is not None),Decimal(0))
    reserved=sum((Decimal(str(r["reserved"])) for r in receipts if r["actual"] is None),Decimal(0))
    hashes=json.loads((study / "source-hashes.json").read_text())
    intact=all(hashlib.sha256((study / "runtime" / p).read_bytes()).hexdigest()==expected for p,expected in hashes.items())
    baseline_intact=None
    if extension:
        baseline_hashes=json.loads((study / "baseline-source-hashes.json").read_text())
        baseline_intact=all(hashlib.sha256((study / "baseline-runtime" / p).read_bytes()).hexdigest()==expected for p,expected in baseline_hashes.items())
    overall=plan["status"] if not extension or plan["status"]!="completed" else extension["status"]
    result={"plan":str(study / "study.json"),"status":overall,"primary_status":plan["status"],
            "baseline_extension_status":extension["status"] if extension else None,"records":rows,
            "allocated_maximum_usd":extension["allocated_maximum_usd"] if extension else plan["allocated_maximum_usd"],
            "confirmed_spend_usd":str(spent),"unconfirmed_reserved_usd":str(reserved),
            "maximum_accounted_usd":str(spent+reserved),"ceiling_usd":float(ceiling),
            "within_ceiling":spent+reserved<=ceiling,"frozen_runtime_unchanged":intact,"frozen_baseline_runtime_unchanged":baseline_intact,
            "limitations":plan.get("limitations", "Two tasks, one fresh run per condition, no controlled provider seed. Single-agent comparisons do not isolate model mix, bank, and recruitment effects. Dollar ceilings vary by condition; use the recorded episode caps, compare actual costs and report stops. Initial population is not active population. Context is bounded per call; total work still grows with active agents. No ant-colony emergence claim follows from a score alone.")}
    (study / "analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="records"},indent=2))
    print(json.dumps([{k:r.get(k) for k in ("id","status","round","final_passed","final_total","api_spent_usd","paid_calls","active_working_agents_by_model","hires","new_spawns","max_depth")} for r in rows],indent=2))


if __name__=="__main__":
    main()
