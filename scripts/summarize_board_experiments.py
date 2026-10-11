"""Summarize every recorded outcome; never select candidates by test scores."""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path
from statistics import mean, median


def summarize(directory):
    directory = Path(directory)
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    counts = Counter(e["event"] for e in events)
    calls = [e for e in events if e["event"] == "call"]
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    final = next((e for e in reversed(events) if e["event"] == "commit0_final"), {})
    snapshots = [e["state"] for e in events if e["event"] == "snapshot"]
    depth = children = population = teams = 0
    for state in snapshots:
        roster = {a["id"]: a for a in state["agents"]}
        population = max(population, len(roster))
        teams = max(teams, len(state["teams"]))
        for a in roster.values():
            children = max(children, len(a["children"]))
            d, parent = 0, a["parent"]
            while parent:
                d += 1
                parent = roster[parent]["parent"]
            depth = max(depth, d)
    strategies = defaultdict(list)
    current_strategy = {}
    strategy_changes = 0
    for e in events:
        if e["event"] == "spawn":
            current_strategy[e["agent"]] = e["strategy"]
        if e["event"] == "strategy_changed":
            if e["strategy"] != current_strategy.get(e["agent"]):
                strategy_changes += 1
                strategies[e["agent"]].append({"round": e["round"], "strategy": e["strategy"]})
                current_strategy[e["agent"]] = e["strategy"]
    costs = sum((Decimal(e["cost_usd"]) for e in calls), Decimal(0))
    phase_tokens = Counter()
    phase_costs = defaultdict(Decimal)
    for call in calls:
        phase_tokens[call["kind"]] += call["output_tokens"]
        phase_costs[call["kind"]] += Decimal(call["cost_usd"])
    tool_calls = Counter(e["kind"] for e in events if e["event"] == "coding_tool")
    # Explicit test actions only; exec may also invoke tests. Do not underlabel
    # this as every test run, or treat harmless rereads as measured redundancy.
    return {
        "directory": str(directory.resolve()),
        "status": report.get("status", "running"),
        "round": max(e.get("round", 0) for e in events),
        "final_passed": final.get("passed") if final.get("status") == "scored" else None,
        "final_total": final.get("total") if final.get("status") == "scored" else None,
        "final_errors": final.get("errors"),
        "final_score": report.get("score"),
        "api_spent_usd": str(costs),
        "api_reserved_usd": report.get("api_reserved_usd"),
        "output_tokens": sum(e["output_tokens"] for e in calls),
        "paid_calls": len(calls),
        "phase_calls": dict(Counter(e["kind"] for e in calls)),
        "phase_output_tokens": dict(phase_tokens),
        "phase_cost_usd": {phase: str(cost) for phase, cost in phase_costs.items()},
        "tool_calls": dict(tool_calls),
        "source_write_events": counts["source_written"],
        "files_written": dict(Counter(e["path"] for e in events if e["event"] == "source_written")),
        "invalid_responses": counts["invalid_reply"],
        "invalid_response_reasons": dict(
            Counter(e["reason"] for e in events if e["event"] == "invalid_reply")
        ),
        "rejected_action_reasons": dict(
            Counter(e["reason"] for e in events if e["event"] == "action_rejected")
        ),
        "board_rejections": counts["board_rejected"],
        "claim_guard_interventions": counts["proposal_overlap"],
        "checkpoint_conflicts": counts["checkpoint_conflict"],
        "peak_agents": population,
        "peak_teams": teams,
        "hires": counts["hire"],
        "new_spawns": sum(e["event"] == "spawn" and bool(e.get("parent")) for e in events),
        "max_depth": depth,
        "max_direct_children": children,
        "strategy_update_events": counts["strategy_changed"],
        "strategy_changes": strategy_changes,
        "agent_chosen_strategies": dict(strategies),
        "max_context_bytes": max(
            (e["input_bytes"] for e in events if e["event"] == "call_reserved"), default=0
        ),
        "context_omissions": counts["context_omitted"],
        "last_board": snapshots[-1].get("bank_board") if snapshots else None,
        "audit": json.loads((directory / "audit.json").read_text())
        if (directory / "audit.json").exists()
        else None,
        "checkpoints": [
            {k: e.get(k) for k in ("round", "version", "status", "passed", "total", "errors")}
            for e in events
            if e["event"] == "checkpoint_test"
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    plan = json.loads((args.study / "plan.json").read_text())
    records = []
    for job in plan["jobs"]:
        directory = args.study.parent / f"{args.study.name}-{job['condition']}-r{job['repetition']}"
        records.append(
            {**job, **summarize(directory)}
            if (directory / "events.jsonl").exists()
            else {**job, "status": "not_started"}
        )
    comparison = {}
    for condition in ("board", "no-board"):
        runs = [r for r in records if r["condition"] == condition]
        scored = [r for r in runs if r.get("final_passed") is not None]
        comparison[condition] = {
            "planned_runs": len(runs),
            "scored_runs": len(scored),
            "passed_counts": [r["final_passed"] for r in scored],
            "mean_passed": mean(r["final_passed"] for r in scored) if scored else None,
            "median_passed": median(r["final_passed"] for r in scored) if scored else None,
            "completed_cost_usd": str(
                sum((Decimal(r["api_spent_usd"]) for r in runs if r["status"] == "completed"), Decimal(0))
            ),
            "all_cost_usd": str(sum((Decimal(r.get("api_spent_usd", "0")) for r in runs), Decimal(0))),
            "retained_reservation_usd": str(
                sum((Decimal(r.get("api_reserved_usd") or "0") for r in runs), Decimal(0))
            ),
        }
    result = {
        "plan": str((args.study / "plan.json").resolve()),
        "records": records,
        "comparison": comparison,
        "limitations": "One library, three stochastic repetitions per condition, no seed control. No causal emergence claim; strategy changes are not proof of useful specialization.",
    }
    (args.study / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
