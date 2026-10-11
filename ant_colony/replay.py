"""A standalone, offline viewer generated exclusively from recorded events."""

from collections import Counter
import json
from pathlib import Path

from .storage import read_events


def write_replay(directory):
    directory = Path(directory)
    states, events, report = [], [], None
    phase_tokens = Counter()
    max_depth = max_children = peak_agents = 0
    last_round, last_phase = 0, "initialize"
    for event in read_events(directory / "events.jsonl"):
        # Early coding-adapter logs omitted round labels. Ordering is enough to
        # locate those local tool events; preserve the raw log unchanged.
        last_round = event.get("round", last_round)
        last_phase = event.get("phase", last_phase)
        event = {"round": last_round, "phase": last_phase, **event}
        if event["event"] == "snapshot":
            state = event["state"]
            states.append(state)
            agents = {a["id"]: a for a in state["agents"]}
            peak_agents = max(peak_agents, len(agents))
            for agent in agents.values():
                max_children = max(max_children, len(agent["children"]))
                depth, node = 0, agent
                while node["parent"]:
                    node = agents[node["parent"]]
                    depth += 1
                max_depth = max(max_depth, depth)
        if event["event"] == "call":
            phase_tokens[event["kind"]] += event["output_tokens"]
        if event["event"] == "finished":
            report = event["report"]
        if event["event"] in {
            "grant",
            "proposal",
            "proposal_review",
            "application_sponsored",
            "application_declined",
            "application_skipped",
            "job_opened",
            "bank_brief",
            "board_updated",
            "board_claimed",
            "board_released",
            "board_rejected",
            "proposal_overlap",
            "strategy_changed",
            "review_rejected",
            "source_written",
            "coding_tool",
            "commit0_final",
            "checkpoint_integrated",
            "checkpoint_conflict",
            "checkpoint_rejected",
            "checkpoint_sync_blocked",
            "checkpoint_test",
            "checkpoint_exported",
            "hire",
            "spawn",
            "sleep",
            "allocate",
            "published",
            "submission",
            "reward",
            "team_closed",
            "action_rejected",
            "stopped",
        }:
            events.append(event)
    data = {
        "states": states,
        "events": events,
        "report": report,
        "metrics": {
            "phase_tokens": dict(phase_tokens),
            "peak_agents": peak_agents,
            "max_depth": max_depth,
            "max_direct_children": max_children,
        },
    }
    (directory / "metrics.json").write_text(json.dumps(data["metrics"], indent=2) + "\n")
    # Literal '<' is escaped to prevent model/task text terminating the JSON script.
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    template = Path(__file__).with_name("replay.html").read_text()
    path = directory / "replay.html"
    path.write_text(template.replace("__COLONY_DATA__", encoded))
    return path
