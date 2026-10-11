"""Read-only reproducibility, context, topology and spending checks for a study."""

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from ant_colony.context import byte_size
from ant_colony.storage import audit


def validate(study):
    study = Path(study)
    plan = json.loads((study / "plan.json").read_text())
    condition_configs = [
        json.loads((study / (condition + ".json")).read_text())
        for condition in ("board", "no-board")
    ]
    assert [c.pop("bank_board_size") for c in condition_configs] == [6, 0]
    assert condition_configs[0] == condition_configs[1]
    allocation = sum(Decimal(job["cap_usd"]) for job in plan["jobs"])
    assert allocation <= Decimal(plan["authorized_cap_usd"]) <= Decimal("50")
    total = Decimal(0)
    records = []
    reference_provenance = None
    for job in plan["jobs"]:
        directory = study.parent / f"{study.name}-{job['condition']}-r{job['repetition']}"
        report = json.loads((directory / "report.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        models = json.loads((directory / "models.json").read_text())
        assert {v["id"] for v in models.values()} == {"openai/gpt-6-luna"}
        expected = json.loads((study / (job["condition"] + ".json")).read_text())
        for key, value in expected.items():
            assert config[key] == value, (directory.name, key)
        for name, expected_hash in plan["implementation_sha256"].items():
            relative = name.removeprefix("scripts/")
            if name.endswith("run_board_experiments.py"):
                continue  # The launcher's own hash is retained in the predeclared plan.
            source = directory / "implementation-sources" / relative
            assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_hash, source
        provenance = json.loads((directory / "coding/provenance.json").read_text())["seed_hashes"]
        if reference_provenance is None:
            reference_provenance = provenance
        assert provenance == reference_provenance
        ledger_audit = audit(directory / "events.jsonl")
        assert ledger_audit["accounting_valid"] and ledger_audit["unresolved_api_calls"] == 0
        assert ledger_audit["complete_evaluation"] and report["usage_verified"]
        assert report["status"] == "completed"
        assert report["submission_mode"] == "automatic_checkpoint"
        assert report["api_reserved_usd"] == "0.000000"
        charges = Decimal(0)
        source = (directory / "events.jsonl").read_text().splitlines()
        bank_calls = 0
        final_grades = []
        for line in source:
            event = json.loads(line)
            if event["event"] == "commit0_final":
                final_grades.append(event)
            if event["event"] == "call":
                charges += Decimal(event["cost_usd"])
                if event["kind"] == "bank":
                    bank_calls += 1
            if event["event"] == "call_reserved":
                assert byte_size(event["messages"]) == event["input_bytes"] <= config["context_bytes"]
                obs = json.loads(event["messages"][-1]["content"])
                if obs["phase"] == "bank":
                    assert len(obs["proposals"]) <= config["bank_contacts"]
                if "bank_board" in obs:
                    assert len(obs["bank_board"]["tasks"]) <= config["bank_board_size"]
            if event["event"] == "snapshot":
                state = event["state"]
                agents = {a["id"]: a for a in state["agents"]}
                assert len(agents) <= config["max_agents"]
                for agent in agents.values():
                    assert len(agent["children"]) <= config["max_children"]
                    depth, parent, seen = 0, agent["parent"], set()
                    while parent:
                        assert parent not in seen
                        seen.add(parent)
                        depth += 1
                        parent = agents[parent]["parent"]
                    assert depth <= config["max_depth"]
                claims = [
                    t
                    for t in state["bank_board"]["tasks"]
                    if t["status"] in ("claimed", "blocked") and t["owner"]
                ]
                assert len({t["owner"] for t in claims}) == len(claims)
        assert bank_calls <= report["rounds"] + bool(config["bank_board_size"])
        assert len(final_grades) == 1 and final_grades[0]["status"] == "scored"
        assert final_grades[0]["total"] == 201
        assert final_grades[0]["score"] == report["score"]
        assert charges == Decimal(report["api_spent_usd"]) <= Decimal(job["cap_usd"])
        total += charges
        records.append(
            {
                "run": directory.name,
                "status": report["status"],
                "paid_usd": str(charges),
                "accounting_valid": True,
                "frozen_sources_match": True,
                "context_and_topology_bounds": True,
                "original_seed_matches": True,
            }
        )
    assert total <= allocation
    return {
        "valid": True,
        "total_usd": str(total),
        "allocated_ceiling_usd": str(allocation),
        "authorized_cap_usd": plan["authorized_cap_usd"],
        "runs": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args()
    result = validate(args.study)
    (args.study / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
