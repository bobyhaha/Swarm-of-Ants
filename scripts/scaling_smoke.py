"""Offline protocol scaling check. No inference-quality or model-cost claims."""

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

from ant_colony.config import ColonyConfig
from ant_colony.engine import ColonyMAS
from ant_colony.environment import Task
from ant_colony.providers import Generation


class AdmissionOnly:
    scripted = True
    spent = reserved = 0

    def complete(self, model, messages, max_tokens, call_id):
        obs = json.loads(messages[-1]["content"])
        reply = (
            {"proposal": {"plan": "A bounded test proposal", "amount": 1000, "task_id": "task-0"}}
            if obs["phase"] == "application"
            else {}
        )
        if obs.get("initial_planning"):
            reply = {
                "tasks": [
                    {
                        "id": f"task-{i}",
                        "title": "Bounded deliverable",
                        "scope": f"Independent scope {i}",
                        "acceptance": "Evidence of completion",
                        "status": "open",
                        "feedback": "",
                    }
                    for i in range(obs["bank_board"]["capacity"])
                ]
            }
        text = json.dumps(reply)
        return Generation(text, (len(text) + 3) // 4, estimated=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--populations", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--board-size", type=int, default=0)
    args = parser.parse_args()
    results = []
    for size in args.populations:
        counter = Counter()

        def record(event, **data):
            if event == "call":
                counter[data["kind"]] += 1

        config = replace(
            ColonyConfig(),
            initial_agents=size,
            max_agents=size,
            max_rounds=1,
            initial_bank=max(100000, size * 1000),
            bank_board_size=args.board_size,
        )
        engine = ColonyMAS(
            config, Task("scale", "Protocol test only; do not solve.", "unused"), AdmissionOnly(), record
        )
        start = perf_counter()
        report = engine.run()
        results.append(
            {
                "population": size,
                "application_calls": counter["application"],
                "unfunded_social_calls": counter["formation"],
                "bank_calls": counter["bank"],
                "board_size": config.bank_board_size,
                "context_cap_bytes": config.context_bytes,
                "max_observed_context_bytes": report["max_context_bytes"],
                "elapsed_seconds": round(perf_counter() - start, 4),
            }
        )
        assert counter["application"] <= config.bank_contacts
        assert counter["formation"] == 0
        assert counter["bank"] <= 1 + bool(config.bank_board_size)
        assert report["max_context_bytes"] <= config.context_bytes
    data = {"kind": "scripted_admission_scaling_smoke_not_a_quality_benchmark", "results": results}
    text = json.dumps(data, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
