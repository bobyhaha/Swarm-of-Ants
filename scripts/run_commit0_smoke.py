"""Real Docker/Commit0 integration with a SCRIPTED policy, not model performance."""

import argparse
from dataclasses import replace
from decimal import Decimal
import json

from ant_colony.commit0 import Commit0Workbench
from ant_colony.config import ColonyConfig
from ant_colony.engine import ColonyMAS
from ant_colony.environment import Task
from ant_colony.providers import Generation
from ant_colony.replay import write_replay
from ant_colony.storage import RunStore, audit


class CodingFixture:
    scripted = True
    spent = reserved = Decimal(0)

    def __init__(self, scaffold):
        # An explicit small fixture change, never a reference implementation.
        self.content = (
            (scaffold / "tinydb/storages.py").read_text()
            + """

    def read(self):
        return self.memory

    def write(self, data):
        self.memory = data
"""
        )
        self.utils = (
            (scaffold / "tinydb/utils.py")
            .read_text()
            .replace(
                "\n    pass\n\nclass LRUCache",
                "\n    return baseclass if TYPE_CHECKING else object\n\nclass LRUCache",
                1,
            )
            .replace(
                "class FrozenDict(dict):",
                'def _immutable(*args, **kwargs):\n    raise TypeError("immutable")\n\nclass FrozenDict(dict):',
            )
        )
        self.step = 0

    def complete(self, model, messages, max_tokens, call_id):
        o = json.loads(messages[-1]["content"])
        if o["phase"] == "application":
            reply = {
                "proposal": {
                    "plan": "Smoke-test paged read, source write and protected tests.",
                    "amount": 50000,
                }
            }
        elif o["phase"] == "bank":
            reply = {
                "decisions": [
                    {
                        "agent": p["agent"],
                        "decision": "approve",
                        "amount": p["amount"],
                        "feedback": "Run the integration check and submit.",
                    }
                    for p in o["proposals"]
                ]
            }
        elif o["phase"] == "work":
            self.step += 1
            if self.step in (1, 3):
                reply = {
                    "tool": {
                        "kind": "read",
                        "path": "tinydb/utils.py" if self.step == 1 else "tinydb/storages.py",
                    }
                }
            elif self.step in (2, 4):
                reply = {
                    "tool": {
                        "kind": "write",
                        "path": "tinydb/utils.py" if self.step == 2 else "tinydb/storages.py",
                        "base_sha256": o["coding"]["result"]["sha256"],
                        "content": self.utils if self.step == 2 else self.content,
                    }
                }
            elif self.step == 5:
                reply = {"tool": {"kind": "test"}}
            else:
                reply = {"submit": "Scripted import/MemoryStorage fixture. Not a model-generated solution."}
        else:
            reply = {}
        text = json.dumps(reply)
        return Generation(text, (len(text.encode()) + 3) // 4, estimated=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    store = RunStore(a.out)
    try:
        bench = Commit0Workbench(a.task, store.directory / "coding", store.emit)
        cfg = replace(ColonyConfig(), initial_agents=1, max_rounds=3, max_output_tokens=2048, max_grant=50000)
        task = Task(
            "tinydb-smoke",
            "Exercise the TinyDB coding adapter, not full library quality.",
            "protected tests",
            "commit0",
        )
        engine = ColonyMAS(cfg, task, CodingFixture(bench.seed), store.emit, workbench=bench)
        store.write_json("config.json", cfg.to_dict())
        try:
            report = engine.run()
        except RuntimeError as error:
            report = engine.report()
            store.write_json("fixture-error.json", {"error": str(error), "score": None})
        store.write_json("report.json", report)
    finally:
        store.close()
    store.write_json("audit.json", audit(store.path))
    write_replay(store.directory)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
