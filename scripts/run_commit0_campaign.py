"""Local live pilot using the existing credential and shared campaign dollar ledger.

The standalone CLI uses an environment key. This optional workspace bridge also
honors the existing campaign's STOP flag, unresolved reservations and $100 cap.
It changes no sibling source files and never resets or acknowledges old calls.
"""

import argparse
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from ant_colony.commit0 import Commit0Workbench
from ant_colony.config import ColonyConfig
from ant_colony.engine import ColonyMAS
from ant_colony.environment import Task
from ant_colony.providers import OpenRouterProvider, BudgetStop, UncertainCall
from ant_colony.replay import write_replay
from ant_colony.storage import RunStore, audit


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--task", required=True, type=Path)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--models", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-usd", type=Decimal, default=Decimal("3"))
    p.add_argument("--parallel", action="store_true", help="Share the campaign's two live request slots")
    a = p.parse_args()
    cfg = ColonyConfig.from_dict(json.loads(a.config.read_text()))
    models = json.loads(a.models.read_text())
    budget = load_module("shared_campaign_budget", a.campaign / "codetest/budget.py")
    keychain = load_module("campaign_keychain", a.campaign / "scripts/openrouter_keychain.py")
    key = os.environ.get("OPENROUTER_API_KEY") or keychain.load_key()
    if not key:
        raise RuntimeError("No environment or configured campaign Keychain credential")
    os.environ["OPENROUTER_API_KEY"] = key
    del key
    ledger = budget.Ledger(a.campaign / "artifacts/campaign.sqlite", parallel=a.parallel)
    if ledger.summary()["blocked"] or (ledger.summary()["pending"] and not a.parallel):
        raise RuntimeError("Campaign has unresolved calls; no paid request sent")
    store = RunStore(a.out)
    run_id = "ant-colony/" + a.out.name
    ledger.register(run_id, float(a.max_usd))
    engine = None
    failure = None
    try:
        provider = OpenRouterProvider(models, a.max_usd, store.emit)
        native_http = provider._http
        current = {}
        original_complete = provider.complete

        def complete(model, messages, max_tokens, call_id):
            current.update(model=model, call_id=call_id)
            return original_complete(model, messages, max_tokens, call_id)

        provider.complete = complete

        def transport(payload):
            if (store.directory / "STOP").exists():
                raise BudgetStop("Run STOP file exists")
            settings = models[current["model"]]
            reserve = (
                settings["input_token_ceiling"] * Decimal(str(settings["input_usd_per_million"]))
                + payload["max_tokens"] * Decimal(str(settings["output_usd_per_million"]))
            ) / 1_000_000
            obs = json.loads(payload["messages"][-1]["content"])
            while True:
                try:
                    ident = ledger.reserve(
                        run_id,
                        float(reserve),
                        obs.get("phase", "judge"),
                        obs.get("self", "bank"),
                        payload["model"],
                    )
                    break
                except budget.BudgetBusy:
                    if (store.directory / "STOP").exists():
                        raise BudgetStop("Run STOP file exists") from None
                    time.sleep(0.5)  # Admission wait only; no HTTP request has been sent.
                except budget.BudgetStop as exc:
                    raise BudgetStop(str(exc)) from None
            store.emit("campaign_reserved", colony_call_id=current["call_id"], campaign_call_id=ident)
            try:
                response = native_http(payload)
                usage = response["usage"]
                ledger.settle(ident, usage["cost"], {"id": response.get("id"), "usage": usage})
                return response
            except BaseException:
                ledger.uncertain(ident)
                raise UncertainCall("Shared campaign request stopped; reservation retained") from None

        provider.transport = transport
        bench = Commit0Workbench(
            a.task, store.directory / "coding", store.emit, output_bytes=cfg.coding_output_bytes
        )
        bench.initial_check = bench.check(bench.seed, "initial")
        store.emit("commit0_baseline", **bench.initial_check)
        task = Task(
            bench.manifest["id"],
            f"Reconstruct the Commit0 {bench.manifest['id']} Python library from the original incomplete scaffold. "
            "Use isolated exec commands to inspect, implement, and test; retain your plan and next step in PROJECT_NOTES.md. "
            "Read SPEC.md and original visible tests. Choose useful responsibilities from evidence; no predefined worker roles. "
            "Managers may divide work and hire or spawn agents when their expected contribution justifies the cost. "
            "Root teams own private branches; conflict-free source changes are checkpointed between rounds, "
            "and new teams inherit the integrated code. Avoid duplicate investigations and conflicting file ownership. "
            "The public checkpoint is exported and graded automatically at the deadline; a root may submit early. "
            f"Budget for coding calls; each reserves up to {cfg.max_output_tokens} credits but pays actual output only.",
            "protected visible tests",
            "commit0",
        )
        engine = ColonyMAS(cfg, task, provider, store.emit, workbench=bench)
        store.write_json("config.json", cfg.to_dict())
        store.write_json("models.json", models)
        import hashlib

        code_root = Path(__file__).resolve().parents[1]
        store.write_json(
            "implementation.json",
            {
                str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (code_root / "ant_colony").glob("*.py")
            },
        )
        import shutil

        for path in (code_root / "ant_colony").glob("*.py"):
            destination = store.directory / "implementation-sources" / "ant_colony" / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        shutil.copyfile(__file__, store.directory / "implementation-sources" / "run_commit0_campaign.py")
        store.write_json(
            "pilot.json",
            {
                "campaign_run": run_id,
                "cap_usd": str(a.max_usd),
                "kind": "single-task development pilot; not a controlled benchmark comparison",
            },
        )
        engine.run()
    except Exception as exc:
        failure = type(exc).__name__ + ": " + str(exc)
        print(failure, file=sys.stderr)
    finally:
        if engine:
            store.write_json("report.json", engine.report())
        store.write_json("campaign-accounting.json", ledger.summary(run_id))
        ledger.close()
        store.close()
    if engine:
        store.write_json("audit.json", audit(store.path))
        write_replay(store.directory)
        print(json.dumps(engine.report(), indent=2))
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
