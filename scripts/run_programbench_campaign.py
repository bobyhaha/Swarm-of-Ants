"""Fresh ProgramBench colony or solo baseline, with one authorized campaign ledger."""

import argparse
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from ant_colony import campaign_budget
from ant_colony.baselines import SoloMAS, solo_config
from ant_colony.config import ColonyConfig
from ant_colony.engine import ColonyMAS
from ant_colony.environment import Task
from ant_colony.programbench import ProgramBenchWorkbench, PIN
from ant_colony.providers import OpenRouterProvider, BudgetStop, UncertainCall
from ant_colony.replay import write_replay
from ant_colony.sandbox import run
from ant_colony.storage import RunStore, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("task", "config", "models", "out", "ledger", "evaluator-python"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--max-usd", required=True, type=Decimal)
    parser.add_argument("--solo", action="store_true")
    parser.add_argument("--solo-model", choices=("astra", "sol", "luna"),
                        help="One actor working directly; --solo alone retains the Astra default")
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY must be in the process environment")
    if not Decimal(0) < args.max_usd <= Decimal(200):
        raise ValueError("Invalid episode ceiling")
    cfg = ColonyConfig.from_dict(json.loads(args.config.read_text()))
    models = json.loads(args.models.read_text())
    if any(m.get("reasoning_effort") != "high" for m in models.values()):
        raise ValueError("This study requires high reasoning for EVERY configured model")
    capital = int(args.max_usd * cfg.credits_per_usd)
    cfg = replace(
        cfg,
        initial_bank=capital,
        max_grant=min(cfg.max_grant, capital),
        application_budget=(capital if cfg.quality_first
                            else min(cfg.application_budget, capital // 3)),
        discovery_budget=(capital if cfg.quality_first else cfg.discovery_budget),
        trial_max_output_tokens=(cfg.max_output_tokens if cfg.quality_first
                                 else cfg.trial_max_output_tokens),
    )
    solo_model = args.solo_model or ("astra" if args.solo else None)
    if cfg.quality_first and (cfg.cost_mode != "api_usd" or cfg.rent_per_round):
        raise ValueError("Quality-first paid experiments require actual API-dollar costs and zero rent")
    if solo_model:
        cfg = solo_config(cfg, solo_model)
    # Inspect only evaluator provenance, not hidden test contents.
    verify = run([str(args.evaluator_python.absolute()), "-c",
                  "import importlib.metadata,json; print(json.loads(importlib.metadata.distribution('programbench').read_text('direct_url.json'))['vcs_info']['commit_id'])"])
    if verify.stdout.decode().strip() != PIN:
        raise ValueError("Native evaluator revision does not match task pin")
    ledger = campaign_budget.Ledger(args.ledger, parallel=True)
    if ledger.summary()["blocked"]:
        raise RuntimeError("Study ledger has unresolved calls; no request sent")
    store = RunStore(args.out)
    ident = args.out.name
    ledger.register(ident, float(args.max_usd))
    engine, failure = None, None
    try:
        provider = OpenRouterProvider(models, args.max_usd, store.emit, timeout_seconds=600,
                                      request_pricing=cfg.request_pricing)
        original, native_http, current = provider.complete, provider._http, {}

        def complete(model, messages, max_tokens, call_id):
            current.update(model=model, call_id=call_id)
            return original(model, messages, max_tokens, call_id)

        def transport(payload):
            _, reserve = provider.reservation(current["model"], payload["messages"], payload["max_tokens"])
            obs = json.loads(payload["messages"][-1]["content"])
            while True:
                if (store.directory / "STOP").exists():
                    raise BudgetStop("Episode STOP file exists")
                try:
                    call = ledger.reserve(ident,float(reserve),obs.get("phase"),obs.get("self","bank"),payload["model"])
                    break
                except campaign_budget.BudgetBusy:
                    time.sleep(0.25)
                except campaign_budget.BudgetStop as exc:
                    raise BudgetStop(str(exc)) from None
            store.emit("campaign_reserved",colony_call_id=current["call_id"],campaign_call_id=call,
                       reasoning_effort=payload["reasoning"]["effort"])
            try:
                response = provider.billing_response(native_http(payload))
                ledger.settle(call,response["usage"]["cost"],{"id":response.get("id"),"usage":response["usage"]})
                return response
            except BaseException as exc:
                if isinstance(exc, TimeoutError):
                    ledger.quarantine_timeout(call)
                else:
                    ledger.uncertain(call)
                store.emit("campaign_transport_error",campaign_call_id=call,
                           error_type=type(exc).__name__,http_status=getattr(exc,"code",None),
                           stop_scope="episode" if isinstance(exc, TimeoutError) else "campaign")
                raise UncertainCall("ProgramBench request failed; full study reservation retained") from None

        provider.complete, provider.transport = complete, transport
        bench = ProgramBenchWorkbench(args.task,store.directory / "coding",args.evaluator_python,store.emit,
                                      output_bytes=cfg.coding_output_bytes)
        bench.bank_finalization = cfg.bank_finalization
        bench.initial_check = bench.check(bench.seed,"initial")
        store.emit("programbench_baseline",**bench.initial_check)
        task = Task(bench.manifest["id"],
                    f"Rebuild ProgramBench instance {bench.manifest['id']} from its bundled documentation and normal reference behavior. "
                    "No original source, internet, tracing or decompilation. Use coding tools; read /reference-docs and SPEC.md. "
                    "Write an independent implementation and compile.sh producing executable. Keep compact PROJECT_NOTES.md. "
                    "Public probe tests are diagnostics; the protected native tests run once after final artifact selection. "
                    "Maximize society wealth after model-specific API costs. Model tiers do not assign worker roles or authority. "
                    "The bank should post a few BROAD, complementary outcomes with interface/acceptance constraints, "
                    "leaving subdivision, hiring, spawning, approaches and stopping decisions to recipients. "
                    f"Any funded agent can hire or spawn any allowed model within its {cfg.max_children}-child cap. "
                    "Root teams integrate conflict-free source changes between rounds; team members share files. "
                    "Avoid duplicate investigation, record reusable evidence, and finish a working build before the deadline. "
                    "The integrated public artifact is selected automatically at the round/dollar limit. "
                    + ("Roots report team readiness; only the bank may end the society episode early."
                       if cfg.bank_finalization else "A root can submit early."),
                    "protected native behavioral tests", "programbench")
        if cfg.quality_first:
            task = replace(task, problem=task.problem.replace(
                "Maximize society wealth after model-specific API costs.",
                "Maximize correctness and completeness within the authorized spending cap. Track actual costs; unused budget is not an objective."))
        if solo_model:
            task = replace(task, problem=(
                f"Rebuild ProgramBench instance {bench.manifest['id']} independently from /reference-docs, SPEC.md, "
                f"and ordinary black-box reference probes. You are the SINGLE-{solo_model.upper()} baseline: no other agents, "
                "hiring, spawning, applications or paid bank. Your only team wallet is already funded. "
                "Use the same isolated coding tools and bounded context as the society. "
                "No original source, internet, tracing, decompilation or reference copying/wrapping. "
                "Implement compile.sh that creates executable from source; keep compact PROJECT_NOTES.md. "
                "Public probes are diagnostics. Protected native tests run once after your artifact is selected. "
                "Maximize final verified score after API costs. Work directly, reuse evidence and test builds. "
                "Submit when finished. Otherwise your integrated public artifact is selected at the round/dollar limit."
            ))
        engine = (SoloMAS if solo_model else ColonyMAS)(cfg,task,provider,store.emit,workbench=bench)
        store.write_json("config.json",cfg.to_dict())
        store.write_json("models.json",models)
        store.write_json("episode.json",{"task":str(args.task.resolve()),"cap_usd":str(args.max_usd),
                                          "condition":"single-"+solo_model if solo_model else "mixed-colony",
                                          "initial_total_including_bank":1 if solo_model else cfg.initial_agents+1,
                                          "maximum_total_including_bank":1 if solo_model else cfg.max_agents+1,
                                          "reasoning_effort":"high","evaluator_revision":PIN})
        code = Path(__file__).resolve().parents[1]
        paths = list((code / "ant_colony").glob("*.py")) + [Path(__file__).resolve()]
        store.write_json("implementation.json",{str(p.relative_to(code)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        for p in paths:
            target = store.directory / "implementation-sources" / p.relative_to(code)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
        engine.run()
    except Exception as exc:
        failure = type(exc).__name__ + ": " + str(exc)
        store.emit("runner_error",reason=failure)
        print(failure,flush=True)
    finally:
        if engine:
            store.write_json("report.json",engine.report())
        store.write_json("campaign-accounting.json",ledger.summary(ident))
        ledger.close()
        store.close()
    if engine:
        store.write_json("audit.json",audit(store.path))
        write_replay(store.directory)
        print(json.dumps(engine.report(),indent=2),flush=True)
    return int(failure is not None)


if __name__ == "__main__":
    raise SystemExit(main())
