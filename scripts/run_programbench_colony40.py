"""A fresh 40-agent, 60-round colony sharing the original authorized campaign ledger."""

import argparse
from collections import Counter
from decimal import Decimal
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from ant_colony.campaign_budget import Ledger
from ant_colony.config import ColonyConfig
from ant_colony.storage import audit
from run_programbench_repaired import financial_failure, write


def experiment_config(base, *, bank_contacts=2, manager_contacts=2, max_children=2, model_mix="mixed", hierarchy_bias=False, quality_first=False):
    if model_mix not in {"mixed", "all-astra"}:
        raise ValueError("Unknown model mix")
    cfg = {**base, "initial_agents": 39, "max_agents": 63, "max_rounds": 60,
           "initial_models": ["astra"] * 3 + ["sol"] * 16 + ["luna"] * 20,
           "bank_model": "astra", "funding_recovery": True, "persistent_teams": True,
           "bank_finalization": True, "request_pricing": True, "event_driven_bank": True,
           "standing_grants": True, "progress_control": True,
           "coordination_max_output_tokens": 4096, "coordination_context_bytes": 32000,
           "bank_review_interval": 4, "standing_grant_rounds": 6, "stagnation_rounds": 6,
           "bank_contacts": bank_contacts, "manager_contacts": manager_contacts,
           "max_children": max_children, "max_depth": 6, "hierarchy_bias": hierarchy_bias}
    cfg["worker_models"] = list(cfg["worker_models"])
    if model_mix == "all-astra":
        cfg["initial_models"] = ["astra"] * cfg["initial_agents"]
        cfg["worker_models"] = ["astra"]
    if quality_first:
        cfg.update(quality_first=True, cost_mode="api_usd", rent_per_round=0,
                   discovery_budget=200_000_000, discovery_contacts=8, discovery_turns=3,
                   message_contacts=4, trial_max_output_tokens=16_384, text_bytes=1800,
                   max_grant=60_000_000, grant_rounds=16)
    ColonyConfig.from_dict(cfg)
    return cfg


def check_admission(accounting, cap, episode_ceiling=Decimal(15)):
    if accounting["blocked"]:
        raise ValueError("Unresolved prior billing; no experiment may start")
    if not Decimal(0) < cap <= episode_ceiling <= Decimal(200):
        raise ValueError("This experiment is capped at $" + str(episode_ceiling))
    # Live reservations are included. Existing future episode caps are ceilings,
    # not guarantees; every controller uses the durable authorized ceiling.
    if Decimal(str(accounting["accounted_usd"])) + cap > Decimal(str(accounting.get("ceiling_usd", 100))):
        raise ValueError("Insufficient shared campaign headroom for the new run")


def check_attempt_budget(ledger, previous_attempts, total, new_cap):
    """All prior attempt ceilings and this cap fit one experiment total."""
    if isinstance(previous_attempts, str):
        previous_attempts = [previous_attempts]
    previous_attempts = list(previous_attempts or [])
    if not previous_attempts and total is None:
        return None
    if not previous_attempts or total is None or not Decimal(0) < total <= Decimal(200):
        raise ValueError("Supply prior attempts and a positive experiment total up to $200")
    attempts, previous_ceiling = [], Decimal(0)
    for previous_attempt in previous_attempts:
        account = ledger.summary(previous_attempt)
        if account['pending'] or account['blocked']:
            raise ValueError("Previous attempt still has pending or unresolved requests")
        with ledger.connect() as db:
            old = db.execute('SELECT cap FROM runs WHERE id=?', (previous_attempt,)).fetchone()
        if old is None:
            raise ValueError("Unknown previous attempt " + previous_attempt)
        sealed_cap = Decimal(str(old['cap']))
        previous_ceiling += sealed_cap
        attempts.append({'id': previous_attempt, 'sealed_cap_usd': str(sealed_cap),
                         'accounting': account})
    if previous_ceiling + new_cap > total:
        raise ValueError("Combined attempt ceilings exceed the total experiment budget")
    return {'ids': previous_attempts, 'sealed_cap_usd': str(previous_ceiling),
            'attempts': attempts}


def campaign_failure(accounting, global_accounting, check, events):
    # A conservatively held timeout stops only its own episode. Missing audits,
    # unknown failures, pending calls, and actual contract violations remain global.
    if global_accounting["blocked"]:
        return True
    isolated = accounting.get("quarantined_timeouts", 0) > 0
    adjusted = {**accounting, "blocked": accounting["blocked"] and not isolated}
    return financial_failure(adjusted, check, events)


def check_episode(ledger, ident):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,95}", ident):
        raise ValueError("Episode ID must be a short lowercase name without path separators")
    with ledger.connect() as db:
        if db.execute("SELECT 1 FROM runs WHERE id=?", (ident,)).fetchone():
            raise ValueError("Episode ID already exists in the shared ledger; use a fresh ID")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior-study", type=Path, required=True)
    parser.add_argument("--max-usd", type=Decimal, default=Decimal(15))
    parser.add_argument("--episode-id", default="v3-colony40-xsv")
    parser.add_argument("--bank-contacts", type=int, default=2)
    parser.add_argument("--manager-contacts", type=int, default=2)
    parser.add_argument("--max-children", type=int, default=2)
    parser.add_argument("--model-mix", choices=("mixed", "all-astra"), default="mixed")
    parser.add_argument("--hierarchy-bias", action="store_true")
    parser.add_argument("--quality-first", action="store_true")
    parser.add_argument("--previous-attempt", action="append", dest="previous_attempts",
                        help="Prior attempt in this experiment; may be repeated")
    parser.add_argument("--experiment-total-usd", type=Decimal)
    parser.add_argument("--prompt-key", action="store_true")
    args = parser.parse_args()
    code, study, prior = Path(__file__).resolve().parents[1], args.out.resolve(), args.prior_study.resolve()
    previous = json.loads((prior / "study.json").read_text())
    ledger_path = Path(previous["ledger"])
    if not ledger_path.is_file() or (ledger_path.parent / "STOP").exists():
        raise RuntimeError("Original ledger must exist and no global STOP may be present")
    ledger = Ledger(ledger_path, parallel=True)
    check_episode(ledger, args.episode_id)
    lock = (ledger_path.parent / (args.episode_id + "-controller.lock")).open("a+b")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_episode(ledger, args.episode_id)
    before = ledger.summary()
    episode_ceiling = (Decimal(200) if args.quality_first else Decimal(100)) if args.model_mix == "all-astra" else Decimal(15)
    check_admission(before, args.max_usd, episode_ceiling)
    prior_attempt = check_attempt_budget(ledger, args.previous_attempts, args.experiment_total_usd, args.max_usd)
    if args.prompt_key and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = getpass.getpass("OpenRouter API key (not saved): ")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required")
    before = ledger.summary()
    check_admission(before, args.max_usd, episode_ceiling)
    prior_attempt = check_attempt_budget(ledger, args.previous_attempts, args.experiment_total_usd, args.max_usd)
    study.mkdir(parents=True, exist_ok=False)
    runtime = study / "runtime"
    for name in ("ant_colony", "hayekmas"):
        shutil.copytree(code / name, runtime / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (runtime / "scripts").mkdir()
    for name in ("run_programbench_campaign.py", Path(__file__).name, "run_programbench_repaired.py"):
        shutil.copyfile(code / "scripts" / name, runtime / "scripts" / name)
    cfg = experiment_config(json.loads((code / "configs/programbench-mixed-high.json").read_text()),
                            bank_contacts=args.bank_contacts, manager_contacts=args.manager_contacts,
                            max_children=args.max_children, model_mix=args.model_mix,
                            hierarchy_bias=args.hierarchy_bias, quality_first=args.quality_first)
    write(study / "main-config.json", cfg)
    model_settings = json.loads((code / "configs/programbench-high-models.json").read_text())
    allowed_models = set(cfg["worker_models"]) | {cfg["bank_model"]}
    write(study / "models.json", {name: settings for name, settings in model_settings.items()
                                  if name in allowed_models})
    hashes = {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in runtime.rglob("*") if p.is_file()}
    write(study / "source-hashes.json", hashes)
    row = {"id": args.episode_id, "task": "xsv", "cap": float(args.max_usd),
           "solo": False, "solo_model": None, "pilot": False}
    society = Counter(cfg["initial_models"])
    society[cfg["bank_model"]] += 1
    plan = {"status": "running", "benchmark": "ProgramBench", "ledger": str(ledger_path),
            "prior_study": str(prior), "reasoning_effort": "high", "max_rounds": 60,
            "initial_society": {**society, "bank_included": True}, "model_mix": args.model_mix,
            "max_population_including_bank": 64, "max_direct_children": cfg["max_children"], "max_depth": 6,
            "bank_contacts": cfg["bank_contacts"], "manager_contacts": cfg["manager_contacts"],
            "allocated_maximum_usd": float(args.max_usd), "global_ceiling_usd": before["ceiling_usd"],
            "accounting_at_start": before, "max_concurrent_api_requests_shared": 2,
            "episodes": [row], "fresh_seed": True,
            "selection": "One final integrated artifact; native hidden tests supply no feedback to agents.",
            "budget_amendment": "User authorized a fresh 40-agent restart and a $500 cumulative campaign cap, including all prior spending and reservations. This episode retains a $15 ceiling. All requests use the original durable campaign ledger; no old spending is reset.",
            "hypothesis": "Outcome ownership prompts and explicit, reviewed recruitment shortfalls let funded workers initiate useful recursive delegation. Renewals preserve the existing outcome, and a rejected top-up defers rather than permanently bars a funded worker from requesting support.",
            "emergence_metrics": ["working agents by model", "introductions routed and forwarded", "recruitment funding requests and review outcomes", "recruitment after funding", "trial hires", "hires and spawns", "peak depth", "productive workers at depth >=2", "standing grants used", "coordination cost share", "context bound", "final score and actual cost"],
            "limitations": "Prompt changes, recruitment funding, and renewal fixes change together; one fresh run cannot isolate their individual effects. Population, horizon, and episode ceiling match the preceding 40-agent condition. Compare explicit configured contact and child caps. A deeper tree alone is not evidence of useful emergence."}
    if args.model_mix == "all-astra":
        plan["hypothesis"] = "With the latest delegation mechanism unchanged, an all-Astra society may improve verified quality and autonomous collaboration relative to the mixed-model v6 run. Initial workers and any new spawns must use Astra."
        plan["limitations"] = "Only model composition and the allowed model set change from the chosen v6 reference configuration; prompts and protocol are unchanged. Model-specific reserves, example budgets and affordable participation change as a consequence. Forty initial agents does not imply forty funded workers. This is one fresh run with no controlled provider seed; equal dollar ceilings do not imply equal actual cost or calls. Compare the recorded caps and observed stopping rounds."
        plan["budget_amendment"] = f"User authorized this all-Astra version with a ${args.max_usd} attempt ceiling within the existing ${before['ceiling_usd']} cumulative campaign cap; no prior charges or holds are reset."
    plan['hierarchy_bias'] = cfg['hierarchy_bias']
    plan['quality_first'] = cfg['quality_first']
    if cfg['hierarchy_bias']:
        plan['hypothesis'] = "An explicitly stated preference for useful hierarchical delegation, inherited by the bank and every worker, may encourage recursive teams under the authorized all-Astra budget."
        plan['limitations'] = "The user requested both a larger budget and explicit hierarchy bias. This changes model composition, budget and prompting relative to v6 and is not a model-only or equal-budget comparison. A deeper tree alone is not useful emergence. Initial population is not active population; native results are unavailable to the agents."
    if args.quality_first:
        plan['hypothesis'] = "Quality-first objectives, sponsored discovery, direct offers, addressable bounded messages and recursive ownership may improve useful participation and task correctness."
        plan['limitations'] = "A new compound mechanism treatment: objective, prompts, admission, communication, hierarchy preference and budget differ. One run with no controlled provider seed; compare outcomes without causal attribution to a single change."
        plan['budget_amendment'] = f"User authorized a fresh quality-first 40-Astra experiment capped at ${args.max_usd}, within the existing $500 campaign including all historical spending and unresolved reservations. This is a new attempt, not restored agent state."
        plan['discovery'] = {k:cfg[k] for k in ('discovery_budget','discovery_contacts','discovery_turns')}
    if prior_attempt:
        plan['previous_attempt'] = prior_attempt
        plan['experiment_total_cap_usd'] = str(args.experiment_total_usd)
        plan['budget_amendment'] += (f" Prior attempts {args.previous_attempts} are retained and charged against the same "
                                     f"${args.experiment_total_usd} experiment total. Their combined sealed ledger "
                                     "ceilings plus this attempt ceiling cannot exceed that total. This is a fresh "
                                     "attempt, not restoration of live agent state.")
    write(study / "study.json", plan)
    write(study / "budget-amendment.json", {
        "prior_study": str(prior), "global_ceiling_usd": before["ceiling_usd"],
        "accounting_before": before, "new_episode_cap_usd": str(args.max_usd),
        "prior_attempts": prior_attempt["attempts"] if prior_attempt else [],
        "experiment_total_cap_usd": str(args.experiment_total_usd) if prior_attempt else None,
        "prior_attempt_sealed_caps_usd": prior_attempt["sealed_cap_usd"] if prior_attempt else None,
        "scope": plan["budget_amendment"],
    })
    env = dict(os.environ, PYTHONPATH=str(runtime),
               ANT_COLONY_NATIVE_LOCK=str(ledger_path.parent / "native-evaluation.lock"))
    command = [sys.executable, str(runtime / "scripts/run_programbench_campaign.py"),
               "--task", str(code.parent / "code-testing-v1/datasets/programbench-xsv/task.json"),
               "--config", str(study / "main-config.json"), "--models", str(study / "models.json"),
               "--out", str(study / row["id"]), "--ledger", str(ledger_path),
               "--evaluator-python", str(code.parent / "code-testing-v1/.venv/bin/python"),
               "--max-usd", str(args.max_usd)]
    print("Starting " + row["id"] + ": 40 initial agents, up to 60 rounds, ceiling $" + str(args.max_usd), flush=True)
    try:
        with (study / (row["id"] + ".log")).open("x") as log:
            result = subprocess.run(command, cwd=runtime, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        directory = study / row["id"]
        event_file, report_file = directory / "events.jsonl", directory / "report.json"
        events = [json.loads(line) for line in event_file.read_text().splitlines()] if event_file.exists() else []
        report = json.loads(report_file.read_text()) if report_file.exists() else {}
        check = audit(event_file) if report_file.exists() else {}
        account = ledger.summary(row["id"])
        unsafe = campaign_failure(account, ledger.summary(), check, events)
        if unsafe:
            (ledger_path.parent / "STOP").write_text("Financial/inference-contract failure in " + row["id"] + ".\n")
        healthy = result.returncode == 0 and report.get("status") == "completed" and check.get("complete_evaluation") and not unsafe
        plan["status"] = "completed" if healthy else "stopped_transport" if account.get("quarantined_timeouts") and not unsafe else "failed_episode"
        write(study / "outcomes.json", [{"id": row["id"], "status": plan["status"], "report": report,
                                          "audit": check, "accounting": account, "returncode": result.returncode}])
    except BaseException as exc:
        plan["status"], plan["error_type"] = "controller_failed", type(exc).__name__
        raise
    finally:
        write(study / "study.json", plan)
        write(study / "budget.json", ledger.summary())
        ledger.close()
        lock.close()


if __name__ == "__main__":
    main()
