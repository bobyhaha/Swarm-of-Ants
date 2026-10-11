"""Append-only audit events and atomic reports. No implicit paid-call resume."""

import json
import os
from pathlib import Path
from decimal import Decimal, ROUND_CEILING

from .context import dumps


class RunStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.path = self.directory / "events.jsonl"
        self.file = self.path.open("x", encoding="utf-8")
        self.sequence = 0

    def emit(self, event, **data):
        self.sequence += 1
        self.file.write(dumps({"seq": self.sequence, "event": event, **data}) + "\n")
        self.file.flush()
        # Persist reservations before any possibly billed request.
        if event in {"api_reserved", "api_settled", "call_reserved", "call", "finished", "snapshot"}:
            os.fsync(self.file.fileno())

    def write_json(self, name, value):
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    def close(self):
        self.file.close()


def read_events(path):
    with Path(path).open(encoding="utf-8") as handle:
        for expected, line in enumerate(handle, 1):
            event = json.loads(line)
            if event["seq"] != expected:
                raise ValueError("Noncontiguous event log")
            yield event


def audit(path):
    """Recompute conservation and usage from events, independently of engine state."""
    bank = None
    wallets = {}
    burned = rewards = tokens = uncertain = calls = max_bytes = 0
    initial = None
    rewarded_tasks, pending, used = set(), {}, set()
    final = None
    config = None
    api_pending, api_spent = {}, Decimal(0)
    sponsorships, application_spent = {}, 0
    discovery_spent = 0
    discovery_authorizations, discovery_counts, discovery_rounds = {}, {}, {}
    generation_charges = 0
    for event in read_events(path):
        kind = event["event"]
        if kind == "start":
            if initial is not None:
                raise ValueError("Duplicate start")
            config = event["config"]
            initial = config["initial_bank"]
            bank = initial
            wallets["bank"] = bank
        elif kind == "transfer":
            source, target, value = event["source"], event["target"], event["amount"]
            if type(value) is not int or value < 0 or wallets.get(source, 0) < value:
                raise ValueError("Invalid transfer")
            wallets[source] = wallets.get(source, 0) - value
            wallets[target] = wallets.get(target, 0) + value
        elif kind == "burn":
            wallet, value = event["wallet"], event["amount"]
            if type(value) is not int or value < 0 or wallets.get(wallet, 0) < value:
                raise ValueError("Invalid burn")
            wallets[wallet] = wallets.get(wallet, 0) - value
            burned += value
        elif kind == "reward":
            if event["task_id"] in rewarded_tasks:
                raise ValueError("Duplicate terminal reward")
            value = event["amount"]
            if type(value) is not int or value < 0:
                raise ValueError("Invalid reward")
            rewarded_tasks.add(event["task_id"])
            wallets["bank"] += value
            rewards += value
        elif kind == "application_sponsored":
            sponsorships[event["agent"]] = event
        elif kind == "discovery_sponsored":
            discovery_authorizations[event["agent"]] = event
        elif kind == "call_reserved":
            key = event["call_id"]
            if key in used:
                raise ValueError("Duplicate call ID")
            used.add(key)
            pending[key] = event
            calls += 1
            if "application_budget" in config:
                if event.get("kind") == "application":
                    sponsorship = sponsorships.pop(event["actor"], None)
                    if not sponsorship or any(
                        (
                            event.get("payer") != sponsorship["target"],
                            event["wallet"] != sponsorship["wallet"],
                            type(event["credits"]) is not int or type(sponsorship["maximum_credits"]) is not int
                            or not 0 < event["credits"] <= sponsorship["maximum_credits"],
                        )
                    ):
                        raise ValueError("Application has no matching sponsor authorization")
                elif event.get("kind") == "discovery":
                    sponsorship = discovery_authorizations.pop(event["actor"], None)
                    if (not config.get("quality_first") or not sponsorship
                            or event["wallet"] != "bank" or event.get("payer") != "bank"
                            or not 0 < event["credits"] <= sponsorship["maximum_credits"]):
                        raise ValueError("Discovery has no matching bank authorization")
                    actor = event["actor"]
                    discovery_counts[actor] = discovery_counts.get(actor, 0) + 1
                    discovery_rounds.setdefault(event["round"], set()).add(actor)
                    if (discovery_counts[actor] > config["discovery_turns"]
                            or len(discovery_rounds[event["round"]]) > config["discovery_contacts"]):
                        raise ValueError("Discovery call/admission bound exceeded")
                elif event["actor"].startswith("a"):
                    if event["wallet"] == "bank" or event.get("payer") != event["actor"]:
                        raise ValueError("Ordinary agent work requires its own funded allowance")
            if event["input_bytes"] > config["context_bytes"]:
                raise ValueError("Context budget exceeded")
            from .context import byte_size

            if byte_size(event["messages"]) != event["input_bytes"]:
                raise ValueError("Recorded context byte count is incorrect")
            max_bytes = max(max_bytes, event["input_bytes"])
        elif kind in {"call", "call_uncertain", "call_cancelled"}:
            reservation = pending.pop(event["call_id"], None)
            if reservation is None:
                raise ValueError("Call without reservation")
            if kind == "call":
                count = event["output_tokens"]
                if type(count) is not int or not 0 <= count <= config["max_output_tokens"]:
                    raise ValueError("Output limit violation")
                tokens += count
                charge = count * config["token_price"]
                if config.get("cost_mode") == "api_usd":
                    charge = int((Decimal(event["cost_usd"]) * config["credits_per_usd"]).to_integral_value(rounding=ROUND_CEILING))
                    if event.get("charged_credits") != charge or event["estimated"]:
                        raise ValueError("Model-specific credit charge disagrees with API receipt")
                if charge > reservation["credits"]:
                    raise ValueError("Credit charge exceeds reservation")
                generation_charges += charge
            elif kind == "call_uncertain":
                uncertain += event["credits"]
            if reservation.get("kind") == "discovery" and reservation["wallet"] == "bank":
                discovery_spent += charge if kind == "call" else event.get("credits", 0)
                if discovery_spent > config.get("discovery_budget", 0):
                    raise ValueError("Discovery pool exceeded")
            if reservation.get("kind") == "application" and reservation["wallet"] == "bank":
                if kind == "call":
                    application_spent += charge
                elif kind == "call_uncertain":
                    application_spent += event["credits"]
        elif kind == "api_reserved":
            api_pending[event["call_id"]] = Decimal(event["usd"])
        elif kind in {"api_settled", "api_cancelled"}:
            if event["call_id"] not in api_pending:
                raise ValueError("API settlement without reservation")
            del api_pending[event["call_id"]]
            if kind == "api_settled":
                api_spent += Decimal(event["usd"])
        elif kind == "snapshot":
            state = event["state"]
            if sum(wallets.values()) != state["wealth"]:
                raise ValueError("Snapshot wealth disagrees with replayed ledger")
        elif kind == "finished":
            final = event["report"]
    if initial is None or final is None:
        raise ValueError("Incomplete run log; reconcile outstanding reservations before any restart")
    wealth = sum(wallets.values())
    if wealth != initial + rewards - burned or wealth != final["final_wealth"]:
        raise ValueError("Wealth does not reconcile")
    if tokens != final["output_tokens"] or calls != final["calls"] or pending:
        raise ValueError("Call usage does not reconcile")
    if not config["rent_per_round"] and burned != generation_charges + uncertain:
        raise ValueError("Token charges do not reconcile")
    if "application_budget" in config and (
        application_spent != final["application_spent"] or application_spent > config["application_budget"]
    ):
        raise ValueError("Sponsored application pool does not reconcile")
    if discovery_spent != final.get("discovery_spent", 0):
        raise ValueError("Sponsored discovery pool does not reconcile")
    if api_spent != Decimal(final["api_spent_usd"]) or sum(api_pending.values()) != Decimal(
        final["api_reserved_usd"]
    ):
        raise ValueError("Dollar usage does not reconcile")
    return {
        "accounting_valid": True,
        "final_wealth": wealth,
        "burned": burned,
        "rewards": rewards,
        "calls": calls,
        "output_tokens": tokens,
        "max_context_bytes": max_bytes,
        "uncertain_credits": uncertain,
        "application_spent": application_spent,
        "discovery_spent": discovery_spent,
        "unresolved_api_calls": len(api_pending),
        "complete_evaluation": final["status"] == "completed",
        "scripted_demo": final["scripted_demo"],
    }
