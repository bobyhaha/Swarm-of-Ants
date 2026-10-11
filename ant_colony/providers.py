"""Metered providers. The live backend never retries a possibly billed request."""

from dataclasses import dataclass
from decimal import Decimal
import json
import os
import queue
import threading
from urllib.request import Request, build_opener, HTTPRedirectHandler

from .context import dumps
from .reservations import request_reservation


class BudgetStop(RuntimeError):
    """Preflight refused a call; nothing was sent."""


class UncertainCall(RuntimeError):
    """A request may have been billed; keep its reservation and stop the run."""


@dataclass(frozen=True)
class Generation:
    text: str
    output_tokens: int
    input_tokens: int = 0
    cost_usd: str = "0"
    estimated: bool = False


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class OpenRouterProvider:
    """Explicit model/price allowlist and durable dollars reserved before I/O.

    input_token_ceiling is an operator-supplied model bound, not a guessed
    tokenizer ratio. Actual usage is checked. A provider breach invalidates the
    run; an API-key spending limit remains the independent external backstop.
    """

    scripted = False

    def __init__(self, models, max_usd, emit, *, transport=None, timeout_seconds=180, request_pricing=False):
        self.models = models
        self.request_pricing = request_pricing
        self.limit = Decimal(str(max_usd))
        if not self.limit.is_finite() or self.limit <= 0:
            raise ValueError("max_usd must be finite and positive")
        for alias, cfg in models.items():
            if not isinstance(cfg, dict) or set(cfg) - {
                "id",
                "input_usd_per_million",
                "output_usd_per_million",
                "input_token_ceiling",
                "reasoning_effort",
                "provider_ignore",
            }:
                raise ValueError("Unexpected model settings; credentials belong only in the environment")
            if not isinstance(alias, str) or not alias or not isinstance(cfg.get("id"), str) or not cfg["id"]:
                raise ValueError("Each model needs a nonempty alias and provider ID")
            if cfg["id"].startswith("REPLACE_"):
                raise ValueError("Replace example model IDs before a live run")
            for key in ("input_usd_per_million", "output_usd_per_million"):
                price = Decimal(str(cfg[key]))
                if not price.is_finite() or not 0 <= price <= 1_000_000:
                    raise ValueError(
                        "Model price ceilings must be finite and between 0 and 1000000 USD/million"
                    )
            if type(cfg["input_token_ceiling"]) is not int or cfg["input_token_ceiling"] < 1:
                raise ValueError("input_token_ceiling must be a positive integer")
            if "provider_ignore" in cfg and (
                not isinstance(cfg["provider_ignore"], list)
                or any(not isinstance(p, str) for p in cfg["provider_ignore"])
            ):
                raise ValueError("provider_ignore must be a list of provider names")
            if cfg.get("reasoning_effort") not in {None, "none", "low", "medium", "high", "xhigh", "max"}:
                raise ValueError("Unknown reasoning effort")
        self.emit = emit
        self.spent = Decimal(0)
        self.reserved = Decimal(0)
        self.blocked = False
        self.transport = transport or self._http
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 600:
            raise ValueError("Invalid HTTP timeout")
        self.timeout_seconds = timeout_seconds

    @property
    def exposure(self):
        return self.spent + self.reserved

    def reservation(self, model, messages, max_tokens):
        return request_reservation(self.models[model], max_tokens,
                                   enabled=self.request_pricing, messages=messages)

    def _http(self, payload):
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise BudgetStop("Set OPENROUTER_API_KEY before running a live experiment")
        request = Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        # No arbitrary endpoint, redirects, plugins, tools, auto-routing, or retries.
        # A socket idle timeout alone permits periodic keep-alive bytes forever.
        # One daemon transport enforces an absolute deadline; it never retries.
        results = queue.Queue(maxsize=1)
        def fetch():
            try:
                with build_opener(NoRedirect).open(request, timeout=self.timeout_seconds) as response:
                    data = response.read(8_000_001)
                if len(data) > 8_000_000:
                    raise ValueError("Provider response exceeded transport size limit")
                results.put((True, json.loads(data)))
            except BaseException as exc:
                results.put((False, exc))
        threading.Thread(target=fetch, daemon=True).start()
        try:
            ok, value = results.get(timeout=self.timeout_seconds)
        except queue.Empty:
            raise TimeoutError("Absolute request deadline exceeded; billing remains reserved") from None
        if not ok:
            raise value
        return value

    @staticmethod
    def receipt(response):
        """Validate billing before indexing it; do not expose raw error bodies."""
        usage = response.get("usage") if isinstance(response, dict) else None
        if not isinstance(usage, dict):
            raise UncertainCall("Provider response has no usage receipt; full reservation retained")
        try:
            cost = Decimal(str(usage["cost"]))
            counts = (usage["prompt_tokens"], usage["completion_tokens"])
            if not cost.is_finite() or cost < 0 or any(type(t) is not int or t < 0 for t in counts):
                raise ValueError("Invalid usage")
        except (KeyError, ValueError, TypeError, ArithmeticError):
            raise UncertainCall("Provider usage receipt is missing or invalid; full reservation retained") from None
        return usage

    def billing_response(self, response):
        try:
            self.receipt(response)
        except UncertainCall:
            error = response.get("error") if isinstance(response, dict) else None
            meta = error.get("metadata") if isinstance(error, dict) else None
            # IDs and typed codes only; upstream text could echo credentials/input.
            def safe(value):
                import re
                return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", value) else None
            self.emit("api_missing_receipt", generation_id=safe(response.get("id")) if isinstance(response, dict) else None,
                      error_code=error.get("code") if isinstance(error, dict) and type(error.get("code")) is int else None,
                      error_type=safe(meta.get("error_type")) if isinstance(meta, dict) else None)
            raise
        return response

    def complete(self, model, messages, max_tokens, call_id):
        if self.blocked:
            raise BudgetStop("Provider stopped after uncertain billing or a usage violation")
        cfg = self.models[model]
        input_price = Decimal(str(cfg["input_usd_per_million"]))
        output_price = Decimal(str(cfg["output_usd_per_million"]))
        input_bound, reservation = self.reservation(model, messages, max_tokens)
        if self.exposure + reservation > self.limit:
            raise BudgetStop("Dollar budget cannot reserve the next call")
        payload = {
            "model": cfg["id"],
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "provider": {
                "sort": "price",
                "require_parameters": True,
                "allow_fallbacks": False,
                "max_price": {"prompt": float(input_price), "completion": float(output_price), "request": 0},
            },
        }
        if messages and '"phase"' in messages[-1]["content"]:
            payload["response_format"] = {"type": "json_object"}
        if cfg.get("provider_ignore"):
            payload["provider"]["ignore"] = cfg["provider_ignore"]
        if "reasoning_effort" in cfg:
            payload["reasoning"] = {"effort": cfg["reasoning_effort"]}
        self.reserved += reservation
        self.emit("api_reserved", call_id=call_id, model=cfg["id"], usd=str(reservation),
                  input_token_envelope=input_bound, max_output_tokens=max_tokens)
        try:
            response = self.transport(payload)
        except BudgetStop:
            self.reserved -= reservation
            self.emit("api_cancelled", call_id=call_id, usd=str(reservation))
            raise
        except Exception as exc:
            self.blocked = True
            self.emit("api_uncertain", call_id=call_id, error_type=type(exc).__name__)
            raise UncertainCall(
                "Transport failed; dollar reservation retained. Reconcile provider billing."
            ) from None
        try:
            usage = response["usage"]
            cost = Decimal(str(usage["cost"]))
            completion, prompt = usage["completion_tokens"], usage["prompt_tokens"]
            if not cost.is_finite() or cost < 0:
                raise ValueError("Invalid usage.cost")
            if any(type(t) is not int or t < 0 for t in (completion, prompt)):
                raise ValueError("Invalid usage tokens")
        except (KeyError, TypeError, ValueError, ArithmeticError):
            self.blocked = True
            self.emit("api_uncertain", call_id=call_id, error_type="MissingOrInvalidUsage")
            raise UncertainCall("Usage unavailable; full reservation retained") from None
        self.reserved -= reservation
        self.spent += cost
        self.emit(
            "api_settled",
            call_id=call_id,
            usd=str(cost),
            reserved_usd=str(reservation),
            input_tokens=prompt,
            output_tokens=completion,
            generation_id=response.get("id"),
        )
        if cost > reservation or prompt > input_bound or completion > max_tokens:
            self.blocked = True
            self.emit("api_limit_violation", call_id=call_id, output_tokens=completion, input_tokens=prompt)
            raise UncertainCall("Provider exceeded declared limits; actual usage logged; run invalidated")
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, TypeError, IndexError):
            text = ""
        # Null/empty/malformed content still has real usage and must be charged.
        return Generation(text if isinstance(text, str) else "", completion, prompt, str(cost))


class ScriptedDemoProvider:
    """Deterministic protocol exercise. No LLM, no claims of emergence/quality."""

    scripted = True
    spent = Decimal(0)
    reserved = Decimal(0)

    def complete(self, model, messages, max_tokens, call_id):
        obs = json.loads(messages[-1]["content"])
        phase = obs["phase"]
        if phase in {"bank", "review"}:
            reply = {"decisions": []}
            for p in obs["proposals"]:
                revise = phase == "bank" and p["attempts"] == 1
                rejected = phase == "bank" and p["agent"] == "a0006"
                reply["decisions"].append(
                    {
                        "agent": p["agent"],
                        "decision": "reject" if rejected else ("revise" if revise else "approve"),
                        "amount": p["amount"] if not revise else 0,
                        "feedback": "Seek a sponsored position in a funded team."
                        if rejected
                        else (
                            "Include an independent check and stopping rule."
                            if revise
                            else "Return checked evidence within budget."
                        ),
                    }
                )
            if phase == "bank":
                reply["brief"] = "Seek independent arithmetic checks; report failures and stop when verified."
        elif phase == "application":
            reply = {
                "accept": True,
                "proposal": {
                    "plan": "Compute using pairing, check the formula, report evidence, then stop.",
                    "amount": obs["opportunity"]["allowance"] if obs["opportunity"] else 16000,
                },
            }
        elif phase == "formation":
            reply = {}
            if (
                obs["manager"]
                and any(p["team"] is None for p in obs["peers"])
                and not obs["opening"]
                and len(obs["children"]) < 2
                and obs["allowance"] > 6500
            ):
                reply["opening"] = {"allowance": 2000, "assignment": "Independently verify the sum."}
            if obs["parent"] and obs["may_propose"] and obs["round"] == 5:
                reply["proposal"] = {
                    "plan": "Check parity and independently validate 5050, then stop.",
                    "amount": 1000,
                }
        else:
            reply = {
                "note": "Pair 1+100, 2+99, ...: 50 pairs each 101. Sum = 5050.",
                "memory": "Pairing proof yields 5050; check with n(n+1)/2.",
            }
            if obs["depth"] < 2 and not obs["children"] and obs["allowance"] > 3000:
                reply["spawn"] = {
                    "model": obs["worker_models"][0],
                    "strategy": "Check arithmetic independently.",
                    "assignment": "Verify n(n+1)/2 for n=100; report failures to parent.",
                    "allowance": 4000 if obs["depth"] == 0 else 2000,
                }
            elif obs["parent"]:
                reply["messages"] = [
                    {"to": obs["parent"], "text": "Independent formula check: 100*101/2 = 5050."}
                ]
                if obs["round"] >= 3:
                    reply["sleep"] = 3
            if obs["manager"] and obs["round"] == 3:
                reply["publish"] = "Pairing and formula checks agree: 5050."
            if obs["manager"] and obs["round"] >= 7:
                reply["submit"] = "5050"
        text = dumps(reply)
        # Deliberately an estimate; live runs use the provider's completion count.
        tokens = max(1, (len(text.encode("utf-8")) + 3) // 4)
        if tokens > max_tokens:
            text = "{}"
            tokens = min(1, max_tokens)
        return Generation(text, tokens, estimated=True)
