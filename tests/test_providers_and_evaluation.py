from dataclasses import replace
from decimal import Decimal
import json

import pytest

from ant_colony.config import ColonyConfig
from ant_colony.engine import ColonyMAS
from ant_colony.environment import Task, ExactEnv, research_env, evaluate
from ant_colony.providers import OpenRouterProvider, Generation, BudgetStop, UncertainCall
from ant_colony.storage import RunStore, audit


MODELS = {
    "bank": {
        "id": "example/strong",
        "input_usd_per_million": 1,
        "output_usd_per_million": 2,
        "input_token_ceiling": 1000,
    },
    "worker": {
        "id": "example/small",
        "input_usd_per_million": 1,
        "output_usd_per_million": 2,
        "input_token_ceiling": 1000,
    },
}


def response(text="{}", tokens=12, cost=0.000124):
    return {
        "id": "mock-generation",
        "choices": [{"message": {"content": text}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": tokens,
            "cost": cost,
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
    }


def client(transport, limit="1"):
    events = []
    provider = OpenRouterProvider(
        MODELS, limit, lambda event, **data: events.append({"event": event, **data}), transport=transport
    )
    return provider, events


def test_provider_reserves_before_transport_and_settles_usage_once():
    def transport(payload):
        assert events[-1]["event"] == "api_reserved"
        assert provider.reserved > 0
        assert payload["max_tokens"] == 100
        assert payload["provider"]["allow_fallbacks"] is False
        assert payload["provider"]["max_price"] == {"prompt": 1.0, "completion": 2.0, "request": 0}
        return response()

    provider, events = client(transport)
    result = provider.complete("worker", [], 100, "c1")
    assert result.output_tokens == 12  # reasoning_tokens is already a subset; no double count
    assert provider.spent == Decimal("0.000124") and provider.reserved == 0
    assert events[-1]["event"] == "api_settled"


def test_no_io_when_next_call_cannot_be_reserved():
    def forbidden(payload):
        pytest.fail("Budget stop must happen before transport")

    provider, events = client(forbidden, limit="0.00001")
    with pytest.raises(BudgetStop):
        provider.complete("worker", [], 100, "c1")
    assert not events and provider.exposure == 0


def test_unknown_response_keeps_usd_reservation_and_never_retries():
    calls = []

    def transport(payload):
        calls.append(payload)
        raise TimeoutError("possibly billed")

    provider, events = client(transport)
    with pytest.raises(UncertainCall):
        provider.complete("worker", [], 100, "c1")
    assert provider.reserved == Decimal("0.0012")
    with pytest.raises(BudgetStop):
        provider.complete("worker", [], 100, "c2")
    assert len(calls) == 1
    assert events[-1]["event"] == "api_uncertain"


@pytest.mark.parametrize("bad", [{}, {"usage": {}}, response(cost=float("nan")), response(tokens=True)])
def test_missing_or_invalid_usage_fails_closed(bad):
    provider, _ = client(lambda payload: bad)
    with pytest.raises(UncertainCall):
        provider.complete("worker", [], 100, "c")
    assert provider.blocked and provider.reserved > 0


@pytest.mark.parametrize("bad", [response(tokens=101), response(cost=5)])
def test_provider_limit_breach_records_known_cost_and_stops(bad):
    provider, events = client(lambda payload: bad)
    with pytest.raises(UncertainCall):
        provider.complete("worker", [], 100, "c")
    assert provider.spent == Decimal(str(bad["usage"]["cost"]))
    assert events[-1]["event"] == "api_limit_violation"
    assert provider.blocked


def test_null_content_still_charges_actual_tokens():
    provider, _ = client(lambda payload: response(None))
    result = provider.complete("worker", [], 100, "c")
    assert result.text == "" and result.output_tokens == 12 and provider.spent > 0


def test_missing_key_is_a_free_preflight_stop(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    provider = OpenRouterProvider(MODELS, 1, lambda *a, **k: None)
    with pytest.raises(BudgetStop):
        provider.complete("worker", [], 100, "c")
    assert provider.exposure == 0


def test_live_transport_mock_integrates_with_credit_audit(tmp_path):
    store = RunStore(tmp_path / "live-mock")

    def transport(payload):
        obs = json.loads(payload["messages"][-1]["content"])
        phase = obs["phase"]
        result = {}
        if phase == "application":
            result = {"proposal": {"plan": "Compute and answer 2", "amount": 4000}}
        elif phase == "bank":
            result = {
                "decisions": [
                    {"agent": p["agent"], "amount": p["amount"], "decision": "approve", "feedback": "Proceed"}
                    for p in obs["proposals"]
                ]
            }
        elif phase == "work":
            result = {"submit": "2"}
        return response(json.dumps(result))

    provider = OpenRouterProvider(MODELS, 1, store.emit, transport=transport)
    colony = ColonyMAS(
        replace(ColonyConfig(), initial_agents=2), Task("test", "1+1?", "2"), provider, store.emit
    )
    report = colony.run()
    store.close()
    assert report["status"] == "completed"
    assert report["burned"] == report["calls"] * 12
    assert report["estimated_tokens"] == 0
    assert Decimal(report["api_spent_usd"]) > 0
    assert audit(store.path)["accounting_valid"]


@pytest.mark.parametrize(
    "score", ["garbage", "SCORE: nan", "SCORE: 2", "SCORE: -0.1", "SCORE: 0.5\nSCORE: 1"]
)
def test_original_judge_error_is_not_silently_converted_to_zero(score):
    env = research_env(Task("t", "Prove the fact.", "SECRET_RUBRIC", "research"), lambda prompt: score)
    with pytest.raises(ValueError):
        evaluate(env, "a candidate", "a1")


def test_research_task_uses_eom_rubric_judge_only_once_and_protects_reference():
    class JudgeProvider:
        scripted = False
        spent = reserved = Decimal(0)
        observations = []

        def complete(self, model, messages, max_tokens, call_id):
            obs = json.loads(messages[-1]["content"])
            self.observations.append(obs)
            if "evaluation" in obs:
                return Generation("SCORE: 0.5\nREASON: partial proof", 9)
            if obs["phase"] == "application":
                result = {"proposal": {"plan": "prove the fact", "amount": 6000}}
            elif obs["phase"] == "bank":
                result = {
                    "decisions": [
                        {
                            "agent": p["agent"],
                            "amount": p["amount"],
                            "decision": "approve",
                            "feedback": "Proceed",
                        }
                        for p in obs["proposals"]
                    ]
                }
            elif obs["phase"] == "work":
                result = {"submit": "My partial proof."}
            else:
                result = {}
            return Generation(json.dumps(result), 5)

    provider = JudgeProvider()
    colony = ColonyMAS(
        replace(ColonyConfig(), initial_agents=1),
        Task("t", "Prove the fact.", "SECRET_RUBRIC_NEEDLE", "research"),
        provider,
    )
    report = colony.run()
    judging = [o for o in provider.observations if "evaluation" in o]
    assert len(judging) == 1 and "SECRET_RUBRIC_NEEDLE" in judging[0]["evaluation"]
    assert not any(
        "SECRET_RUBRIC_NEEDLE" in json.dumps(o) for o in provider.observations if "evaluation" not in o
    )
    assert report["rewarded"] == 25000 and report["burned"] == 24


def test_exact_environment_numeric_edge_cases():
    from ant_colony.state import ColonyAction

    env = ExactEnv(Task("t", "1+1?", "2"))
    assert env.apply(ColonyAction("2.000", "a")) == 1
    with pytest.raises(RuntimeError):
        env.apply(ColonyAction("2", "a"))
    for text in ("nan", "inf", "2.000001", "True"):
        assert ExactEnv(Task("t", "1+1?", "2")).apply(ColonyAction(text, "a")) == 0


def test_invalid_task_and_oversized_rubric_are_rejected_before_calls():
    with pytest.raises(ValueError):
        Task.from_dict({"id": "x", "problem": "x", "answer": float("nan")})
    provider, events = client(lambda payload: pytest.fail("No paid calls before task validation"))
    with pytest.raises(ValueError, match="Required context"):
        ColonyMAS(ColonyConfig(), Task("t", "Prove x.", "x" * 25000, "research"), provider)
    assert not events


def test_example_models_and_embedded_secrets_are_rejected():
    for cfg in (
        {**MODELS["worker"], "api_key": "forbidden"},
        {**MODELS["worker"], "id": "REPLACE_WITH_MODEL_ID"},
    ):
        with pytest.raises(ValueError):
            OpenRouterProvider({"worker": cfg}, 1, lambda *a, **k: None)
