"""Regression tests for affordable requests, conditional renewals, and stopping."""

from dataclasses import replace
from decimal import Decimal
import json

import pytest

from ant_colony.engine import ColonyMAS
from ant_colony.environment import Task
from ant_colony.providers import OpenRouterProvider, UncertainCall
from ant_colony.reservations import input_envelope
from test_programbench import config, MODELS, fixture
from test_providers_and_evaluation import response
from test_mechanism import engine, funded, Policy


def test_request_reservation_matches_credit_budget_and_http_envelope():
    events, sent = [], []
    p = OpenRouterProvider(MODELS, 4, lambda e, **kw: events.append({"event": e, **kw}),
                           transport=lambda x: sent.append(x) or response(cost=0.001))
    c = ColonyMAS(config(request_pricing=True), Task("f", "Compute 2", "2"), p)
    messages = [{"role": "system", "content": "Return JSON"},
                {"role": "user", "content": '{"phase":"bank","text":"雪🧪"}'}]
    bound, dollars = p.reservation("astra", messages, c.output_limit("bank"))
    assert c.call_budget("astra", "bank", messages) == c.dollar_credits(dollars)
    assert dollars < Decimal("0.30")
    assert bound >= len(json.dumps(messages, ensure_ascii=False).encode())
    c.call("astra", messages, kind="bank")
    assert sent[0]["max_tokens"] == 4096
    assert sent[0]["reasoning"] == {"effort": "high"}
    assert events[0]["input_token_envelope"] == bound
    assert p.spent == Decimal("0.001") and p.reserved == 0
    # General work retains its full reasoning/output allowance.
    assert c.output_limit("work") == 16384


def test_envelope_falls_back_for_unknown_models_and_nontext_messages():
    cfg = {**MODELS["astra"], "id": "unknown/model"}
    assert input_envelope(cfg, enabled=True, context_bytes=100) == 100000
    assert input_envelope(MODELS["astra"], enabled=True,
                          messages=[{"role": "user", "content": [{"type": "image"}]}]) == 100000


def test_request_usage_breach_stops_after_recording_real_receipt():
    r = response(cost=0.01)
    r["usage"]["prompt_tokens"] = 50000
    p = OpenRouterProvider(MODELS, 4, lambda *a, **kw: None,
                           request_pricing=True, transport=lambda x: r)
    with pytest.raises(UncertainCall, match="declared limits"):
        p.complete("astra", [{"role": "user", "content": "hello"}], 4096, "c1")
    assert p.blocked and p.spent == Decimal("0.01")


def test_actual_request_can_run_below_the_old_full_context_reservation():
    p = OpenRouterProvider(MODELS, 4, lambda *a, **kw: None,
                           transport=lambda x: response(cost=0.001))
    c = ColonyMAS(config(request_pricing=True, initial_bank=4_000_000, bank_board_size=0), Task("f", "Compute 2", "2"), p)
    a = next(a for a in c.agents.values() if a.model == "sol")
    c.grant(a, 250000)
    assert c.rights(a) < c.call_budget("sol")
    c.ask(a, "work")
    assert p.spent > 0 and c.rights(a) == 249000


def test_standing_grant_needs_new_progress_expiry_and_real_parent_money():
    c = engine(Policy(), initial_agents=1, standing_grants=True)
    parent = funded(c)
    child = c.new_agent("worker", "verify", parent, 2000, "Check module")
    c.authorize_renewal(child, parent.name,
                        c.renewal_spec({"total": 4000, "tranche": 2000, "rounds": 2}, 6000, 2000, child))
    c.allocate(parent, [{"agent": child.name, "amount": -2000}])
    wealth, calls, parent_before = c.ledger.wealth, c.calls, c.rights(parent)
    c.standing_renewals([parent, child])
    assert c.rights(child) == 0
    child.productive_steps += 1
    c.standing_renewals([parent, child])
    assert c.rights(child) == 2000 and c.rights(parent) == parent_before - 2000
    assert c.calls == calls and c.ledger.wealth == wealth
    c.allocate(parent, [{"agent": child.name, "amount": -2000}])
    c.standing_renewals([parent, child])
    assert c.rights(child) == 0  # Same progress cannot redeem another tranche.
    child.productive_steps += 1
    c.round += 3
    c.standing_renewals([parent, child])
    assert c.rights(child) == 0 and child.standing_grant is None
    c.check()


def test_same_contract_renews_an_intermediate_manager_from_child_evidence():
    c = engine(Policy(), initial_agents=1, standing_grants=True)
    root = funded(c)
    child = c.new_agent("worker", "manage", root, 6000, "Module")
    leaf = c.new_agent("worker", "test", child, 2000, "Verify")
    c.authorize_renewal(child, root.name, {"remaining": 4000, "tranche": 2000, "expires": 5})
    c.allocate(root, [{"agent": child.name, "amount": -4000}])
    leaf.productive_steps = 1
    c.standing_renewals([root, child, leaf])
    assert c.rights(child) == 2000
    assert child.subtree_progress == 1
    c.check()


def test_renewal_cannot_expand_a_proposal_or_survive_scope_change():
    c = engine(Policy(), initial_agents=1, standing_grants=True)
    a = funded(c)
    with pytest.raises(ValueError, match="remaining requested"):
        c.renewal_spec({"total": 6000, "tranche": 2000, "rounds": 2}, 5000, 2000, a)
    c.authorize_renewal(a, "bank", {"remaining": 2000, "tranche": 2000, "expires": 5})
    a.assignment = "Different task"
    c.standing_renewals([a])
    assert a.standing_grant is None


def test_rejected_renewal_defers_without_permanently_closing_funding():
    c = engine(Policy(), initial_agents=1, standing_grants=True, funding_recovery=True)
    a = funded(c)
    c.record_proposal(a, {"amount": 4000, "plan": "Continue"}, "bank")
    a.application["runtime_recovery"] = True
    c.apply_review("bank", [c.proposal_view(a)], {"decisions": [{"agent": a.name,
                   "decision": "reject", "feedback": "Wait for a milestone"}]})
    assert not a.bank_closed and a.application is None
    assert a.funding_retry_round == c.round + c.config.bank_review_interval
    c.queue_funding_recovery([a])
    assert a.application is None


def test_bank_reviews_are_event_driven_with_a_periodic_backstop():
    c = engine(Policy(), initial_agents=1, event_driven_bank=True)
    c.last_bank_review = 1
    c.round = 2
    assert not c.bank_review_due([], False)
    assert c.bank_review_due([{"agent": "a0001"}], False)
    assert c.bank_review_due([], True)
    c.round = 5
    assert c.bank_review_due([], False)


def test_stagnation_gets_review_and_grace_then_selects_public_artifact(tmp_path):
    bench, _ = fixture(tmp_path)
    cfg = replace(config(), initial_agents=1, initial_models=("luna",), progress_control=True,
                  stagnation_rounds=3, bank_review_interval=2)
    c = ColonyMAS(cfg, Task("f", "Rebuild", "native tests", "programbench"), Policy(), workbench=bench)
    c.round = 3
    c.assess_progress()
    assert c.force_bank_review and c.final_answer is None
    c.round = 4
    c.assess_progress()
    assert c.final_answer is None
    c.round = 5
    c.assess_progress()
    assert c.final_author == "__public__"
    assert c.submission_mode == "automatic_stagnation_checkpoint"
