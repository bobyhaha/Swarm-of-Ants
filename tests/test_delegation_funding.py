"""Fund delegation without forcing employment or replacing the owned outcome."""

from dataclasses import replace
import json

import pytest

from ant_colony.config import ColonyConfig
from ant_colony.engine import ColonyMAS
from ant_colony.environment import Task
from ant_colony.providers import OpenRouterProvider
from ant_colony.storage import RunStore, audit
from test_mechanism import engine, funded, Policy
from test_programbench import config, MODELS
from test_providers_and_evaluation import response


def request(c, agent, mode="hire", **changes):
    return {"mode": mode, "model": agent.model, "allowance": 2 * c.call_budget(agent.model),
            "assignment": "Own a separate reproducible regression suite with exact acceptance cases",
            "reason": "Independent fixtures can run in parallel without touching implementation files",
            **changes}


def approve(c, agent, value=None):
    app = agent.application
    c.apply_review(app["target"], [c.proposal_view(agent)], {"decisions": [{
        "agent": agent.name, "decision": "approve", "amount": app["amount"] if value is None else value,
        "feedback": "Fund complementary delegation; preserve the original owned outcome"}]})


def test_observed_work_affordable_hire_unaffordable_gap_gets_scoped_funding():
    p = OpenRouterProvider(MODELS, 15, lambda *a, **k: None, transport=lambda x: response(cost=0.001))
    c = ColonyMAS(config(initial_agents=3, initial_models=("luna",)*3, bank_board_size=0,
                          funding_recovery=True, request_pricing=True), Task("f", "Fixture", "2"), p)
    c.round, c.phase = 6, "formation"
    root, child, applicant = c.agents.values()
    c.grant(root, 1_000_000)
    c.hire(root, child, 37219, "Own core regression quality; subdivide useful cases")
    assert c.funding_request(child) == 0  # The old operating-cash trigger misses this need.
    assert c.hiring_reserve(child, applicant, {"allowance": 30824, "trial": True}) == 42960
    c.request_recruitment(child, request(c, child, candidate=applicant.name))
    assert child.application["target"] == root.name
    quote = child.recruitment_plan
    assert quote["requested_topup"] == quote["required_balance"] - 37219
    assert quote["costs"]["own_working_capital"] == 30824
    before = c.ledger.wealth
    approve(c, child)
    assert child.recruitment_plan["status"] == "funded"
    assert child.assignment == "Own core regression quality; subdivide useful cases"
    c.open_job(child, {"assignment": quote["assignment"], "allowance": 30824,
                       "trial": True, "candidate": applicant.name})
    c.record_proposal(applicant, {"plan": quote["assignment"], "amount": 30824}, child.name, consent=True)
    assert c.accept_trial(child, applicant, {"accept": True})
    assert c.depth(applicant) == 2 and child.recruitment_plan is None
    assert c.ledger.wealth == before and c.calls == 0
    c.check()


@pytest.mark.parametrize("nested", [False, True])
def test_small_shortfall_topup_is_allowed_but_new_worker_minimum_stays_intact(nested):
    c = engine(initial_agents=1, funding_recovery=True)
    root = funded(c, value=19000)
    owner = c.new_agent("worker", "Choose methods", root, 5000, "Own a verifiable outcome") if nested else root
    target = 6 * c.call_budget("worker")  # child 2 + own 2 + screening 1 + coordination 1
    c.charge(owner, owner.team, c.rights(owner) - (target - 1), "fixture")
    original = owner.assignment
    c.request_recruitment(owner, request(c, owner))
    assert owner.application["amount"] == c.proposal_view(owner)["minimum_grant"] == 1
    approve(c, owner)
    assert c.rights(owner) == target and owner.assignment == original
    assert owner.recruitment_plan["status"] == "funded"
    c.check()


def test_operating_topup_preserves_the_child_assignment():
    c = engine(initial_agents=1, funding_recovery=True)
    root = funded(c)
    child = c.new_agent("worker", "Choose methods", root, 2000, "Own the module and its verification")
    c.allocate(root, [{"agent": child.name, "amount": -2000}])
    c.queue_funding_recovery([child])
    approve(c, child)
    assert child.assignment == "Own the module and its verification"


def test_request_is_explicit_single_bounded_and_never_auto_hires():
    c = engine(initial_agents=3, funding_recovery=True)
    root = funded(c)
    c.phase = "formation"
    child, applicant = c.agents["a0002"], c.agents["a0003"]
    c.hire(root, child, 2000, "Own verification")
    c.introduce(applicant, child.name, "bank", "A useful complement")
    c.queue_funding_recovery([child])
    assert child.application is None and child.recruitment_plan is None
    calls, wealth = c.calls, c.ledger.wealth
    c.request_recruitment(child, request(c, child))
    old = dict(child.application)
    with pytest.raises(ValueError, match="pending"):
        c.request_recruitment(child, request(c, child, allowance=3000))
    assert child.application == old and not applicant.team
    assert c.calls == calls and c.ledger.wealth == wealth
    c.request_recruitment(child, None)
    assert child.application is None and child.recruitment_plan is None


def test_rejection_defers_and_revision_keeps_review_authority():
    c = engine(initial_agents=1, funding_recovery=True)
    root = funded(c, value=2000)
    c.request_recruitment(root, request(c, root))
    c.apply_review("bank", [c.proposal_view(root)], {"decisions": [{
        "agent": root.name, "decision": "revise", "feedback": "Identify an independent acceptance case"}]})
    c.request_recruitment(root, request(c, root, reason="Independent coverage of quoted-input failures"))
    assert root.application["attempts"] == 2
    c.apply_review("bank", [c.proposal_view(root)], {"decisions": [{
        "agent": root.name, "decision": "reject", "feedback": "Current work can be completed without a hire"}]})
    assert not root.bank_closed and root.application is None
    assert root.recruitment_plan["status"] == "reject"
    with pytest.raises(ValueError, match="funding_retry_round"):
        c.request_recruitment(root, request(c, root))
    assert not root.children


def test_partial_approval_and_expiry_do_not_fund_automatically():
    c = engine(initial_agents=1, funding_recovery=True)
    root = funded(c, value=2000)
    c.request_recruitment(root, request(c, root))
    approve(c, root, value=1)
    assert root.recruitment_plan["status"] == "partial"
    assert c.rights(root) == 2001 and not root.children
    c.round = root.recruitment_plan["expires"] + 1
    c.prune_recruitment_plans()
    assert root.recruitment_plan is None


def test_spawn_funding_is_an_option_not_an_automatic_population_increase():
    c = engine(initial_agents=1, funding_recovery=True)
    root = funded(c, value=2000)
    c.request_recruitment(root, request(c, root, mode="spawn"))
    assert root.recruitment_plan["costs"]["screening"] == 0
    approve(c, root)
    assert len(c.agents) == 1
    c.new_agent("worker", "Own outcome; choose useful subdivision", root, 1536, "Complementary work")
    assert len(c.agents) == 2 and root.recruitment_plan is None
    c.check()


def test_ordinary_denied_topup_does_not_permanently_block_later_delegation():
    c = engine(initial_agents=1, funding_recovery=True)
    root = funded(c, value=2000)
    c.record_proposal(root, {"plan": "Continue ordinary work", "amount": 2000}, "bank")
    c.apply_review("bank", [c.proposal_view(root)], {"decisions": [{
        "agent": root.name, "decision": "reject", "feedback": "No personal work top-up needed yet"}]})
    assert root.application is None and not root.bank_closed
    with pytest.raises(ValueError, match="funding_retry_round"):
        c.record_proposal(root, {"plan": "Retry now", "amount": 2000}, "bank")
    c.round = root.funding_retry_round
    c.request_recruitment(root, request(c, root))
    assert root.application["recruitment"] and root.application["status"] == "pending"


@pytest.mark.parametrize("change", [{"model": "unknown"}, {"allowance": True}, {"allowance": 1},
                                    {"mode": "mint"}, {"candidate": "absent"}, {"reason": ""}])
def test_invalid_recruitment_request_cannot_create_money_or_pending_review(change):
    c = engine(initial_agents=1, funding_recovery=True)
    root = funded(c, value=2000)
    before = c.ledger.wealth
    with pytest.raises(ValueError):
        c.request_recruitment(root, request(c, root, **change))
    assert root.application is None and root.recruitment_plan is None
    assert c.ledger.wealth == before and c.calls == 0


def test_scope_change_and_membership_end_invalidate_pending_recruitment():
    c = engine(initial_agents=1, funding_recovery=True)
    root = funded(c, value=2000)
    c.request_recruitment(root, request(c, root))
    root.assignment = "Different outcome"
    c.prune_recruitment_plans()
    assert root.application is None and root.recruitment_plan is None
    c.request_recruitment(root, request(c, root))
    c.close_team(c.teams[root.team])
    assert root.application is None and root.recruitment_plan is None


def test_full_degree_and_depth_caps_cannot_be_bypassed_by_funding_request():
    c = engine(initial_agents=1, funding_recovery=True, max_children=1, max_depth=1)
    root = funded(c)
    child = c.new_agent("worker", "Own outcome", root, 2000, "Scoped work")
    for a in (root, child):
        with pytest.raises(ValueError, match="degree/depth"):
            c.request_recruitment(a, request(c, a))
    assert not root.application and not child.application


def test_child_request_propagates_as_bounded_parent_liquidity_need():
    c = engine(initial_agents=1, funding_recovery=True)
    root = funded(c, value=4000)
    child = c.new_agent("worker", "Own outcome", root, 2000, "Separate test outcome")
    assert c.funding_request(root) == 0
    c.request_recruitment(child, request(c, child))
    c.queue_funding_recovery([root])
    assert root.application["target"] == "bank" and root.application["runtime_recovery"]
    assert c.funding_receipt(root)["children"][0]["pending_request"] == child.application["amount"]
    approve(c, root)
    approve(c, child)
    assert child.recruitment_plan["status"] == "funded"
    assert c.calls == 0


@pytest.mark.parametrize("count", [6, 60, 200])
def test_funded_request_keeps_review_and_context_bounded(count):
    c = engine(initial_agents=count, max_agents=count, funding_recovery=True, manager_contacts=3, max_children=3)
    root = funded(c)
    c.phase = "formation"
    for i in range(2, 5):
        a = c.agents[f"a{i:04d}"]
        c.hire(root, a, 2000, "Own distinct tests")
        c.request_recruitment(a, request(c, a, assignment="x"*900, reason="y"*900))
    proposals = [c.proposal_view(c.agents[n]) for n in root.children]
    c.ask(root, "review", extra={"proposals": proposals, "review_target": root.name})
    obs = c.provider.observations[-1]
    assert len(obs["proposals"]) == 3 and len(obs["funding_status"]["children"]) == 3
    assert c.max_context_seen <= c.config.context_bytes
    assert all(len(c.agents[n].recruitment_plan["assignment"].encode()) <= 450 for n in root.children)


def test_full_paid_phase_flow_funds_then_hires_at_depth_two_and_audits(tmp_path):
    def policy(o):
        if o["phase"] == "application":
            return {"proposal": {"plan": "Own a verifiable portfolio", "amount": 20000}} if o["application_target"] == "bank" else {"accept": True}
        if o["phase"] in {"bank", "review"}:
            return {"decisions": [{"agent": p["agent"], "decision": "approve" if p["agent"] in {"a0001", "a0004"} else "reject",
                                   "amount": p["amount"] if p["agent"] in {"a0001", "a0004"} else 0,
                                   "feedback": "Own the outcome, with useful scoped delegation"} for p in o["proposals"]]}
        if o["phase"] == "work" and o["self"] == "a0001" and not o["children"]:
            return {"spawn": {"model": "worker", "strategy": "Choose methods and useful subdivision",
                              "assignment": "Own verification", "allowance": 2000}}
        if o["phase"] == "formation" and o["self"] == "a0004" and not o["children"]:
            if not o["recruitment"]["plan"]:
                return {"recruitment_request": {"mode": "hire", "model": "worker", "allowance": 1536,
                                               "assignment": "Own a complementary fixture", "reason": "Separate cases proceed in parallel",
                                               "candidate": "a0002"}}
            if o["recruitment"]["plan"]["status"] == "funded":
                return {"opening": {"assignment": "Own a complementary fixture", "allowance": 1536,
                                    "trial": True, "candidate": "a0002"}}
        return {}

    provider = Policy(policy)
    provider.scripted = True
    store = RunStore(tmp_path / "delegation")
    c = ColonyMAS(replace(ColonyConfig(), initial_agents=3, max_rounds=5, bank_contacts=1,
                          funding_recovery=True, persistent_teams=True), Task("fixture", "Verify", "2"), provider, store.emit)
    try:
        c.run()
    finally:
        store.close()
    events = [json.loads(line) for line in store.path.read_text().splitlines()]
    assert any(e["event"] == "recruitment_funding_requested" and e["agent"] == "a0004" for e in events)
    assert any(e["event"] == "hire" and e["parent"] == "a0004" and e["child"] == "a0002" for e in events)
    assert audit(store.path)["accounting_valid"]
