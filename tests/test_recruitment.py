"""Recruitment paths that previously stalled at saturated root managers."""

import json

import pytest

from ant_colony.context import byte_size
from ant_colony.protocol import peer_graph
from test_mechanism import Policy, engine, funded


def bank_referral(c, candidate, owner, status="reject"):
    c.record_proposal(candidate, {"plan": "Own quaternion validation and its evidence", "amount": 5000}, "bank")
    c.apply_review("bank", [c.proposal_view(candidate)], {"decisions": [{
        "agent": candidate.name, "decision": status, "amount": 0,
        "feedback": "Useful complementary validation; collaborate with this owner",
        "introduce_to": owner.name}]})


def test_bank_referral_routes_without_free_calls_and_owner_sponsors_exact_trial():
    c = engine(Policy(lambda o: {"accept": True} if o["phase"] == "application" else {}),
               initial_agents=6, funding_recovery=True)
    root = funded(c)
    candidate = c.agents["a0004"]
    for a in c.agents.values():
        a.bank_closed = a is not candidate
    before = c.ledger.wealth
    bank_referral(c, candidate, root)
    assert c.calls == 0 and c.ledger.wealth == before
    assert candidate.referral["target"] == root.name and candidate.name in root.introductions
    assert candidate.bank_closed and not candidate.team
    c.open_job(root, {"candidate": candidate.name, "trial": True,
                      "assignment": "Own quaternion validation and evidence; stop after the trial", "allowance": 4000})
    c.round += 1
    bank = c.ledger.available("bank")
    c.formation()
    assert candidate.parent == root.name and c.rights(candidate) == 4000
    assert candidate.referral is None and not root.introductions and root.opening is None
    assert c.ledger.available("bank") == bank
    assert c.rights(root) == 15000 - 4000 - 10  # owner formation + sponsored acceptance
    assert [o["phase"] for o in c.provider.observations] == ["formation", "application"]
    assert any(e["event"] == "trial_hired" for e in c.events)
    c.check()


def test_saturated_root_forwards_to_child_then_trial_recursively_spawns():
    def policy(o):
        if o["phase"] == "formation" and o["self"] == "a0001" and o["recruitment"]["introductions"]:
            return {"introductions": [{"agent": "a0004", "to": "a0002", "reason": "Own a complementary outcome"}]}
        if o["phase"] == "formation" and o["self"] == "a0002" and o["recruitment"]["introductions"]:
            return {"opening": {"candidate": "a0004", "trial": True, "allowance": 5000,
                                "assignment": "Own independent validation; subdivide if useful"}}
        if o["phase"] == "application":
            return {"accept": True}
        if o["phase"] == "work" and o["self"] == "a0004" and not o["children"]:
            return {"spawn": {"model": "worker", "strategy": "Choose a useful verification method",
                              "assignment": "Independently verify a bounded case", "allowance": 2000}}
        return {}

    c = engine(Policy(policy), initial_agents=4, funding_recovery=True)
    root = funded(c, value=20000)
    child, sibling, applicant = [c.agents[f"a{i:04d}"] for i in (2, 3, 4)]
    c.phase = "formation"
    c.hire(root, child, 10000, "Own validation and choose useful subdivision")
    c.hire(root, sibling, 2000, "Own a different outcome")
    bank_referral(c, applicant, root)
    expires = applicant.referral["expires"]
    c.formation()
    assert applicant.referral["target"] == child.name
    assert applicant.referral["expires"] == expires and len(root.children) == 2
    c.round += 1
    c.formation()
    assert applicant.parent == child.name and c.depth(applicant) == 2
    c.work()
    grandchild = c.agents[applicant.children[0]]
    assert c.depth(grandchild) == 3 and c.rights(grandchild) == 2000
    assert grandchild.born_round == c.round
    assert not any(o.get("self") == grandchild.name for o in c.provider.observations)
    assert all(len(a.children) <= 2 for a in c.agents.values())
    assert not any(o["phase"] == "review" for o in c.provider.observations)
    assert c.ledger.wealth == c.config.initial_bank - c.calls * c.provider.tokens
    c.check()


@pytest.mark.parametrize("count", [6, 60, 200])
def test_indexed_scope_matching_finds_candidates_outside_rotating_peers(count):
    def policy(o):
        return {"accept": True} if o["phase"] == "application" else {}

    c = engine(Policy(policy), initial_agents=count, max_agents=count)
    root = funded(c)
    peers = peer_graph(c.agents, c.round, c.config.peer_contacts)[root.name]
    target = next(a for a in reversed(list(c.agents.values())) if a.name not in peers and a is not root)
    target.trainable_system_prompt = "quaternion validation"
    for a in c.agents.values():
        a.bank_closed = True
    c.open_job(root, {"trial": True, "assignment": "Own quaternion validation", "allowance": 4000})
    c.formation()
    assert target.parent == root.name
    assert c.calls == 2
    assert len([e for e in c.events if e["event"] == "application_sponsored"]) == 1
    assert c.max_context_seen <= c.config.context_bytes
    assert all(len(o.get("recruitment", {}).get("introductions", [])) <= c.config.manager_contacts
               for o in c.provider.observations)
    c.check()


def test_introduction_queue_expiry_and_forwarding_authority_are_bounded():
    c = engine(initial_agents=6, manager_contacts=1)
    root, unrelated = funded(c), funded(c, "a0002")
    candidate, another = c.agents["a0003"], c.agents["a0004"]
    bank_referral(c, candidate, root)
    with pytest.raises(ValueError, match="full"):
        c.introduce(another, root.name, "bank", "More work")
    with pytest.raises(ValueError, match="direct child"):
        c.introduce(candidate, unrelated.name, root.name, "Cross-owner forwarding")
    with pytest.raises(ValueError, match="pending introduction"):
        c.introduce(candidate, unrelated.name, "bank", "Duplicate")
    assert not another.referral and not unrelated.introductions
    c.round = candidate.referral["expires"] + 1
    c.prune_introductions()
    assert candidate.referral is None and not root.introductions and c.calls == 0
    c.introduce(another, root.name, "bank", "A later introduction")
    c.close_team(c.teams[root.team])
    c.prune_introductions()
    assert another.referral is None and not root.introductions


@pytest.mark.parametrize("reply", [{}, {"accept": False}, {"accept": "true"}])
def test_trial_never_assumes_consent(reply):
    c = engine(Policy(lambda o: reply if o["phase"] == "application" else {}), initial_agents=2)
    root = funded(c)
    candidate = c.agents["a0002"]
    candidate.bank_closed = True
    c.open_job(root, {"trial": True, "candidate": candidate.name,
                      "assignment": "Verify one case", "allowance": 2000})
    c.formation()
    assert not candidate.team and not root.children
    assert not any(e["event"] == "trial_hired" for e in c.events)
    c.check()


@pytest.mark.parametrize("proposal", [
    {"plan": "Different scope", "amount": 2000},
    {"plan": "Verify one case", "amount": 1800},
    {"plan": "Verify one case", "amount": 2500},
])
def test_changed_trial_terms_need_a_paid_review(proposal):
    c = engine(Policy(lambda o: {"accept": True, "proposal": proposal} if o["phase"] == "application" else {}), initial_agents=2)
    root = funded(c)
    candidate = c.agents["a0002"]
    candidate.bank_closed = True
    c.open_job(root, {"trial": True, "candidate": candidate.name,
                      "assignment": "Verify one case", "allowance": 2000})
    c.formation()
    assert not candidate.team
    assert any(o["phase"] == "review" for o in c.provider.observations)
    assert not any(e["event"] == "trial_hired" for e in c.events)


def test_hiring_reserves_offer_and_owner_runway_before_screening():
    c = engine(Policy(lambda o: {"accept": True}), initial_agents=2)
    root = funded(c)
    candidate = c.agents["a0002"]
    candidate.bank_closed = True
    c.open_job(root, {"trial": True, "candidate": candidate.name,
                      "assignment": "Verify", "allowance": 2000})
    c.charge(root, root.team, c.rights(root) - 3000, "fixture")
    bank = c.ledger.available("bank")
    c.formation()
    assert not candidate.team
    assert not any(e["event"] == "application_sponsored" for e in c.events)
    assert c.ledger.available("bank") == bank
    assert any(e["event"] == "recruitment_waiting" for e in c.events)
    assert c.calls == 1


def test_delegation_budget_quotes_include_worker_capital_and_coordination():
    c = engine(initial_agents=2)
    root = funded(c)
    costs = c.recruitment_costs(root.model)[0]
    own, child = c.call_budget(root.model), c.call_budget("worker")
    assert costs["spawn_total"] == 2 * own + 2 * child
    assert costs["trial_hire_total"] == costs["spawn_total"] + c.call_budget("worker", "application")
    assert costs["reviewed_hire_total"] == costs["trial_hire_total"] + c.call_budget(root.model, "review")
    messages, _ = c.observation(root, "formation")
    observed = json.loads(messages[-1]["content"])
    assert observed["recruitment"]["budget_options"][0] == costs
    assert byte_size(messages) <= c.config.context_bytes
    assert c.ledger.wealth == c.config.initial_bank


def test_trial_offer_is_one_shot_and_can_be_cancelled():
    c = engine(Policy(lambda o: {"accept": True} if o["phase"] == "application" else {}), initial_agents=3)
    root = funded(c)
    for a in c.agents.values():
        a.bank_closed = True
    c.open_job(root, {"trial": True, "assignment": "Verify", "allowance": 3000})
    c.formation()
    assert len(root.children) == 1 and root.opening is None
    c.round += 1
    c.formation()
    assert len(root.children) == 1
    c.open_job(root, {"trial": True, "assignment": "Another case", "allowance": 3000})
    c.open_job(root, None)
    c.round += 1
    c.formation()
    assert len(root.children) == 1


def test_two_owners_cannot_screen_or_hire_the_same_candidate_twice():
    c = engine(Policy(lambda o: {"accept": True} if o["phase"] == "application" else {}), initial_agents=3)
    first, second = funded(c), funded(c, "a0002")
    candidate = c.agents["a0003"]
    candidate.bank_closed = True
    for owner in (first, second):
        c.open_job(owner, {"trial": True, "candidate": candidate.name,
                           "assignment": "Verify a small case", "allowance": 2000})
    c.formation()
    assert sum(len(a.children) for a in (first, second)) == 1
    assert sum(o["phase"] == "application" for o in c.provider.observations) == 1
    assert c.rights(candidate) == 2000
    c.check()


def test_approve_cannot_also_refer_or_mint_introduction_money():
    c = engine(initial_agents=3)
    owner = funded(c)
    candidate = c.agents["a0002"]
    c.record_proposal(candidate, {"plan": "Check", "amount": 5000}, "bank")
    before = c.ledger.wealth
    c.apply_review("bank", [c.proposal_view(candidate)], {"decisions": [{
        "agent": candidate.name, "decision": "approve", "amount": 5000,
        "feedback": "Try both", "introduce_to": owner.name}]})
    assert not candidate.team and not candidate.referral and not owner.introductions
    assert c.calls == 0 and c.ledger.wealth == before
    assert candidate.application["status"] == "pending"


def test_scripted_full_run_recruits_recursively_and_passes_financial_audit(tmp_path):
    from dataclasses import replace
    from ant_colony.config import ColonyConfig
    from ant_colony.engine import ColonyMAS
    from ant_colony.environment import Task
    from ant_colony.storage import RunStore, audit

    def choose(o):
        if o["phase"] == "application":
            if o["application_target"] == "bank":
                return {"proposal": {"plan": "Own a verifiable outcome", "amount": 18000}, "accept": True}
            return {"accept": True}
        if o["phase"] == "bank":
            return {"decisions": [
                {"agent": p["agent"], "decision": "approve", "amount": p["amount"], "feedback": "Own the outcome and subdivide if useful"}
                if p["agent"] == "a0001" else
                {"agent": p["agent"], "decision": "reject", "amount": 0, "introduce_to": "a0001",
                 "feedback": "Complement the existing outcome inside this owner's team"}
                for p in o["proposals"]]}
        if o["phase"] == "formation" and o.get("recruitment", {}).get("introductions"):
            name = o["recruitment"]["introductions"][0]["agent"]
            if o["children"]:
                return {"introductions": [{"agent": name, "to": o["children"][0], "reason": "Complementary subdivision"}]}
            return {"opening": {"candidate": name, "trial": True,
                                "allowance": 6000 if o["depth"] == 0 else 3000,
                                "assignment": "Own a bounded outcome and preserve verification evidence"}}
        return {}

    policy = Policy(choose)
    policy.scripted = True
    store = RunStore(tmp_path / "recruitment-smoke")
    c = ColonyMAS(replace(ColonyConfig(), initial_agents=4, max_agents=8, max_rounds=8,
                          bank_contacts=1, funding_recovery=True, persistent_teams=True),
                  Task("scripted-recruitment", "Exercise the recruitment protocol", "fixture"), policy, store.emit)
    try:
        c.run()
    finally:
        store.close()
    events = [json.loads(line) for line in store.path.read_text().splitlines()]
    assert sum(e["event"] == "trial_hired" for e in events) >= 2
    assert any(e["event"] == "introduction_routed" and e["source"] != "bank" for e in events)
    snapshots = [e["state"] for e in events if e["event"] == "snapshot"]
    assert any(any(a["parent"] and next(p for p in s["agents"] if p["id"] == a["parent"])["parent"]
                   for a in s["agents"]) for s in snapshots)
    assert audit(store.path)["accounting_valid"]
