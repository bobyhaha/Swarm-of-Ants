"""Regression cases observed in the first mixed ProgramBench study."""

from dataclasses import replace
import json

import pytest

from ant_colony.config import ColonyConfig
from ant_colony.engine import ColonyMAS
from ant_colony.environment import Task
from test_mechanism import engine, funded, Policy
from test_sponsorship import decide
from test_programbench import fixture


def review_policy(obs):
    return decide(obs["proposals"]) if obs["phase"] in {"bank", "review"} else {}


def test_accepting_existing_offer_does_not_require_duplicate_proposal():
    def policy(obs):
        if obs["phase"] == "application":
            return {"accept": True}
        return review_policy(obs)
    c = engine(Policy(policy), initial_agents=2, funding_recovery=True)
    parent = funded(c)
    child = c.agents["a0002"]
    child.bank_closed = True
    c.open_job(parent, {"assignment": "Run isolated fixture checks", "allowance": 3000})
    c.formation()
    assert child.parent == parent.name and c.rights(child) == 3000
    assert not any(e["event"] == "application_declined" for e in c.events)
    c.check()


def test_missing_acceptance_is_incomplete_not_a_permanent_refusal():
    c = engine(Policy(), initial_agents=2)
    parent = funded(c)
    child = c.agents["a0002"]
    child.bank_closed = True
    c.open_job(parent, {"assignment": "Verify", "allowance": 3000})
    c.formation()
    assert child.application is None and not child.parent
    assert any(e["event"] == "application_incomplete" for e in c.events)
    assert not any(e["event"] == "application_declined" for e in c.events)


def test_zero_balance_child_can_request_renewal_without_a_model_call():
    c = engine(Policy(review_policy), initial_agents=1, funding_recovery=True)
    parent = funded(c)
    child = c.new_agent("worker", "verify", parent, 2000, "Run fixture checks")
    c.allocate(parent, [{"agent": child.name, "amount": -2000}])
    c.round = 2
    c.formation()
    assert c.rights(child) >= c.call_budget(child.model) * 2
    assert not any(o.get("self") == child.name for o in c.provider.observations)
    assert any(e["event"] == "funding_needed" and e["agent"] == child.name for e in c.events)
    c.work()
    assert any(o.get("self") == child.name and o["phase"] == "work" for o in c.provider.observations)
    c.check()


def test_funding_recovery_propagates_through_two_manager_levels():
    c = engine(Policy(review_policy), initial_agents=1, funding_recovery=True)
    root = funded(c)
    child = c.new_agent("worker", "manage", root, 7000, "Verify module")
    grandchild = c.new_agent("worker", "verify", child, 2000, "Run checks")
    c.allocate(child, [{"agent": grandchild.name, "amount": -2000}])
    c.allocate(root, [{"agent": child.name, "amount": -c.rights(child)}])
    for r in (2, 3, 4):
        c.round = r
        c.fund(c.formation())
    assert c.rights(child) >= c.call_budget("worker")
    assert c.rights(grandchild) >= c.call_budget("worker")
    assert all(len(o.get("proposals", [])) <= 2 for o in c.provider.observations)
    assert all(len(o.get("funding_status", {}).get("children", [])) <= 2
               for o in c.provider.observations if isinstance(o.get("funding_status"), dict))
    c.check()


def test_recovery_gets_bounded_priority_even_with_thousand_unfunded_agents():
    c = engine(Policy(review_policy), initial_agents=1000, max_agents=1000,
               funding_recovery=True, application_budget=0)
    root = funded(c, "a1000", 1000)
    c.charge(root, root.team, 999, "fixture")
    proposals = c.formation()
    assert {p["agent"] for p in proposals} == {root.name}
    assert c.calls == 0
    c.fund(proposals)
    assert c.rights(root) > c.call_budget("worker")
    bank = c.provider.observations[-1]
    assert len(bank["funding_status"]) <= 2
    assert bank["funding_status"][0]["can_afford_call"] is False


def test_recovery_rejects_grants_that_fund_only_one_call():
    c = engine(initial_agents=1, funding_recovery=True)
    a = c.agents["a0001"]
    c.record_proposal(a, {"plan": "check", "amount": 5000}, "bank")
    c.apply_review("bank", [c.proposal_view(a)], {"decisions": [{"agent": a.name,
                   "decision": "approve", "amount": c.call_budget(a.model), "feedback": "Check"}]})
    assert a.team is None
    assert any(e["event"] == "review_rejected" for e in c.events)


def test_grants_preserve_the_banks_next_call_and_operating_cash():
    c = engine(initial_agents=1, funding_recovery=True, initial_bank=10000)
    agent = c.agents["a0001"]
    with pytest.raises(ValueError, match="treasury"):
        c.grant(agent, c.bank_spendable())
    c.grant(agent, c.bank_grantable())
    assert c.bank_spendable() > c.call_budget(c.config.bank_model)
    c.check()


def test_bank_approval_is_limited_to_post_generation_operating_envelope():
    c = engine(Policy(review_policy), initial_agents=1, funding_recovery=True, initial_bank=10000)
    agent = c.agents["a0001"]
    c.record_proposal(agent, {"plan": "Verify", "amount": 10000}, "bank")
    c.fund([c.proposal_view(agent)])
    assert agent.team and c.rights(agent) < 10000
    assert c.bank_spendable() > c.call_budget(c.config.bank_model)
    grant = next(e for e in c.events if e["event"] == "grant")
    review = next(e for e in c.events if e["event"] == "proposal_review")
    assert grant["amount"] == review["amount"]
    c.check()


def test_persistent_team_does_not_disband_at_old_grant_deadline():
    c = engine(initial_agents=1, max_rounds=3, grant_rounds=1, persistent_teams=True)
    funded(c)
    c.run()
    closed = [e for e in c.events if e["event"] == "team_closed"]
    assert len(closed) == 1 and closed[0]["phase"] == "settle"


def test_team_ready_cannot_end_society_or_predict_unseen_tool_results(tmp_path):
    bench, _ = fixture(tmp_path)
    c = ColonyMAS(replace(ColonyConfig(), initial_agents=2, bank_board_size=2, bank_finalization=True),
                  Task("f", "fixture", "tests", "programbench"), Policy(), workbench=bench)
    c.round = 1
    c.board.replace([{"id": name, "title": name, "scope": name, "acceptance": "Check"}
                     for name in ("first", "second")])
    first, second = c.agents.values()
    c.grant(first, 10000, "first")
    c.grant(second, 10000, "second")
    bench.public_check = {"status": "build_only"}
    with pytest.raises(ValueError, match="prior turn"):
        c.mark_ready(first, {"ready": "Claim", "tool": {"kind": "test"}})
    c.mark_ready(first, {"submit": "Old protocol completion becomes team readiness"})
    c.finalize_society("Ready")
    assert c.final_answer is None
    c.mark_ready(second, {"ready": "Second scope checked"})
    bench.progress[second.team]["revision"] += 1
    c.finalize_society("Stale readiness")
    assert c.final_answer is None
    c.mark_ready(second, {"ready": "Latest revision checked"})
    c.finalize_society("Both outcomes checked against integrated build")
    assert c.final_author == "__public__" and c.submission_mode == "bank_approved_public"


def test_shell_receipts_reach_bank_without_claiming_all_commands_are_tests(tmp_path):
    bench, agent = fixture(tmp_path)
    for status in (0, 1, 0):
        bench.remember_tool(agent, {"kind": "exec", "command": "python3 tests_fixture.py"},
                            {"status": "ok", "exit_code": status, "output": "fixture output"})
    progress = bench.bank_evidence([agent.team])["applicants"][0]["progress"]
    assert progress["reads"] == progress["tests"] == 0
    assert progress["executions"] == 3
    assert len(progress["recent_executions"]) == 2
    assert [r["exit_code"] for r in progress["recent_executions"]] == [1, 0]
    assert "python3 tests_fixture.py" in json.dumps(progress)
