"""Economic and admission regressions for sponsored recursive review."""

import json
import pytest
from test_mechanism import engine, funded, Policy


def propose(obs):
    if obs["phase"] == "application":
        return {"proposal": {"plan": "Deliver a checked answer, then stop", "amount": 6000}, "accept": True}
    return {}


def decide(proposals, status="approve"):
    return {
        "decisions": [
            {
                "agent": p["agent"],
                "decision": status,
                "amount": p["amount"],
                "feedback": "Show an independent check and stop when verified.",
            }
            for p in proposals
        ]
    }


def test_unfunded_agents_only_generate_in_admitted_bank_slots():
    colony = engine(Policy(propose), initial_agents=1000, max_agents=1000)
    proposals = colony.formation()
    assert len(proposals) == colony.calls == 2
    assert {o["phase"] for o in colony.provider.observations} == {"application"}
    assert all(a.wealth == 0 and colony.rights(a) == 0 for a in colony.agents.values())
    assert colony.application_spent == colony.ledger.burned == 10


def test_unfunded_cannot_use_application_to_send_messages_or_open_jobs():
    colony = engine(Policy(lambda o: {"messages": [{"to": "a0003", "text": "free spam"}]}))
    colony.formation()
    assert not any(e["event"] == "message" for e in colony.events)
    assert colony.ledger.burned == 10  # Invalid output is still charged.


def test_application_pool_exhaustion_waits_without_polling():
    colony = engine(Policy(propose, tokens=768), application_budget=768, max_rounds=4)
    colony.run()
    assert sum(o["phase"] == "application" for o in colony.provider.observations) == 1
    assert not any(o["phase"] == "formation" for o in colony.provider.observations)
    assert colony.application_spent == 768


def test_malformed_applications_have_a_finite_retry_allowance():
    colony = engine(Policy(lambda o: "not json"), initial_agents=1, max_rounds=12)
    colony.run()
    assert colony.calls == 1 + colony.config.max_revisions
    assert colony.agents["a0001"].bank_closed


def test_feedback_reaches_next_revision_and_context_does_not_accumulate_history():
    def policy(o):
        return decide(o["proposals"], "revise") if o["phase"] == "bank" else propose(o)

    colony = engine(Policy(policy), initial_agents=1, max_rounds=10)
    colony.run()
    applications = [o for o in colony.provider.observations if o["phase"] == "application"]
    assert len(applications) == 3
    assert "independent check" in applications[1]["application"]["feedback"]
    assert applications[2]["application"]["attempts"] == 2
    reviews = [e for e in colony.events if e["event"] == "proposal_review"]
    assert [e["decision"] for e in reviews] == ["revise", "revise", "reject"]


def test_bank_rejection_does_not_retry_or_prevent_sponsored_employment():
    def policy(o):
        if o["phase"] == "formation" and o["self"] == "a0001":
            return {"opening": {"assignment": "check", "allowance": 6000}}
        if o["phase"] == "review":
            return decide(o["proposals"])
        return propose(o)

    colony = engine(Policy(policy), initial_agents=2)
    root = funded(colony)
    child = colony.agents["a0002"]
    colony.record_proposal(child, {"plan": "try", "amount": 6000}, "bank")
    colony.apply_review(
        "bank", [colony.proposal_view(child)], decide([colony.proposal_view(child)], "reject")
    )
    bank = colony.ledger.available("bank")
    colony.formation()
    assert child.parent == root.name
    assert colony.ledger.available("bank") == bank
    assert colony.rights(child) == 6000
    assert colony.rights(root) == 15000 - 6000 - 15  # formation + application + review
    sponsored = [e for e in colony.events if e["event"] == "application_sponsored"]
    assert sponsored[-1]["target"] == root.name
    colony.check()


def test_employer_can_request_revision_before_consent_based_hire():
    def policy(o):
        if o["phase"] == "formation" and o["self"] == "a0001" and not o["opening"]:
            return {"opening": {"assignment": "check", "allowance": 6000}}
        if o["phase"] == "review":
            return decide(o["proposals"], "revise" if o["proposals"][0]["attempts"] == 1 else "approve")
        return propose(o)

    colony = engine(Policy(policy), initial_agents=2)
    root = funded(colony)
    child = colony.agents["a0002"]
    child.bank_closed = True
    colony.formation()
    assert not child.team and child.application["status"] == "revise"
    colony.round += 1
    colony.formation()
    assert child.parent == root.name
    apps = [o for o in colony.provider.observations if o["phase"] == "application"]
    assert len(apps) == 2 and apps[1]["application"]["attempts"] == 1


def test_child_and_grandchild_request_budget_from_local_parent_only():
    def policy(o):
        if o["phase"] == "formation" and o["parent"] and o["may_propose"]:
            return {"proposal": {"plan": "Independent verification", "amount": 1000}}
        if o["phase"] == "review":
            return decide(o["proposals"])
        return {}

    colony = engine(Policy(policy), initial_agents=1)
    root = funded(colony)
    child = colony.new_agent("worker", "verify", root, 6000, "subtask")
    grandchild = colony.new_agent("worker", "verify", child, 2000, "subtask")
    colony.round = 2
    bank = colony.ledger.available("bank")
    assert colony.formation() == []
    assert colony.ledger.available("bank") == bank
    reviews = [e for e in colony.events if e["event"] == "proposal_review"]
    assert {(e["reviewer"], e["agent"]) for e in reviews} == {
        (root.name, child.name),
        (child.name, grandchild.name),
    }
    colony.check()


def test_zero_allowance_child_revision_is_sponsored_by_its_parent():
    colony = engine(Policy(propose), initial_agents=1)
    root = funded(colony)
    child = colony.new_agent("worker", "verify", root, 1000, "subtask")
    colony.record_proposal(child, {"plan": "check", "amount": 2000}, root.name)
    colony.apply_review(
        root.name, [colony.proposal_view(child)], decide([colony.proposal_view(child)], "revise")
    )
    colony.allocate(root, [{"agent": child.name, "amount": -1000}])
    colony.round = 2
    bank = colony.ledger.available("bank")
    colony.formation()
    assert child.application["status"] == "pending" and child.application["attempts"] == 2
    assert colony.ledger.available("bank") == bank
    assert colony.rights(child) == 0


def test_insufficient_employer_budget_never_falls_back_to_bank():
    colony = engine(Policy(), initial_agents=2)
    root = funded(colony, value=1000)
    child = colony.agents["a0002"]
    colony.charge(root, root.team, 900, "test")
    assert colony.sponsored_application(child, root.name) == {}
    assert colony.calls == 0
    assert colony.ask(root, "work") == {}
    assert colony.calls == 0


def test_failed_sponsored_bank_call_consumes_application_pool():
    from ant_colony.providers import UncertainCall

    class Failed(Policy):
        def complete(self, *args):
            raise UncertainCall("possibly billed")

    colony = engine(Failed(), application_budget=768)
    with pytest.raises(UncertainCall):
        colony.formation()
    assert colony.application_spent == colony.uncertain_credits == 768


def test_applicant_refusal_is_not_hired_and_cannot_spam_sponsor():
    def policy(o):
        if o["phase"] == "formation" and o["self"] == "a0001":
            return {"opening": {"assignment": "check", "allowance": 2000}}
        if o["phase"] == "application":
            return {"accept": False}
        return {}

    colony = engine(Policy(policy), initial_agents=2)
    root = funded(colony)
    child = colony.agents["a0002"]
    child.bank_closed = True
    for r in range(1, 4):
        colony.round = r
        colony.formation()
    assert not root.children and child.team is None
    assert sum(o["phase"] == "application" for o in colony.provider.observations) == 1


def test_bank_brief_is_bounded_and_visible_without_broadcast_calls():
    colony = engine(Policy(lambda o: {"brief": "priority" * 1000}), initial_agents=1)
    agent = colony.agents["a0001"]
    colony.record_proposal(agent, {"plan": "check", "amount": 2000}, "bank")
    colony.fund([colony.proposal_view(agent)])
    assert len(colony.bank_brief.encode()) <= colony.config.text_bytes
    messages, _ = colony.observation(agent, "application")
    assert json.loads(messages[-1]["content"])["bank_brief"] == colony.bank_brief


@pytest.mark.parametrize("bank_status", ["pending", "revise"])
def test_unfunded_agent_can_choose_employment_instead_of_bank_revision(bank_status):
    def policy(o):
        if o["phase"] == "formation" and o["self"] == "a0001":
            return {"opening": {"assignment": "check", "allowance": 6000}}
        if o["phase"] == "review":
            return decide(o["proposals"])
        return propose(o)

    colony = engine(Policy(policy), initial_agents=2, bank_contacts=1)
    root = funded(colony)
    child = colony.agents["a0002"]
    colony.record_proposal(child, {"plan": "verify", "amount": 6000}, "bank")
    if bank_status == "revise":
        colony.apply_review(
            "bank", [colony.proposal_view(child)], decide([colony.proposal_view(child)], "revise")
        )
    # Root occupies this round's bank slot. The employer may independently sponsor
    # the candidate, whose explicit acceptance cancels its old bank application.
    colony.formation()
    assert child.parent == root.name
    assert child.application is None
    assert child.last_feedback["from"] == root.name


def test_dynamic_examples_respect_minimum_working_capital():
    colony = engine(max_output_tokens=8192, max_grant=250000)
    a = colony.agents["a0001"]
    messages, _ = colony.observation(a, "formation")
    obs = json.loads(messages[-1]["content"])
    assert obs["schema"]["proposal"]["amount"] >= colony.call_credits
    assert obs["schema"]["opening"]["allowance"] >= colony.call_credits


def test_audit_detects_changed_application_sponsor(tmp_path):
    from ant_colony.config import ColonyConfig
    from ant_colony.engine import ColonyMAS
    from ant_colony.environment import Task
    from ant_colony.providers import ScriptedDemoProvider
    from ant_colony.storage import RunStore, audit

    store = RunStore(tmp_path / "audit-sponsor")
    ColonyMAS(
        ColonyConfig(), Task("demo", "Sum 1 through 100.", "5050"), ScriptedDemoProvider(), store.emit
    ).run()
    store.close()
    assert audit(store.path)["application_spent"] > 0
    events = [json.loads(line) for line in store.path.read_text().splitlines()]
    event = next(e for e in events if e["event"] == "application_sponsored")
    event["target"] = "a9999"
    store.path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    with pytest.raises(ValueError, match="sponsor"):
        audit(store.path)


def test_expired_job_releases_pending_applicant_to_bank_queue():
    colony = engine(Policy(propose), initial_agents=2)
    root = funded(colony)
    child = colony.agents["a0002"]
    colony.open_job(root, {"assignment": "verify", "allowance": 6000})
    colony.record_proposal(child, {"plan": "verify", "amount": 6000}, root.name, consent=True)
    colony.round = root.opening["expires"] + 1
    proposals = colony.formation()
    assert child.name in {p["agent"] for p in proposals}
    assert child.application["target"] == "bank"


def test_company_demo_exercises_review_hiring_and_recursive_spawning():
    from ant_colony.config import ColonyConfig
    from ant_colony.engine import ColonyMAS
    from ant_colony.environment import Task
    from ant_colony.providers import ScriptedDemoProvider

    events = []
    c = ColonyMAS(
        ColonyConfig(),
        Task("demo", "Sum 1 through 100.", "5050"),
        ScriptedDemoProvider(),
        lambda event, **data: events.append({"event": event, **data}),
    )
    result = c.run()
    assert result["status"] == "completed"
    assert any(e["event"] == "hire" for e in events)
    assert any(e["event"] == "proposal_review" and e["reviewer"] != "bank" for e in events)
    assert any(e["event"] == "proposal_review" and e["decision"] == "revise" for e in events)
    assert any(
        e["event"] == "snapshot" and any(len(a["children"]) == 2 for a in e["state"]["agents"])
        for e in events
    )
