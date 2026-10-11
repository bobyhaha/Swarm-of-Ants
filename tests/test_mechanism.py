from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

from hayekmas.base.agent import BaseAgent
from hayekmas.base.env import BaseEnv
from ant_colony.config import ColonyConfig
from ant_colony.context import build_context, byte_size, clip
from ant_colony.engine import ColonyMAS
from ant_colony.environment import Task
from ant_colony.ledger import Ledger
from ant_colony.protocol import peer_graph
from ant_colony.providers import Generation, ScriptedDemoProvider, BudgetStop, UncertainCall
from ant_colony.storage import RunStore, audit


class Policy:
    scripted = False
    spent = reserved = Decimal(0)

    def __init__(self, fn=lambda obs: {}, tokens=5):
        self.fn, self.tokens = fn, tokens
        self.observations = []

    def complete(self, model, messages, max_tokens, call_id):
        obs = json.loads(messages[-1]["content"])
        self.observations.append(obs)
        value = self.fn(obs)
        return Generation(value if isinstance(value, str) else json.dumps(value), self.tokens)


def engine(policy=None, **kwargs):
    events = []
    config = replace(ColonyConfig(), **kwargs)
    colony = ColonyMAS(
        config,
        Task("t", "Compute 1+1. Submit an integer.", "2"),
        policy or Policy(),
        lambda event, **data: events.append({"event": event, **data}),
    )
    colony.events = events
    colony.round = 1
    return colony


def funded(colony, name="a0001", value=15000):
    agent = colony.agents[name]
    colony.grant(agent, value)
    return agent


def test_original_interfaces_and_source_provenance():
    colony = engine()
    assert isinstance(colony.agents["a0001"], BaseAgent)
    assert isinstance(colony.env, BaseEnv)
    root = Path(__file__).parents[1]
    manifest = json.loads((root / "vendor/SOURCE_MANIFEST.json").read_text())
    assert manifest["commit"] == "d272266f7d6a0c302b7af5d20f1326b63c5656fb"
    for path, entry in manifest["files"].items():
        if not entry["modified"]:
            assert hashlib.sha256((root / path).read_bytes()).hexdigest() == entry["upstream_sha256"]


def test_invalid_reply_feedback_is_bounded_private_and_clears_on_recovery():
    colony = engine()
    a = funded(colony)
    for text in ['Prose {"memory":"x"}', '{"sleep":1,"sleep":2}', "{} {}"]:
        assert colony.parse(text, {"memory", "sleep"}, a.name) == {}
        messages, _ = colony.observation(a, "work")
        error = json.loads(messages[-1]["content"])["last_protocol_error"]
        assert "executed no actions" in error and len(error.encode()) < 400
        other, _ = colony.observation(colony.agents["a0002"], "formation")
        assert json.loads(other[-1]["content"])["last_protocol_error"] is None
    assert colony.parse('{"memory":"recovered"}', {"memory"}, a.name) == {"memory": "recovered"}
    assert a.name not in colony.format_errors


def test_reward_schedule_is_visible_to_applicants_and_bank_without_private_answer():
    policy = Policy(
        lambda o: (
            {"proposal": {"plan": "Verify a result", "amount": 6000}, "accept": True}
            if o["phase"] == "application"
            else {}
        )
    )
    colony = engine(policy, initial_agents=1, max_rounds=1, reward=80000, token_price=2)
    colony.task = Task("t", "Compute a result", "private-reference-answer")
    colony.run()
    bank = next(o for o in policy.observations if o["phase"] == "bank")
    application = next(o for o in policy.observations if o["phase"] == "application")
    assert bank["objective"] == application["objective"]
    assert bank["objective"]["maximum_terminal_reward"] == 80000
    assert bank["objective"]["cost_per_generated_token"] == 2
    assert "private-reference-answer" not in json.dumps(policy.observations)


def test_money_only_changes_through_tokens_and_unique_external_reward():
    ledger = Ledger(1000, lambda *a, **k: None)
    ledger.transfer("bank", "team", 600, "grant")
    assert ledger.wealth == 1000
    ledger.burn("team", 70, "tokens")
    ledger.transfer("team", "bank", 530, "return")
    assert ledger.reward_once("task", 200)
    assert not ledger.reward_once("task", 200)
    assert ledger.wealth == 1130
    with pytest.raises(ValueError):
        ledger.transfer("bank", "other", 2000, "overdraw")
    with pytest.raises(ValueError):
        ledger.burn("bank", True, "bool")


def test_recursive_spawn_conserves_money_and_enforces_both_caps():
    colony = engine(max_children=1, max_depth=2)
    root = funded(colony)
    before = colony.ledger.wealth
    child = colony.new_agent("worker", "check", root, 4000, "derive")
    grandchild = colony.new_agent("worker", "check", child, 1000, "verify")
    assert colony.ledger.wealth == before
    assert child.wealth == grandchild.wealth == 0
    assert colony.depth(grandchild) == 2
    with pytest.raises(ValueError, match="cap"):
        colony.new_agent("worker", "another", root, 1000, "check")
    with pytest.raises(ValueError, match="cap"):
        colony.new_agent("worker", "too deep", grandchild, 800, "check")
    colony.check()


def test_spawn_is_not_a_free_money_or_immediate_recursion_loophole():
    colony = engine(initial_agents=1, max_agents=2)
    root = funded(colony)
    with pytest.raises(ValueError):
        colony.new_agent("worker", "x", root, 0, "x")
    with pytest.raises(ValueError):
        colony.new_agent("unlisted", "x", root, 1000, "x")
    child = colony.new_agent("worker", "x", root, 1000, "x")
    assert child.born_round == 1
    colony.work()
    assert not any(o.get("self") == child.name for o in colony.provider.observations)
    with pytest.raises(ValueError, match="Population"):
        colony.new_agent("worker", "x", root, 1000, "x")


def test_hiring_requires_consent_and_recruitment_is_local():
    def choose(obs):
        if obs["phase"] == "formation" and obs["self"] == "a0001":
            return {"opening": {"allowance": 1000, "assignment": "verify"}}
        return {}  # invitee declines

    colony = engine(Policy(choose), initial_agents=2)
    funded(colony)
    colony.agents["a0002"].bank_closed = True
    colony.formation()
    assert colony.agents["a0002"].team is None
    assert any(
        o["phase"] == "application" and o.get("application_target") == "a0001"
        for o in colony.provider.observations
    )


def test_consent_hire_transfers_rights_and_freezes_work_membership():
    def choose(obs):
        if obs["phase"] == "formation" and obs["self"] == "a0001":
            return {"opening": {"allowance": 1000, "assignment": "verify"}}
        if obs["phase"] == "application" and obs.get("application_target") == "a0001":
            return {"accept": True, "proposal": {"plan": "verify", "amount": 1000}}
        if obs["phase"] == "review":
            return {
                "decisions": [
                    {"agent": p["agent"], "amount": 1000, "decision": "approve", "feedback": "Proceed"}
                    for p in obs["proposals"]
                ]
            }
        return {}

    colony = engine(Policy(choose), initial_agents=2)
    root = funded(colony)
    colony.agents["a0002"].bank_closed = True
    colony.formation()
    child = colony.agents["a0002"]
    assert child.parent == root.name and child.team == root.team
    assert colony.teams[root.team].allowances[child.name] == 1000
    colony.phase = "work"
    with pytest.raises(ValueError, match="frozen"):
        colony.leave(child)
    with pytest.raises(ValueError):
        colony.hire(child, root, 0, "cycle")
    colony.check()


def test_parent_can_reclaim_dormant_child_rights():
    colony = engine()
    root = funded(colony)
    child = colony.new_agent("worker", "verify", root, 1000, "check")
    colony.allocate(root, [{"agent": child.name, "amount": -900}])
    assert colony.teams[root.team].allowances[child.name] == 100
    with pytest.raises(ValueError):
        colony.allocate(root, [{"agent": child.name, "amount": -101}])
    colony.check()


@pytest.mark.parametrize("n", [1, 2, 3, 7, 20, 101])
def test_peer_graph_degree_and_symmetry(n):
    names = [f"a{i}" for i in range(n)]
    for round_number in range(1, 5):
        graph = peer_graph(names, round_number, 3)
        for name, neighbors in graph.items():
            assert len(neighbors) <= 3
            assert name not in neighbors
            assert all(name in graph[p] for p in neighbors)


def test_peer_rotation_reaches_all_without_global_roster_in_context():
    names = [f"a{i}" for i in range(8)]
    encountered = set()
    for r in range(1, 8):
        encountered.update(peer_graph(names, r, 1)["a0"])
    assert encountered == set(names) - {"a0"}


def test_bank_admission_and_context_do_not_grow_with_population():
    def propose(obs):
        if obs["phase"] == "application":
            return {"proposal": {"plan": "test plan", "amount": 1000}}  # adversarial non-slot attempts
        return {}

    for count in (6, 60, 200):
        colony = engine(Policy(propose), initial_agents=count, max_agents=count)
        proposals = colony.formation()
        colony.fund(proposals)
        assert len(proposals) == colony.config.bank_contacts
        bank = [o for o in colony.provider.observations if o["phase"] == "bank"]
        assert len(bank) == 1 and len(bank[0]["proposals"]) == 2
        assert colony.max_context_seen <= colony.config.context_bytes
        assert max(len(o.get("peers", [])) for o in colony.provider.observations) <= 2


def test_bounded_memory_context_and_multibyte_clipping():
    colony = engine()
    root = funded(colony)
    root.memory = "m" * 900
    root.inbox = [{"from": "x", "text": "字" * 5000}] * 100
    messages, omitted = colony.observation(root, "work")
    assert byte_size(messages) <= colony.config.context_bytes and omitted > 0
    assert len(clip("字" * 100, 10).encode()) <= 10
    with pytest.raises(ValueError, match="Required context"):
        build_context("x", {"task": "x" * 200}, [], 100)


def test_cross_team_work_messages_are_paid_without_private_scratchpad_access():
    colony = engine()
    root, other = funded(colony), funded(colony, "a0002")
    colony.teams[other.team].scratchpad[other.name] = {"text": "OTHER_TEAM_PRIVATE_NEEDLE", "round": 1}
    root.read_keys = [other.name]
    messages, _ = colony.observation(root, "work")
    text = json.dumps(messages)
    assert "OTHER_TEAM_PRIVATE_NEEDLE" not in text
    assert '"peers"' not in messages[-1]["content"]
    assert "When in team work with the team members closely. You may also communicate with agents from other teams." in messages[0]["content"]
    colony.provider.fn = lambda obs: {
        "messages": [{"to": other.name, "text": "Please verify this interface"}],
        "read_keys": [other.name],
    } if obs["self"] == root.name else {}
    before = colony.ledger.wealth
    colony.work()
    assert colony.ledger.wealth == before - colony.calls * colony.provider.tokens
    assert any(e["event"] == "action_rejected" and e["action"] == "read_keys" for e in colony.events)
    deliveries = [e for e in colony.events if e["event"] == "message" and e["scope"] == "work"]
    assert len(deliveries) == colony.config.work_turns
    assert all(e["source"] == root.name and e["target"] == other.name for e in deliveries)
    observations = [o for o in colony.provider.observations if o["self"] == other.name]
    assert all(len(o["work_peers"]) == 1 and o["work_peers"][0]["id"] == root.name for o in observations)
    # Same-wave responses cannot see one another; the next wave gets the message.
    assert not any(c.get("message", {}).get("from") == root.name for c in observations[0]["context"])
    assert any(c.get("message", {}).get("from") == root.name for c in observations[1]["context"])
    assert "OTHER_TEAM_PRIVATE_NEEDLE" not in json.dumps(
        [o for o in colony.provider.observations if o["self"] == root.name])


def test_work_cannot_message_unlisted_agents_or_unfunded_population():
    colony = engine(initial_agents=8, work_turns=1)
    roots = [funded(colony, f"a{i:04d}", 2000) for i in range(1, 7)]
    root = roots[0]
    peers = peer_graph([a.name for a in roots], colony.round, colony.config.peer_contacts)[root.name]
    outside = next(a.name for a in roots[1:] if a.name not in peers)
    colony.provider.fn = lambda obs: {"messages": [{"to": outside, "text": "outside contacts"}]} if obs["self"] == root.name else {}
    colony.work()
    assert not any(e["event"] == "message" for e in colony.events)
    assert any(e["event"] == "action_rejected" and e["agent"] == root.name and e["action"] == "messages" for e in colony.events)
    assert all(p["id"] not in {"a0007", "a0008"} for o in colony.provider.observations for p in o["work_peers"])
    assert not any(o["self"] in {"a0007", "a0008"} for o in colony.provider.observations)


@pytest.mark.parametrize("count", [6, 60, 200])
def test_work_contact_context_and_message_volume_stay_bounded(count):
    def send(obs):
        return {"messages": [{"to": p["id"], "text": "Share a concise result"} for p in obs["work_peers"]]}

    colony = engine(Policy(send), initial_agents=count, max_agents=count, initial_bank=count*2000, work_turns=1)
    for a in colony.agents.values():
        colony.grant(a, 1000)
    colony.work()
    assert all(len(o["work_peers"]) <= colony.config.peer_contacts for o in colony.provider.observations)
    assert all(o["limits"]["peer_contacts"] == colony.config.peer_contacts for o in colony.provider.observations)
    assert colony.max_context_seen <= colony.config.context_bytes
    sent = [e for e in colony.events if e["event"] == "message"]
    assert len(sent) == count * colony.config.peer_contacts
    assert all(len(a.inbox) <= colony.config.inbox_size for a in colony.agents.values())


def test_sleep_has_no_polling_cost_and_messages_wake_agents():
    colony = engine(Policy(), max_rounds=2)
    agent = funded(colony)
    colony.remember_and_sleep(agent, {"sleep": 2})
    assert agent.wake_round == 4
    colony.work()
    assert colony.calls == 0
    colony.deliver([("runtime", agent.name, "new work", "receipt")])
    colony.work()
    assert colony.calls == 2


def test_malformed_response_is_charged_and_not_retried():
    colony = engine(Policy(lambda obs: "not json", tokens=17), initial_agents=1)
    before = colony.ledger.wealth
    colony.formation()
    assert colony.calls == 1
    assert colony.ledger.wealth == before - 17
    assert any(e["event"] == "invalid_reply" for e in colony.events)


@pytest.mark.parametrize("text", ['{"memory":"a","memory":"b"}', '{"sleep":NaN}', "[]", '{"mint":9999}'])
def test_malformed_decision_envelopes_fail_closed(text):
    colony = engine()
    assert colony.parse(text, {"memory", "sleep"}, "a") == {}


def test_expiry_and_shutdown_return_unused_grants():
    colony = engine(initial_agents=1, max_rounds=2, grant_rounds=1)
    root = funded(colony)
    old = root.team
    result = colony.run()
    assert old not in colony.teams
    assert any(e["event"] == "team_closed" and e["round"] == 2 for e in colony.events)
    assert result["bank"] == result["final_wealth"]
    assert result["rewarded"] == 0 and result["status"] == "round_limit"


def test_bank_cannot_duplicate_grants_or_fund_nonapplicants():
    def bank(obs):
        return {
            "decisions": [{"agent": "a0001", "amount": 2000, "decision": "approve", "feedback": "Proceed"}]
            * 2
        }

    colony = engine(Policy(bank))
    for name in ("a0001", "a0002"):
        colony.record_proposal(colony.agents[name], {"amount": 2000, "plan": "x"}, "bank")
    colony.fund([colony.proposal_view(colony.agents[n]) for n in ("a0001", "a0002")])
    assert colony.ledger.available(colony.agents["a0001"].team) == 2000
    assert colony.agents["a0002"].team is None
    assert any(e["event"] == "review_rejected" for e in colony.events)


def test_only_one_submission_is_evaluated_without_best_of_oracle():
    def work(obs):
        if obs["phase"] == "work" and obs["manager"]:
            return {"submit": "WRONG" if obs["self"] == "a0001" else "2"}
        return {}

    colony = engine(Policy(work), initial_agents=2, max_rounds=1)
    funded(colony)
    funded(colony, "a0002")
    report = colony.run()
    assert report["final_answer"] == "WRONG" and report["score"] == 0
    assert len([e for e in colony.events if e["event"] == "reward"]) == 1
    assert colony.env.step_count == 1


def test_demo_audit_and_determinism(tmp_path):
    reports = []
    for i in range(2):
        store = RunStore(tmp_path / str(i))
        colony = ColonyMAS(
            ColonyConfig(),
            Task("demo", "Sum integers 1 through 100.", "5050"),
            ScriptedDemoProvider(),
            store.emit,
        )
        reports.append(colony.run())
        store.close()
        result = audit(store.path)
        assert result["accounting_valid"] and result["complete_evaluation"]
        assert reports[-1]["population"] > 6
        assert reports[-1]["estimated_tokens"] == reports[-1]["output_tokens"]
    assert reports[0] == reports[1]


def test_log_tampering_is_detected(tmp_path):
    store = RunStore(tmp_path / "audit")
    ColonyMAS(
        ColonyConfig(),
        Task("demo", "Sum integers 1 through 100.", "5050"),
        ScriptedDemoProvider(),
        store.emit,
    ).run()
    store.close()
    data = store.path.read_text()
    data = data.replace('"amount":50000', '"amount":50001', 1)
    store.path.write_text(data)
    with pytest.raises(ValueError):
        audit(store.path)


def test_uncertain_provider_retains_credits_and_stops(tmp_path):
    class Failing(Policy):
        def complete(self, *args):
            raise UncertainCall("timeout")

    store = RunStore(tmp_path / "uncertain")
    colony = ColonyMAS(ColonyConfig(), Task("t", "1+1?", "2"), Failing(), store.emit)
    with pytest.raises(UncertainCall):
        colony.run()
    store.close()
    assert colony.report()["uncertain_credits"] == 768
    assert colony.calls == 1
    assert audit(store.path)["uncertain_credits"] == 768


def test_preflight_budget_stop_is_free(tmp_path):
    class Stopped(Policy):
        def complete(self, *args):
            raise BudgetStop("limit")

    store = RunStore(tmp_path / "stop")
    colony = ColonyMAS(ColonyConfig(), Task("t", "1+1?", "2"), Stopped(), store.emit)
    report = colony.run()
    store.close()
    assert report["burned"] == 0 and report["status"] == "budget_stopped"
    assert audit(store.path)["accounting_valid"]


def test_nonconforming_usage_stops_without_losing_audit_reservation(tmp_path):
    store = RunStore(tmp_path / "usage-overflow")
    colony = ColonyMAS(ColonyConfig(), Task("t", "1+1?", "2"), Policy(tokens=9999), store.emit)
    with pytest.raises(UncertainCall):
        colony.run()
    store.close()
    assert not colony.report()["usage_verified"]
    assert audit(store.path)["uncertain_credits"] == 768


def test_leave_returns_rights_and_removes_private_scratchpad():
    colony = engine()
    root = funded(colony)
    child = colony.new_agent("worker", "verify", root, 1000, "check")
    team = colony.teams[root.team]
    team.scratchpad[child.name] = {"text": "secret", "round": 1}
    colony.phase = "formation"
    colony.leave(child)
    assert child.team is None and not root.children
    assert child.name not in team.scratchpad and team.allowances[root.name] == 15000
    colony.check()


@pytest.mark.parametrize(
    "change",
    [
        {"bank_contacts": 0},
        {"max_children": True},
        {"context_bytes": 100},
        {"initial_agents": 500},
        {"token_price": -1},
        {"rent_per_round": -1},
    ],
)
def test_invalid_limits_rejected(change):
    with pytest.raises(ValueError):
        replace(ColonyConfig(), **change)
