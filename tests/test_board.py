from copy import deepcopy
import json

import pytest

from ant_colony.board import TaskBoard
from ant_colony.context import byte_size
from test_mechanism import engine, Policy
from test_sponsorship import decide


def task(task_id="implement", **kwargs):
    return dict(
        id=task_id,
        title="Deliver checked behavior",
        scope="Implement one missing behavior",
        acceptance="Original test passes",
        status="open",
        feedback="",
        **kwargs,
    )


def propose(colony, name, task_id="implement"):
    agent = colony.agents[name]
    colony.record_proposal(agent, {"task_id": task_id, "plan": "Implement then test", "amount": 6000}, "bank")
    return colony.proposal_view(agent)


def test_board_is_bounded_atomic_and_claims_are_runtime_owned():
    board = TaskBoard(2)
    board.replace([task(owner="fake", team="fake")])
    assert board.items["implement"]["owner"] is None
    board.claim("implement", "a", "t", 1)
    before = board.view()
    for rows in ([task("x"), task("y"), task("z")], [], [task("bad ID")]):
        with pytest.raises(ValueError):
            board.replace(rows)
        assert board.view() == before
    rows = deepcopy(before["tasks"])
    rows[0]["status"] = "open"
    with pytest.raises(ValueError, match="feedback"):
        board.replace(rows)
    rows[0]["feedback"] = "Scope overlapped; return to proposals."
    board.replace(rows)
    assert board.items["implement"]["owner"] is None
    assert not board.can_work("t")


def test_duplicate_approvals_do_not_transfer_duplicate_funding():
    colony = engine(bank_board_size=2)
    colony.board.replace([task()])
    proposals = [propose(colony, name) for name in ("a0001", "a0002")]
    colony.apply_review("bank", proposals, decide(proposals))
    assert len(colony.teams) == 1
    assert colony.ledger.available("bank") == colony.config.initial_bank - 6000
    second = colony.agents["a0002"]
    assert second.application["status"] == "revise"
    assert "already belongs" in second.application["feedback"]
    assert "a0001" in colony.proposal_view(second)["claim_feedback"]
    colony.check()


def test_bank_can_describe_claim_and_approve_in_same_response():
    def policy(o):
        rows = deepcopy(o["bank_board"]["tasks"])
        rows[0]["status"] = "claimed"
        rows[0]["owner"] = "forged-owner-ignored"
        return {"tasks": rows, **decide(o["proposals"])}

    colony = engine(Policy(policy), bank_board_size=2)
    colony.board.replace([task()])
    colony.fund([propose(colony, "a0001")])
    assert colony.board.items["implement"]["owner"] == "a0001"
    assert colony.board.items["implement"]["status"] == "claimed"
    assert len(colony.teams) == 1
    # A requested claim with no valid approval leaves the task available.
    board = TaskBoard(1)
    row = task()
    row["status"] = "claimed"
    board.replace([row])
    assert board.items["implement"]["status"] == "open"
    assert board.items["implement"]["owner"] is None


def test_one_active_task_per_root_but_own_topup_allowed():
    colony = engine(bank_board_size=2)
    colony.board.replace([task(), task("verify")])
    root = colony.agents["a0001"]
    colony.grant(root, 6000, "implement")
    colony.grant(root, 2000, "implement")
    with pytest.raises(ValueError, match="current task"):
        colony.grant(root, 2000, "verify")
    assert colony.rights(root) == 8000
    colony.close_team(colony.teams[root.team])
    assert colony.board.items["implement"]["status"] == "open"
    assert colony.board.items["implement"]["owner"] is None
    colony.check()


@pytest.mark.parametrize("status", ["done", "open"])
def test_done_or_reopened_task_stops_work_and_notifies_owner(status):
    colony = engine(bank_board_size=2)
    colony.board.replace([task()])
    root = colony.agents["a0001"]
    colony.grant(root, 6000, "implement")
    rows = colony.board.view()["tasks"]
    rows[0].update(status=status, feedback="Deliverable changed; propose the next useful contribution.")
    colony.update_board(rows)
    colony.work()
    assert colony.calls == 0
    assert "Deliverable changed" in root.last_feedback["text"]
    assert colony.ledger.rewarded == 0


def test_planning_precedes_applications_and_duplicate_feedback_reaches_revision():
    def policy(o):
        if o["phase"] == "bank":
            if o["initial_planning"]:
                return {"tasks": [task()], "decisions": []}
            return decide(o["proposals"])
        if o["phase"] == "application":
            return {"proposal": {"task_id": "implement", "plan": "Implement and test", "amount": 6000}}
        return {}

    colony = engine(Policy(policy), bank_board_size=2, initial_agents=2, max_rounds=2)
    colony.run()
    observed = colony.provider.observations
    assert observed[0]["initial_planning"]
    apps = [o for o in observed if o["phase"] == "application"]
    assert apps[0]["bank_board"]["tasks"][0]["id"] == "implement"
    assert "already belongs" in apps[-1]["application"]["feedback"]
    assert len([e for e in colony.events if e["event"] == "grant"]) == 1


def test_invalid_board_update_does_not_approve_against_stale_state():
    colony = engine(
        Policy(lambda o: {"tasks": [task("a"), task("b")], **decide(o["proposals"])}), bank_board_size=1
    )
    colony.board.replace([task()])
    colony.fund([propose(colony, "a0001")])
    assert not colony.teams
    assert "Board update rejected" in colony.format_errors["bank"]


def test_board_does_not_claim_semantic_overlap_is_automatically_solved():
    colony = engine(bank_board_size=2)
    colony.board.replace([task("a"), task("b")])  # Identical scopes under different IDs.
    colony.grant(colony.agents["a0001"], 6000, "a")
    proposal = propose(colony, "a0002", "b")
    assert proposal["claim_feedback"] is None  # Semantic review is the bank's job.
    colony.fund([proposal])
    obs = colony.provider.observations[-1]
    assert obs["bank_board"]["tasks"][0]["owner"] == "a0001"
    assert len(obs["proposals"]) <= colony.config.bank_contacts


def test_worker_chooses_and_changes_specialization_with_no_role_enum():
    colony = engine(
        Policy(lambda o: {"strategy": "Follow failing serialization tests; change focus when fixed."})
    )
    root = colony.agents["a0001"]
    colony.grant(root, 15000)
    colony.work()
    assert root.trainable_system_prompt.startswith("Follow failing")
    child = colony.new_agent("worker", "Compare storage edge cases", root, 2000, "One subtask")
    assert child.trainable_system_prompt == "Compare storage edge cases"
    assert root.children == [child.name]
    assert any(e["event"] == "strategy_changed" for e in colony.events)


def test_board_inputs_and_admission_remain_bounded_as_population_grows():
    sizes = []
    for population in (10, 1000):
        colony = engine(bank_board_size=6, initial_agents=population, max_agents=population)
        colony.board.replace([task(f"t{i}") for i in range(6)])
        colony.fund([], planning=True)
        colony.formation()
        assert colony.calls == 3  # One planning call and two sponsored applications.
        obs = colony.provider.observations
        assert all(len(o["bank_board"]["tasks"]) == 6 for o in obs)
        sizes.append(max(byte_size(o) for o in obs))
        messages, _ = colony.observation(colony.agents["a0001"], "work")
        assert byte_size(messages) <= colony.config.context_bytes
    assert abs(sizes[1] - sizes[0]) < 32


def test_board_and_strategy_text_are_byte_bounded():
    colony = engine(bank_board_size=8, context_bytes=64000)
    long = '\\"漢' * 1000
    rows = [task(f"t{i}") for i in range(8)]
    for row in rows:
        for key in ("title", "scope", "acceptance", "feedback"):
            row[key] = long
    colony.board.replace(rows)
    root = colony.agents["a0001"]
    colony.remember_and_sleep(root, {"strategy": long})
    assert len(root.trainable_system_prompt.encode()) <= colony.config.strategy_bytes
    messages, _ = colony.observation(root, "application")
    assert byte_size(messages) <= colony.config.context_bytes
    assert len(json.loads(messages[-1]["content"])["bank_board"]["tasks"]) == 8
