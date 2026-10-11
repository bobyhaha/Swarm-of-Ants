"""Durable authorized caps and conservative isolation of transport timeouts."""

from decimal import Decimal
import json

import pytest

from ant_colony.campaign_budget import Ledger, BudgetStop
from ant_colony.config import ColonyConfig
from ant_colony.engine import ColonyMAS
from ant_colony.environment import Task
from ant_colony.providers import ScriptedDemoProvider
from ant_colony.storage import RunStore, audit
from test_colony40_plan import module


def test_authorized_cap_persists_and_preserves_prior_spending_and_holds(tmp_path):
    path = tmp_path / "campaign.sqlite"
    first, second = Ledger(path), Ledger(path)
    first.register("old", 100)
    paid = first.reserve("old", 80, "work", "a1", "sol")
    first.settle(paid, 80, {})
    failed = first.reserve("old", 20, "work", "a1", "sol")
    first.uncertain(failed)
    first.retain_failed_reservation(failed, "Inspected timeout; retain the full maximum, with no retry")
    before = first.summary()
    first.amend_limit(500, "User explicitly authorized a $500 total campaign budget including prior spending")
    assert second.limit == 500
    assert second.summary()["accounted_usd"] == before["accounted_usd"] == 100
    second.register("fresh", 450)
    with pytest.raises(BudgetStop, match="budget"):
        second.reserve("fresh", 401, "work", "a1", "sol")
    remaining = second.reserve("fresh", 400, "work", "a1", "sol")
    second.settle(remaining, 400, {})
    assert Ledger(path).summary()["accounted_usd"] == 500
    with first.connect() as db:
        assert db.execute('SELECT old_limit,new_limit FROM budget_amendments').fetchone()[:] == (100, 500)


def test_lowering_below_exposure_and_unexplained_amendments_are_rejected(tmp_path):
    ledger = Ledger(tmp_path / "campaign.sqlite")
    ledger.register("old", 100)
    call = ledger.reserve("old", 60, "work", "a1", "sol")
    ledger.settle(call, 60, {})
    for cap, reason in [(50, "User requests a lower cap but charges already exceed it"), (500, ""), (0, "User requests zero spending after prior charges")]:
        with pytest.raises(ValueError):
            ledger.amend_limit(cap, reason)
    assert ledger.limit == 100 and ledger.summary()["accounted_usd"] == 60


def test_sealed_attempt_and_replacement_share_total_without_erasing_holds(tmp_path, monkeypatch):
    ledger = Ledger(tmp_path / "campaign.sqlite")
    ledger.amend_limit(500, "User authorized $500 cumulative campaign spending")
    ledger.register('first', 15)
    call = ledger.reserve('first', 2, 'work', 'a1', 'astra')
    with pytest.raises(ValueError, match='pending'):
        ledger.amend_run_cap('first', 2, 'User changed the experiment budget and prompt')
    ledger.settle(call, 1, {})
    held = ledger.reserve('first', 2, 'work', 'a1', 'astra')
    ledger.uncertain(held)
    ledger.retain_failed_reservation(held, 'Reviewed historical error; its full reserve remains held')
    with ledger.connect() as db:
        original = [tuple(r) for r in db.execute('SELECT * FROM calls')]
    for value in (2.99, 501):
        with pytest.raises(ValueError, match='Cap'):
            ledger.amend_run_cap('first', value, 'User changed the experiment budget and prompt')
    change = ledger.amend_run_cap('first', 3, 'User authorized one $100 experiment including the first attempt')
    assert change['old_cap'] == 15 and change['new_cap'] == 3
    with pytest.raises(BudgetStop, match='budget'):
        ledger.reserve('first', 0.01, 'work', 'a1', 'astra')
    mod = module(monkeypatch)
    mod.check_admission(ledger.summary(), Decimal(97), Decimal(100))
    mod.check_attempt_budget(ledger, 'first', Decimal(100), Decimal(97))
    with pytest.raises(ValueError, match='Combined'):
        mod.check_attempt_budget(ledger, 'first', Decimal(100), Decimal('97.01'))
    ledger.register('replacement', 97)
    paid = ledger.reserve('replacement', 97, 'work', 'a1', 'astra')
    ledger.settle(paid, 97, {})
    assert ledger.summary()['accounted_usd'] == 100 and ledger.limit == 500
    with ledger.connect() as db:
        assert [tuple(r) for r in db.execute('SELECT * FROM calls WHERE run=?', ('first',))] == original
        assert db.execute('SELECT old_cap,new_cap FROM run_budget_amendments').fetchone()[:] == (15, 3)


def test_timeout_stops_own_episode_and_retains_full_cost_without_blocking_peer(tmp_path):
    path = tmp_path / "campaign.sqlite"
    broken, peer = Ledger(path, parallel=True), Ledger(path, parallel=True)
    try:
        broken.register("broken", 80)
        peer.register("peer", 80)
        call = broken.reserve("broken", 60, "work", "a1", "sol")
        broken.quarantine_timeout(call)
        assert broken.summary("broken")["blocked"]
        assert not peer.summary()["blocked"]
        assert peer.summary()["accounted_usd"] == 60 and peer.summary()["pending"] == 0
        with pytest.raises(BudgetStop, match="episode timed out"):
            broken.reserve("broken", 1, "work", "a1", "sol")
        with pytest.raises(BudgetStop, match="budget"):
            peer.reserve("peer", 41, "work", "a2", "luna")
        allowed = peer.reserve("peer", 40, "work", "a2", "luna")
        peer.settle(allowed, 10, {})
        assert peer.summary()["accounted_usd"] == 70
        with pytest.raises(BudgetStop):
            peer.attach("broken", 80)
    finally:
        broken.close()
        peer.close()
    with Ledger(path).connect() as db:
        row = db.execute('SELECT actual,reserved,status FROM calls WHERE id=?', (call,)).fetchone()
        assert tuple(row) == (None, 60, 'quarantined_timeout')


@pytest.mark.parametrize("failure", ["uncertain", "overrun"])
def test_other_billing_failures_still_stop_the_entire_campaign(tmp_path, failure):
    ledger = Ledger(tmp_path / "campaign.sqlite")
    ledger.register("first", 20)
    ledger.register("peer", 20)
    call = ledger.reserve("first", 10, "work", "a1", "sol")
    if failure == "uncertain":
        ledger.uncertain(call)
    else:
        with pytest.raises(BudgetStop):
            ledger.settle(call, 11, {})
    with pytest.raises(ValueError):
        ledger.quarantine_timeout(call)
    with pytest.raises(BudgetStop):
        ledger.reserve("peer", 1, "work", "a2", "luna")
    assert ledger.summary()["blocked"]


def test_controller_timeout_isolation_never_suppresses_contract_or_audit_failure(monkeypatch):
    mod = module(monkeypatch)
    local = {"blocked": True, "pending": 0, "quarantined_timeouts": 1}
    global_account = {"blocked": False}
    check = {"accounting_valid": True}
    events = [{"event": "campaign_reserved", "reasoning_effort": "high"}]
    assert not mod.campaign_failure(local, global_account, check, events)
    assert mod.campaign_failure(local, {"blocked": True}, check, events)
    assert mod.campaign_failure({**local, "blocked": False, "quarantined_timeouts": 0}, {"blocked": True}, check, events)
    assert mod.campaign_failure(local, global_account, {"accounting_valid": False}, events)
    assert mod.campaign_failure(local, global_account, check, events + [{"event": "api_limit_violation"}])
    assert mod.campaign_failure({**local, "pending": 1}, global_account, check, events)
    mod.check_admission({"blocked": False, "accounted_usd": 450, "ceiling_usd": 500}, Decimal(15))
    with pytest.raises(ValueError, match="headroom"):
        mod.check_admission({"blocked": False, "accounted_usd": 486, "ceiling_usd": 500}, Decimal(15))


@pytest.mark.parametrize("delta,valid", [(100, True), (-1, False)])
def test_sponsorship_authorizes_a_ceiling_not_an_exact_charge(tmp_path, delta, valid):
    store = RunStore(tmp_path / "sponsor")
    ColonyMAS(ColonyConfig(), Task("demo", "Sum 1 through 100", "5050"),
              ScriptedDemoProvider(), store.emit).run()
    store.close()
    events = [json.loads(line) for line in store.path.read_text().splitlines()]
    event = next(e for e in events if e["event"] == "application_sponsored")
    event["maximum_credits"] += delta
    store.path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    if valid:
        assert audit(store.path)["accounting_valid"]
    else:
        with pytest.raises(ValueError, match="sponsor"):
            audit(store.path)
