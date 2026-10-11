"""Quality-first activation, paid discovery, collaboration and hard boundaries."""

from collections import Counter
from decimal import Decimal
import json
from types import SimpleNamespace

import pytest

from ant_colony.engine import ColonyMAS
from ant_colony.environment import Task
from ant_colony.providers import OpenRouterProvider, UncertainCall
from ant_colony.quality import rules
from test_programbench import config, MODELS, fixture
from test_providers_and_evaluation import response


def colony(policy=lambda o: {}, **changes):
    cfg = config(initial_agents=5, initial_models=("astra",) * 5,
                 worker_models=("astra",), bank_board_size=0, quality_first=True,
                 funding_recovery=True, request_pricing=True, discovery_budget=10_000_000,
                 discovery_contacts=2, hierarchy_bias=True, **changes)
    events, payloads = [], []
    def transport(payload):
        payloads.append(payload)
        obs = json.loads(payload["messages"][-1]["content"])
        return response(json.dumps(policy(obs)), tokens=20, cost=0.001)
    p = OpenRouterProvider(MODELS, 100, lambda *a, **k: None, transport=transport)
    c = ColonyMAS(cfg, Task("fixture", "Solve a documented fixture", "2"), p,
                  lambda e, **kw: events.append({"event": e, **kw}))
    c.round = 1
    c.events, c.payloads = events, payloads
    return c


def test_sponsored_discovery_reads_then_queues_evidence_before_grant():
    def policy(o):
        if o["calls_remaining"] == 3:
            return {"inspect": {"kind": "read", "path": "SPEC.md"}}
        return {"offer": {"target": "bank", "plan": "Implement based on SPEC.md evidence", "amount": 5_000_000},
                "memory": "SPEC.md inspected; acceptance is a passing fixture"}
    c = colony(policy)
    c.workbench = SimpleNamespace(inspect_public=lambda a: {"text": "Fixture evidence"})
    before = c.ledger.wealth
    c.discover()
    assert c.calls == 4 and c.discovery_spent == 4000
    assert c.ledger.wealth == before - 4000
    assert not c.teams and all(c.rights(a) == 0 for a in c.agents.values())
    assert len([a for a in c.agents.values() if a.application]) == 2
    assert all(a.discovery_calls <= 3 for a in c.agents.values())
    c.round = 2
    c.discover()
    assert len([a for a in c.agents.values() if a.application]) == 4
    assert all(e["wallet"] == "bank" for e in c.events if e["event"] == "call_reserved")
    c.check()


def test_discovery_is_bounded_even_without_an_offer():
    c = colony()
    c.workbench = SimpleNamespace(inspect_public=lambda a: {})
    c.discover()
    assert c.calls == 6
    assert Counter(a.discovery_calls for a in c.agents.values()) == {3: 2, 0: 3}
    c.discovery_spent = c.config.discovery_budget
    c.discover()
    assert c.calls == 6


def test_direct_offer_hires_small_trial_then_executes_with_lower_real_output_limit():
    c = colony(lambda o: {"accept": True} if o["phase"] == "application" else {})
    root, child = c.agents["a0001"], c.agents["a0002"]
    c.grant(root, 15_000_000)
    before = c.ledger.wealth
    capital = 2 * c.working_minimum("astra")
    assert capital < c.call_budget("astra")
    c.discovery_offer(child, {"target": root.name, "amount": capital, "plan": "Verify a distinct boundary"})
    assert c.ledger.wealth == before and c.calls == 0
    assert c.offers_for(root)[0]["agent"] == child.name
    c.open_job(root, {"candidate": child.name, "trial": True, "allowance": capital,
                      "assignment": "Own boundary verification and evidence"})
    c.formation()
    assert child.parent == root.name and c.rights(child) == capital
    before = c.rights(child)
    c.ask(child, "work")
    assert c.rights(child) == before - 1000
    assert c.payloads[-1]["max_tokens"] == c.config.trial_max_output_tokens
    assert c.payloads[-1]["reasoning"] == {"effort": "high"}
    c.check()


def test_addressable_messages_keep_constant_volume_and_lookup_is_bounded():
    c = colony(lambda o: {"lookup": "boundary verification"})
    sender = c.agents["a0001"]
    for a in c.agents.values():
        c.grant(a, 5_000_000)
    c.ask(sender, "formation")  # lookup is a legal phase action, not a parse failure
    assert not any(e["event"] == "invalid_reply" for e in c.events)
    msg = {"messages": [{"to": "a0005", "text": "Evidence from another team"}]}
    c.deliver(c.outbound(sender, msg, set(), "work"))
    assert c.agents["a0005"].inbox[-1]["from"] == sender.name
    with pytest.raises(ValueError, match="Too many"):
        c.outbound(sender, {"messages": msg["messages"] * 5}, set(c.agents), "work")
    with pytest.raises(ValueError, match="Recipient"):
        c.outbound(sender, {"messages": [{"to": "missing", "text": "x"}]}, set(), "work")
    c.lookup_collaborators(sender, "verification")
    assert len(sender.directory) <= c.config.manager_contacts


def test_phase_prompts_and_newborn_inherit_quality_and_hierarchy():
    c = colony()
    root = c.agents["a0001"]
    c.grant(root, 15_000_000)
    child = c.new_agent("astra", "Choose methods", root, 4_000_000, "Own independent evidence")
    for a in (root, child):
        for phase in ("work", "formation", "application", "review"):
            messages, _ = c.observation(a, phase)
            system = messages[0]["content"]
            assert "Maximize correctness and completeness" in system
            assert "Prefer useful nested teams" in system
            assert "Maximize SOCIETY wealth" not in system
            assert len(system.encode()) < 4000
            if phase == "application":
                assert "lookup" not in json.loads(messages[-1]["content"])["schema"]
    assert "only the current schema" in rules(c.config, "bank")


def test_quality_stagnation_requests_review_without_automatic_submission():
    c = colony(progress_control=True)
    c.workbench = object()
    c.round = 20
    c.assess_progress()
    assert c.force_bank_review and c.final_answer is None
    c.round = 40
    c.assess_progress()
    assert c.final_answer is None


@pytest.mark.parametrize("path", ["/hidden-tests/a", "../secret", "/reference-docs/../../secret", "/public/../secret"])
def test_discovery_rejects_nonpublic_paths_before_container_creation(tmp_path, path):
    bench, _ = fixture(tmp_path)
    bench.sandbox = lambda *a, **k: pytest.fail("Must reject before opening a sandbox")
    with pytest.raises(ValueError):
        bench.inspect_public({"kind": "read", "path": path})


def test_provider_missing_receipt_records_typed_diagnostic_without_raw_body():
    events = []
    p = OpenRouterProvider(MODELS, 10, lambda e, **k: events.append({"event": e, **k}))
    with pytest.raises(UncertainCall, match="receipt"):
        p.billing_response({"id": "gen-test", "error": {"code": 502, "message": "DO_NOT_LOG_RAW_BODY",
                                                            "metadata": {"error_type": "server"}}})
    assert "DO_NOT_LOG_RAW_BODY" not in json.dumps(events)
    assert events[-1]["error_code"] == 502 and events[-1]["generation_id"] == "gen-test"
    assert p.billing_response(response())["usage"]["cost"] == 0.000124
    assert p.spent == Decimal(0)


def test_quality_plan_keeps_40_astras_hard_200_and_shared_500(monkeypatch):
    from test_colony40_plan import module
    mod = module(monkeypatch)
    cfg = mod.experiment_config(config().to_dict(), model_mix="all-astra", hierarchy_bias=True,
                                quality_first=True, bank_contacts=4, manager_contacts=3, max_children=3)
    assert cfg["initial_models"] == ["astra"] * 39 and cfg["bank_model"] == "astra"
    assert cfg["worker_models"] == ["astra"] and cfg["cost_mode"] == "api_usd" and cfg["rent_per_round"] == 0
    mod.check_admission({"blocked": False, "accounted_usd": 77.35, "ceiling_usd": 500}, Decimal(200), Decimal(200))
    with pytest.raises(ValueError):
        mod.check_admission({"blocked": False, "accounted_usd": 301, "ceiling_usd": 500}, Decimal(200), Decimal(200))
    with pytest.raises(ValueError):
        mod.check_admission({"blocked": False, "accounted_usd": 0, "ceiling_usd": 500}, Decimal('200.01'), Decimal(200))


def test_discovery_receipts_reconcile_independently_and_detect_corrupt_report(tmp_path):
    from ant_colony.storage import RunStore, audit
    c = colony(lambda o: {"offer": {"target": "bank", "amount": 5_000_000, "plan": "Own one evidenced outcome"}})
    store = RunStore(tmp_path / "discovery-audit")
    c.sink = store.emit
    c.provider.emit = store.emit
    c.emit("start", config=c.config.to_dict())
    c.workbench = SimpleNamespace(inspect_public=lambda a: {})
    c.discover()
    c.workbench = None
    c.emit("finished", report=c.report())
    store.close()
    result = audit(store.path)
    assert result["accounting_valid"] and result["discovery_spent"] == 2000
    rows = [json.loads(s) for s in store.path.read_text().splitlines()]
    rows[-1]["report"]["discovery_spent"] = 0
    store.path.write_text("\n".join(json.dumps(e) for e in rows) + "\n")
    with pytest.raises(ValueError, match="discovery pool"):
        audit(store.path)


def test_http_deadline_retains_uncertainty_without_retry(monkeypatch):
    import threading
    from ant_colony import providers
    gate = threading.Event()
    calls = []
    class Opener:
        def open(self, request, timeout):
            calls.append(True)
            gate.wait(1)
            raise TimeoutError("fixture")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture-not-a-real-key")
    monkeypatch.setattr(providers, "build_opener", lambda *a: Opener())
    p = OpenRouterProvider(MODELS, 100, lambda *a, **k: None)
    p.timeout_seconds = 0.01
    try:
        with pytest.raises(UncertainCall):
            p.complete("astra", [], 100, "c1")
        assert p.blocked and p.reserved > 0 and calls == [True]
    finally:
        gate.set()
