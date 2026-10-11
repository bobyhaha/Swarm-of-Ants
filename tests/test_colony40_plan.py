"""The population change and concurrent experiment retain explicit bounds."""

from collections import Counter
from decimal import Decimal
import importlib.util
from pathlib import Path

import pytest

from test_programbench import config


def module(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("colony40", scripts / "run_programbench_colony40.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_forty_initial_agents_and_recursive_constant_bounds(monkeypatch):
    cfg = module(monkeypatch).experiment_config(config().to_dict())
    counts = Counter(cfg["initial_models"])
    counts[cfg["bank_model"]] += 1
    assert counts == {"astra": 4, "sol": 16, "luna": 20}
    assert cfg["initial_agents"] + 1 == 40 and cfg["max_agents"] + 1 == 64
    assert cfg["max_rounds"] == 60
    assert cfg["max_children"] == cfg["bank_contacts"] == 2
    assert cfg["context_bytes"] == 64000 and cfg["coordination_context_bytes"] == 32000


def test_pending_charges_count_before_parallel_admission(monkeypatch):
    mod = module(monkeypatch)
    mod.check_admission({"blocked": False, "accounted_usd": 30, "pending": 2}, Decimal(15))
    with pytest.raises(ValueError, match="headroom"):
        mod.check_admission({"blocked": False, "accounted_usd": 85.01}, Decimal(15))
    with pytest.raises(ValueError, match="billing"):
        mod.check_admission({"blocked": True, "accounted_usd": 1}, Decimal(15))
    with pytest.raises(ValueError, match="15"):
        mod.check_admission({"blocked": False, "accounted_usd": 1}, Decimal(16))


def test_fresh_episode_identity_cannot_reuse_shared_spending_or_escape_directory(monkeypatch, tmp_path):
    mod = module(monkeypatch)
    ledger = mod.Ledger(tmp_path / "campaign.sqlite")
    try:
        ledger.register("v3-colony40-xsv", 15)
        mod.check_episode(ledger, "v4-recruitment40-xsv")
        with pytest.raises(ValueError, match="already exists"):
            mod.check_episode(ledger, "v3-colony40-xsv")
        for invalid in ("../escape", "/absolute", "", "a" * 97):
            with pytest.raises(ValueError, match="Episode ID"):
                mod.check_episode(ledger, invalid)
    finally:
        ledger.close()


def test_larger_fixed_capacity_is_explicit_and_context_limits_do_not_grow(monkeypatch):
    cfg = module(monkeypatch).experiment_config(config().to_dict(), bank_contacts=4,
                                                manager_contacts=3, max_children=3)
    assert (cfg["bank_contacts"], cfg["manager_contacts"], cfg["max_children"]) == (4, 3, 3)
    assert cfg["context_bytes"] == 64000 and cfg["coordination_context_bytes"] == 32000
    assert cfg["max_agents"] == 63 and cfg["initial_agents"] == 39


def test_all_astra_changes_only_population_models_and_spawn_allowlist(monkeypatch):
    mod = module(monkeypatch)
    bounds = dict(bank_contacts=4, manager_contacts=3, max_children=3)
    mixed = mod.experiment_config(config().to_dict(), **bounds)
    astra = mod.experiment_config(config().to_dict(), model_mix="all-astra", **bounds)
    assert set(k for k in astra if astra[k] != mixed[k]) == {"initial_models", "worker_models"}
    assert Counter(astra["initial_models"] + [astra["bank_model"]]) == {"astra": 40}
    assert astra["worker_models"] == ["astra"]
    assert astra["max_agents"] == 63 and astra["max_rounds"] == 60
    assert astra["bank_contacts"] == 4 and astra["max_children"] == astra["manager_contacts"] == 3
    with pytest.raises(ValueError, match="model mix"):
        mod.experiment_config(config().to_dict(), model_mix="typo")


def test_hierarchy_bias_is_explicit_and_inherited_by_every_worker_and_bank(monkeypatch):
    from ant_colony.config import ColonyConfig
    from ant_colony.engine import ColonyMAS
    from ant_colony.environment import Task
    from ant_colony.protocol import HIERARCHY_RULES
    from test_mechanism import Policy

    cfg = module(monkeypatch).experiment_config(config().to_dict(), model_mix='all-astra', hierarchy_bias=True)
    cfg.update(cost_mode='output_tokens', initial_bank=100_000_000, bank_board_size=0)
    c = ColonyMAS(ColonyConfig.from_dict(cfg), Task('fixture', 'Solve fixture', '2'), Policy())
    root = next(iter(c.agents.values()))
    c.round = 1
    c.grant(root, 10_000_000)
    child = c.new_agent('astra', 'Choose useful work', root, 1_000_000, 'Own a subproject')
    for agent in (root, child):
        for phase in ('application', 'formation', 'work', 'review'):
            messages, _ = c.observation(agent, phase)
            assert HIERARCHY_RULES in messages[0]['content']
            assert c.config.max_children == 2 and c.config.max_depth == 6
    captured = []
    c.call = lambda model, messages, **kw: captured.append(messages) or '{}'
    c.record_proposal(root, {'amount': 1000, 'plan': 'Own outcome'}, 'bank')
    c.fund([c.proposal_view(root)])
    assert HIERARCHY_RULES in captured[0][0]['content']
