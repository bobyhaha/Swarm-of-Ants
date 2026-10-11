"""New benchmark isolation, heterogeneous billing, and terminal-only grading."""

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile

import pytest

from ant_colony.campaign_budget import Ledger, BudgetStop as StudyStop
from ant_colony.config import ColonyConfig
from ant_colony.engine import ColonyMAS
from ant_colony.environment import Task
from ant_colony.programbench import ProgramBenchWorkbench, PIN
from ant_colony.providers import OpenRouterProvider, Generation, UncertainCall
from ant_colony.storage import RunStore, audit
from test_providers_and_evaluation import response


ROOT = Path(__file__).resolve().parents[1]
MODELS = json.loads((ROOT / "configs/programbench-high-models.json").read_text())


def config(**changes):
    return replace(ColonyConfig.from_dict(json.loads((ROOT / "configs/programbench-mixed-high.json").read_text())), **changes)


def fixture(tmp_path):
    root = tmp_path / "task"
    (root / "seed").mkdir(parents=True)
    (root / "seed/PROJECT_NOTES.md").write_text("Blank project.\n")
    (root / "seed/.gitignore").write_text("executable\n")
    spec = b"Reconstruct a fictional command-line fixture from behavior."
    (root / "SPEC.md").write_bytes(spec)
    manifest = {"benchmark":"programbench", "id":"fixture__tool.abc", "source_revision":PIN,
                "image":"unused", "spec_sha256":hashlib.sha256(spec).hexdigest(),
                "build_command":"./compile.sh && test -x executable"}
    path = root / "task.json"
    path.write_text(json.dumps(manifest))
    bench = ProgramBenchWorkbench(path,tmp_path / "coding", "/venv/bin/python")
    agent = SimpleNamespace(name="a1",team="t1")
    bench.observation(agent)
    return bench,agent


def test_exact_population_prices_and_no_model_role_assignments():
    colony = ColonyMAS(config(bank_board_size=0),Task("fixture","compute 1+1","2"),
                       OpenRouterProvider(MODELS,100,lambda *a,**k:None,transport=lambda p:response()))
    assert Counter(a.model for a in colony.agents.values()) == {"astra":1,"sol":8,"luna":10}
    assert colony.config.bank_model == "astra"  # total including bank: 2/8/10
    assert len({a.trainable_system_prompt for a in colony.agents.values()}) == 1
    assert all(a.wealth == 0 and a.parent is None for a in colony.agents.values())
    assert colony.call_budget("astra") == 1_819_200
    assert colony.call_budget("sol") == 363_840
    assert colony.call_budget("luna") == 18_192
    assert colony.generation_credits(Generation("",1,100,"0.1200001")) == 120001
    # A funded Luna can manage another Luna; the same child cap applies to all tiers.
    parent = next(a for a in colony.agents.values() if a.model == "luna")
    colony.grant(parent,500000)
    with pytest.raises(ValueError,match="one bounded call"):
        colony.new_agent("astra","verify",parent,100000)
    for _ in range(2):
        colony.new_agent("luna","verify",parent,50000)
    with pytest.raises(ValueError):
        colony.new_agent("luna","verify",parent,50000)
    colony.check()


@pytest.mark.parametrize("alias",["astra","sol","luna"])
def test_every_model_payload_requests_high(alias):
    seen = []
    provider = OpenRouterProvider(MODELS,100,lambda *a,**k:None,
                                  transport=lambda p:seen.append(p) or response())
    provider.complete(alias,[{"role":"user","content":'{"phase":"work"}'}],16384,"c1")
    assert seen[0]["reasoning"] == {"effort":"high"}
    assert seen[0]["model"] == MODELS[alias]["id"]
    assert seen[0]["provider"]["require_parameters"] is True


@pytest.mark.parametrize("request_pricing", [False, True])
def test_dollar_priced_run_and_independent_audit(tmp_path, request_pricing):
    store = RunStore(tmp_path / "run")
    def transport(payload):
        obs = json.loads(payload["messages"][-1]["content"])
        if obs["phase"] == "application":
            reply = {"proposal":{"plan":"Compute 2","amount":500000}}
        elif obs["phase"] == "bank":
            reply = {"decisions":[{"agent":p["agent"],"amount":500000,"decision":"approve","feedback":"Compute 2"} for p in obs["proposals"]]}
        else:
            reply = {"submit":"2"}
        return response(json.dumps(reply),tokens=17,cost=0.00123)
    cfg = config(initial_agents=1,initial_models=("luna",),bank_board_size=0,max_rounds=2,
                 request_pricing=request_pricing)
    colony = ColonyMAS(cfg,Task("fixture","1+1","2"),OpenRouterProvider(MODELS,100,store.emit,transport=transport),store.emit)
    report = colony.run()
    store.close()
    assert report["status"] == "completed"
    assert report["burned"] == report["calls"] * 1230
    assert report["output_tokens"] == report["calls"] * 17
    assert audit(store.path)["accounting_valid"]


@pytest.mark.parametrize("model", ["astra", "sol", "luna"])
def test_solo_baseline_has_no_paid_bank_or_application(tmp_path, model):
    from ant_colony.baselines import SoloMAS, solo_config
    store=RunStore(tmp_path / "solo")
    phases=[]
    def transport(payload):
        phases.append(json.loads(payload["messages"][-1]["content"])["phase"])
        assert payload["model"] == MODELS[model]["id"]
        assert payload["reasoning"] == {"effort":"high"}
        return response('{"submit":"2"}',cost=0.001)
    original=config(initial_bank=20000000)
    cfg=solo_config(original,model)
    assert cfg.initial_models == cfg.worker_models == (model,)
    assert cfg.initial_agents == cfg.max_agents == 1
    assert cfg.bank_model == model and cfg.bank_board_size == 0
    assert (cfg.work_turns,cfg.max_rounds,cfg.context_bytes) == (original.work_turns,original.max_rounds,original.context_bytes)
    provider=OpenRouterProvider(MODELS,20,store.emit,transport=transport)
    colony=SoloMAS(cfg,Task("fixture","1+1","2"),provider,store.emit)
    report=colony.run()
    store.close()
    assert report["status"]=="completed" and phases==["work"]
    assert audit(store.path)["accounting_valid"]


def test_sponsorship_uses_child_model_and_pays_from_parent():
    provider=OpenRouterProvider(MODELS,100,lambda *a,**k:None,transport=lambda p:response(cost=0.01))
    colony=ColonyMAS(config(bank_board_size=0),Task("f","1+1","2"),provider)
    parent,child=[a for a in colony.agents.values() if a.model=="luna"][:2]
    colony.grant(parent,100000)
    bank=colony.ledger.available("bank")
    colony.sponsored_application(child,parent.name)
    assert colony.rights(parent)==90000 and colony.rights(child)==0
    assert colony.ledger.available("bank")==bank
    costly=next(a for a in colony.agents.values() if a.model=="astra")
    calls=colony.calls
    assert colony.sponsored_application(costly,parent.name)=={}
    assert colony.calls==calls


@pytest.mark.parametrize("bad",[Generation("",1,0,"nan"),Generation("",1,0,"1",True)])
def test_invalid_receipt_keeps_full_virtual_reserve(bad):
    provider = OpenRouterProvider(MODELS,100,lambda *a,**k:None,transport=lambda p:response())
    provider.complete = lambda *a:bad
    colony = ColonyMAS(config(),Task("f","1+1","2"),provider)
    with pytest.raises(UncertainCall):
        colony.call("astra",[{"role":"user","content":"fixture"}])
    assert colony.uncertain_credits == colony.ledger.burned == colony.call_budget("astra")


def test_programbench_preserves_languages_and_compact_notes(tmp_path):
    bench,agent = fixture(tmp_path)
    for name,text in {"main.rs":"fn main() {}\n","Cargo.toml":"[package]\n", ".cargo/config.toml":"[net]\noffline = true\n", "compile.sh":"#!/bin/sh\necho fixture\n"}.items():
        bench.execute(agent,{"kind":"write","path":name,"base_sha256":"new","content":text})
    assert set(bench.hashes(bench.source(agent.team))) == {"main.rs","Cargo.toml","compile.sh",".gitignore",".cargo/config.toml"}
    assert bench.syntax_error(bench.code_files(bench.source(agent.team))) is None
    for path in ["../secret","executable","target/cache","SPEC.md"]:
        with pytest.raises(ValueError):
            bench.execute(agent,{"kind":"write","path":path,"base_sha256":"new","content":"bad"})
    note = bench.source(agent.team) / "PROJECT_NOTES.md"
    note.write_text("x"*15999)
    obs = bench.observation(agent)
    assert obs["notes_bytes"] == 15999 and obs["notes_limit_bytes"] == 16000
    assert len(obs["project_notes"].encode()) <= 1600
    (bench.source(agent.team) / "escape.rs").symlink_to(tmp_path / "secret")
    with pytest.raises(ValueError,match="Unsafe"):
        bench.code_files(bench.source(agent.team))


def test_native_selection_is_frozen_before_tests_and_drops_keys(tmp_path,monkeypatch):
    bench,agent = fixture(tmp_path)
    bench.execute(agent,{"kind":"write","path":"compile.sh","base_sha256":"new","content":"#!/bin/sh\ntrue\n"})
    monkeypatch.setenv("OPENROUTER_API_KEY","test-secret")
    calls=[]
    def native(argv,**kwargs):
        calls.append(argv)
        assert "OPENROUTER_API_KEY" not in kwargs["env"]
        assert (bench.directory / "selected-artifact.json").exists()
        archive = bench.directory / "submission" / bench.manifest["id"] / "submission.tar.gz"
        with tarfile.open(archive) as tar:
            assert tar.getnames() == [".gitignore","compile.sh"]
            assert tar.getmember("compile.sh").mode == 0o755
        content = b"" if len(calls)==1 else b'{"status":"native_scored","score":0.5,"passed":1,"total":2}'
        return SimpleNamespace(returncode=0,stdout=content,stderr=b"")
    monkeypatch.setattr("ant_colony.programbench.run",native)
    assert bench.apply(SimpleNamespace(author=agent.name)) == 0.5
    assert len(calls) == 2 and "eval" in calls[0]
    with pytest.raises(RuntimeError,match="already performed"):
        bench.apply(SimpleNamespace(author=agent.name))


def test_budget_stop_grades_public_once_without_more_inference(tmp_path):
    bench,_ = fixture(tmp_path)
    calls=[]
    def final(action):
        calls.append(action.author)
        bench.score = 0.5
        return 0.5
    bench.apply=final
    provider=OpenRouterProvider(MODELS,0.00001,lambda *a,**k:None,
                                transport=lambda p:pytest.fail("No request can be afforded"))
    colony=ColonyMAS(config(),Task("fixture","rebuild fixture","hidden","programbench"),provider,workbench=bench)
    report=colony.run()
    assert report["status"] == "completed"
    assert report["submission_mode"] == "automatic_budget_checkpoint"
    assert calls == ["__public__"] and report["rewarded"] == config().reward//2


def test_study_cap_cannot_reset_between_episodes(tmp_path):
    ledger=Ledger(tmp_path / "campaign.sqlite")
    ledger.register("first",60)
    first=ledger.reserve("first",60,"work","a1","astra")
    ledger.settle(first,60,{"fixture":True})
    ledger.register("second",60)
    with pytest.raises(StudyStop):
        ledger.reserve("second",41,"work","a1","astra")
    second=ledger.reserve("second",40,"work","a1","astra")
    ledger.uncertain(second)
    assert ledger.summary()["accounted_usd"] == 100
    assert ledger.summary()["blocked"]
    ledger.close()
