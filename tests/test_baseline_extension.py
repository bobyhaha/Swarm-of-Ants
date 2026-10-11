"""The requested extra controls must fit the existing study authorization."""

import importlib.util
from pathlib import Path

import pytest


def extension_module():
    path=Path(__file__).resolve().parents[1] / "scripts/add_programbench_baselines.py"
    spec=importlib.util.spec_from_file_location("baseline_extension",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_supplemental_plan_accounts_for_existing_eighty_five_dollars():
    plan=extension_module().extension_plan({"allocated_maximum_usd":85,"global_ceiling_usd":100})
    assert plan["allocated_maximum_usd"]==99
    assert plan["unallocated_reserve_usd"]==1
    assert {(r["task"],r["solo_model"],r["cap"]) for r in plan["episodes"]}=={
        (task,model,cap) for task in ("xsv","hexyl") for model,cap in (("sol",5),("luna",2))}
    assert len({r["id"] for r in plan["episodes"]})==4
    assert all(r["solo"] and not r["pilot"] for r in plan["episodes"])


@pytest.mark.parametrize("allocated,ceiling",[(90,100),(85,150)])
def test_extension_cannot_increase_global_authorization(allocated,ceiling):
    with pytest.raises(ValueError,match="100"):
        extension_module().extension_plan({"allocated_maximum_usd":allocated,"global_ceiling_usd":ceiling})
