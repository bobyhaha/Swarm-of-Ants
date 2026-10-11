from copy import deepcopy
import json

import pytest

from ant_colony.context import build_review_context, byte_size


def test_bank_overflow_preserves_review_and_failed_execution_facts():
    required = {"proposals": [{"amount": 100, "minimum_grant": 1, "plan": "Useful delegation"}] * 4,
                "verified_evidence": {"public_version": 4, "public_test": {"passed": 1, "total": 2},
                    "applicants": [{"team": "t1", "progress": {"writes": 3, "tests": 0,
                        "recent_executions": [{"command": "python test.py", "output": "failure" * 1000,
                                               "status": "ok", "exit_code": 1, "revision": 4}]}}]}}
    before = deepcopy(required)
    messages, dropped = build_review_context("Review evidence", required, [{"memory": "Remember"}], 2000)
    obs = json.loads(messages[-1]["content"])
    receipt = obs["verified_evidence"]["applicants"][0]["progress"]["recent_executions"][0]
    assert byte_size(messages) <= 2000 and dropped == 1
    assert obs["proposals"] == required["proposals"]
    assert receipt["exit_code"] == 1 and receipt["revision"] == 4 and receipt["details_optional"]
    assert "output" not in receipt and obs["context"] == [{"memory": "Remember"}]
    assert required == before


def test_bank_overflow_keeps_public_score_and_failure_identity():
    required = {"verified_evidence": {"public_test": {"status": "public_probe_score", "passed": 1, "total": 2,
                    "failures": [{"probe": "p1", "input": {"stdin": "x"*4000}, "expected": {"exit_code": 1}}]}}}
    messages, dropped = build_review_context("Review", required, [], 1000)
    result = json.loads(messages[-1]["content"])["verified_evidence"]["public_test"]
    assert byte_size(messages) <= 1000 and dropped == 1
    assert result["passed"] == 1 and result["total"] == 2
    assert result["failures"] == [{"probe": "p1", "details_optional": True}]


def test_review_never_shortens_essential_proposals_or_silently_raises_limit():
    with pytest.raises(ValueError, match="Required context"):
        build_review_context("Review", {"proposals": [{"plan": "x"*4000}]}, [], 1000)


def test_review_keeps_full_evidence_when_it_fits():
    required = {"verified_evidence": {"applicants": [], "public_test": {"passed": 1}}}
    messages, dropped = build_review_context("Review", required, [], 1000)
    assert json.loads(messages[-1]["content"])["verified_evidence"] == required["verified_evidence"]
    assert dropped == 0
