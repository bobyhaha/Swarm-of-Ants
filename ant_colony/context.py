"""A hard serialized-input bound; history is never automatically concatenated."""

from copy import deepcopy
import json


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def clip(text, limit):
    if not isinstance(text, str):
        raise ValueError("Expected text")
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def byte_size(value):
    return len(dumps(value).encode("utf-8"))


def messages(system, observation):
    return [{"role": "system", "content": system}, {"role": "user", "content": dumps(observation)}]


def build_context(system, required, optional, limit):
    """Add bounded optional items until full. Never truncate the task or schema.

    The unit is UTF-8 bytes of the *entire serialized messages array*, not a
    tokenizer estimate. Provider framing is separate, fixed overhead.
    """
    observation = dict(required)
    observation["context"] = []
    if byte_size(messages(system, observation)) > limit:
        raise ValueError(
            "Required context exceeds context_bytes; raise the configured constant or shorten task"
        )
    dropped = 0
    for item in optional:
        observation["context"].append(item)
        if byte_size(messages(system, observation)) > limit:
            observation["context"].pop()
            dropped += 1
    return messages(system, observation), dropped


def build_review_context(system, required, optional, limit):
    """Retain financial/review facts; fit verbose evidence after them if needed."""
    if byte_size(messages(system, {**required, "context": []})) <= limit:
        return build_context(system, required, optional, limit)
    required = deepcopy(required)
    details = []
    evidence = required.get("verified_evidence", {})
    for applicant in evidence.get("applicants", []):
        for index, receipt in enumerate(applicant.get("progress", {}).get("recent_executions", [])):
            details.append({"execution_detail": {"team": applicant["team"], "receipt_index": index,
                                                  **deepcopy(receipt)}})
            receipt["command"] = clip(receipt.get("command", ""), 120)
            receipt.pop("output", None)
            receipt["details_optional"] = True
    evidence["verbose_details_in_context_if_space"] = True
    if byte_size(messages(system, {**required, "context": []})) > limit:
        public_test = evidence.get("public_test") or {}
        for failure in public_test.get("failures", []):
            details.append({"public_probe_failure_detail": deepcopy(failure)})
        if public_test.get("failures"):
            public_test["failures"] = [{"probe": f.get("probe"), "details_optional": True}
                                       for f in public_test["failures"]]
    # Status, exit codes, revision, counts, scores and all proposals stay required.
    # A truncated shell output must never be mistaken for a passing test.
    return build_context(system, required, details + optional, limit)
