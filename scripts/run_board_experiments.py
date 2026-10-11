"""Predeclared, fresh-start board ablation; finite allocations never exceed $50.

Six independent episodes, three per condition, run in two concurrent processes.
Every child uses both its provider cap and the existing durable campaign ledger.
No automatic retries, checkpoint continuation, or selection by final test score.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    study = args.out.resolve()
    study.mkdir(parents=True, exist_ok=False)  # Never replay paid work on rerun.
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("Provide the authorized credential through the environment")
    jobs = [
        dict(condition=condition, repetition=i, cap_usd="2")
        for i in range(1, 4)
        for condition in ("board", "no-board")
    ]
    allocation = sum(Decimal(j["cap_usd"]) for j in jobs)
    if allocation > Decimal("50"):
        raise ValueError("Allocated run ceilings exceed the authorized $50 study budget")
    source_files = list((root / "ant_colony").glob("*.py")) + [
        root / "scripts/run_commit0_campaign.py",
        Path(__file__).resolve(),
    ]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    plan = {
        "authorized_cap_usd": "50",
        "allocated_ceiling_usd": str(allocation),
        "design": "Three independent fresh-start repetitions per condition; no API seed control. Compare all outcomes.",
        "primary_measure": "Original tests passed / 201 at selected terminal artifact; missing scores reported separately.",
        "secondary_measures": [
            "API dollars",
            "source edit events",
            "test calls",
            "redundant task claims",
            "team structure",
            "self-selected strategies",
            "context bounds",
            "invalid replies",
        ],
        "only_config_difference": "bank_board_size: 6 versus 0; includes board planning/review token overhead",
        "implementation_sha256": hashes,
        "jobs": jobs,
    }
    (study / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    base = json.loads((root / "configs/commit0-coding.json").read_text())
    for condition, size in (("board", 6), ("no-board", 0)):
        (study / f"{condition}.json").write_text(
            json.dumps(dict(base, bank_board_size=size), indent=2) + "\n"
        )

    def run(job):
        if (study / "STOP").exists():
            return {**job, "status": "not_started_stop"}
        for path, expected in hashes.items():
            if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
                return {**job, "status": "not_started_source_changed"}
        name = f"{study.name}-{job['condition']}-r{job['repetition']}"
        directory = root / "runs" / name
        command = [
            sys.executable,
            str(root / "scripts/run_commit0_campaign.py"),
            "--campaign",
            str(root.parent / "code-testing-v1"),
            "--task",
            str(root.parent / "code-testing-v1/datasets/commit0-tinydb/task.json"),
            "--config",
            str(study / f"{job['condition']}.json"),
            "--models",
            str(root / "configs/commit0-luna-coding-models.json"),
            "--out",
            str(directory),
            "--max-usd",
            job["cap_usd"],
            "--parallel",
        ]
        with (study / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
        report = (
            json.loads((directory / "report.json").read_text())
            if (directory / "report.json").exists()
            else {}
        )
        return {**job, "directory": str(directory), "exit_code": result.returncode, "report": report}

    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            (study / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(
                json.dumps({k: result[k] for k in ("condition", "repetition", "exit_code") if k in result}),
                flush=True,
            )
    return 0 if all(r.get("exit_code") == 0 for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
