import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .config import ColonyConfig
from .engine import ColonyMAS
from .environment import Task
from .providers import OpenRouterProvider, ScriptedDemoProvider
from .replay import write_replay
from .storage import RunStore, audit, read_events


def load_task(path, task_id):
    tasks = [Task.from_dict(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Task IDs must be unique")
    chosen = [t for t in tasks if t.id == task_id] if task_id else tasks
    if len(chosen) != 1:
        raise ValueError("Select exactly one task with --task-id")
    return chosen[0]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="EoM ant colony: shared grants and bounded local collaboration"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("demo", "run", "commit0"):
        sub = commands.add_parser(command)
        sub.add_argument("--out", help="New output directory (must not already exist)")
        sub.add_argument("--config", help="Colony mechanism JSON")
        if command == "commit0":
            sub.add_argument("--task", required=True, help="Prepared Commit0 task.json")
        if command == "run":
            sub.add_argument("--tasks", required=True, help="Exact/research JSONL")
            sub.add_argument("--task-id")
        if command in {"run", "commit0"}:
            sub.add_argument(
                "--models", required=True, help="Provider model aliases and explicit price ceilings"
            )
            sub.add_argument("--max-usd", required=True, help="Maximum reserved + settled API dollars")
    sub = commands.add_parser("audit")
    sub.add_argument("run_dir")
    sub = commands.add_parser("replay")
    sub.add_argument("run_dir")
    sub.add_argument("--round", type=int)
    sub.add_argument("--html", action="store_true", help="Generate a standalone interactive replay")
    args = parser.parse_args(argv)
    if args.command == "audit":
        print(json.dumps(audit(Path(args.run_dir) / "events.jsonl"), indent=2))
        return 0
    if args.command == "replay":
        if args.html:
            print(write_replay(args.run_dir).resolve())
            return 0
        return replay(args.run_dir, args.round)
    config = (
        ColonyConfig.from_dict(json.loads(Path(args.config).read_text())) if args.config else ColonyConfig()
    )
    task = (
        load_task(args.tasks, args.task_id)
        if args.command == "run"
        else Task(
            "demo-sum",
            "Compute the sum of the integers 1 through 100 inclusive. Submit only the integer.",
            "5050",
        )
    )
    if args.command == "commit0":
        manifest = json.loads(Path(args.task).read_text())
        task = Task(
            manifest["id"],
            "Reconstruct the Commit0 "
            + manifest["id"]
            + " Python library from its original scaffold. Use isolated exec commands and persistent PROJECT_NOTES.md. "
            "Delegate implementation locally and run tests. Team changes integrate at round boundaries when conflict-free; "
            "the public checkpoint is automatically exported and graded at the deadline.",
            "protected visible tests",
            "commit0",
        )
    models = json.loads(Path(args.models).read_text()) if args.command in {"run", "commit0"} else None
    if models is not None and (set(config.worker_models) | {config.bank_model}) - models.keys():
        raise ValueError("Every worker/bank alias must exist in --models")
    out = args.out or "runs/" + args.command + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    store = RunStore(out)
    engine = None
    try:
        provider = (
            OpenRouterProvider(models, args.max_usd, store.emit)
            if models is not None
            else ScriptedDemoProvider()
        )
        workbench = None
        if args.command == "commit0":
            from .commit0 import Commit0Workbench

            workbench = Commit0Workbench(
                args.task, store.directory / "coding", store.emit, output_bytes=config.coding_output_bytes
            )
        engine = ColonyMAS(config, task, provider, store.emit, workbench=workbench)
        store.write_json("config.json", config.to_dict())
        if models is not None:
            store.write_json("models.json", models)
        report = engine.run()
        store.write_json("report.json", report)
    except Exception as exc:
        if engine:
            store.write_json("report.json", engine.report())
        print(f"Stopped: {exc}. Audit files: {store.directory.resolve()}", file=sys.stderr)
        return 1
    finally:
        store.close()
    store.write_json("audit.json", audit(store.path))
    write_replay(store.directory)
    print(json.dumps(report, indent=2))
    print(f"Run: {store.directory.resolve()}")
    return 0 if report["status"] == "completed" else 2


def replay(directory, selected_round):
    from rich.console import Console
    from rich.tree import Tree

    console = Console()
    states = [e["state"] for e in read_events(Path(directory) / "events.jsonl") if e["event"] == "snapshot"]
    if selected_round is not None:
        states = [s for s in states if s["round"] == selected_round]
    if not states:
        raise ValueError("No matching round snapshot")
    for state in states:
        console.print(
            f"Round {state['round']} | society {state['wealth']:,} | bank {state['bank']:,} "
            f"| burned {state['burned']:,} | reward {state['rewarded']:,}"
        )
        agents = {a["id"]: a for a in state["agents"]}
        for team in state["teams"]:
            tree = Tree(
                f"{team['id']} — manager {team['manager']} — {sum(team['allowances'].values()):,} credits"
            )

            def add(node, name):
                a = agents[name]
                branch = node.add(
                    f"{name} ({a['model']}) | rights {team['allowances'][name]:,} "
                    f"| wake round {a['wake_round']}"
                )
                for child in a["children"]:
                    add(branch, child)

            add(tree, team["manager"])
            console.print(tree)
    return 0
