"""A bounded strategic task board with exclusive root-team claims."""

from copy import deepcopy
import re

from .context import clip


BOARD_SCHEMA = [
    {
        "id": "short-stable-id",
        "title": "one deliverable",
        "scope": "owned files or responsibilities; avoid overlap",
        "acceptance": "observable completion check",
        "status": "open|claimed|blocked|done",
        "feedback": "next step, redundancy decision, or reason for reopening",
    }
]

BOARD_RULES = """
Maintain the short bank_board as the society's shared priorities and task registry.
List complementary, concrete deliverables with scope and acceptance checks. Agents
propose task_id to take a task; approving it assigns one root team exclusively.
The runtime creates claims after approvals. For a new task you are approving now,
leave status open; it will become claimed automatically. Claimed without an actual
approved grant never reserves a task or assigns its owner.
Check redundancy against BOTH the presented proposals and existing board claims.
For redundant proposals request revision toward a complementary open task, or
suggest joining the existing owner. Similar work under different IDs can still
overlap: compare their scopes semantically. A separate verification task is useful
only with an explicit distinct purpose. Reject broad duplicate reconstruction plans.
Use verified progress to give actionable feedback: what changed, what is blocked,
and the next implementation/test step. Do not repeat a generic 'inspect the repo'.
You can replace the bounded tasks list, retaining IDs for ongoing work. Keep active
claimed tasks unless explicitly marking them done or reopening with feedback.
Reopening suspends that team's work until another task is granted. Completion on
the board is your judgment, not a reward; only the environment awards reward.
Local managers still choose methods, hires, spawning and recursive delegation.
Specify outcomes, not permanent job titles or a prescribed organization chart.
Workers have the same action permissions; their models and prices may differ. Let specializations form and change
through voluntary proposals, observed results and local delegation.
"""

WORKER_BOARD_RULES = """
The bounded bank_board contains strategic tasks, not fixed agent roles. For a bank
proposal include task_id. Prefer unclaimed complementary work; an existing owner
may request a top-up for its claim. To suggest unlisted work, propose a short new
task_id and scope: the bank must add it before funding. Do not duplicate a claimed
deliverable; seek hiring by its owner during formation or propose distinct work.
Only teams with a claimed/blocked board task can work. Done or reopened tasks stop
work until a new claim is approved. Board feedback is advisory; runtime test evidence
is authoritative. Managers decompose their task locally, with bounded recursive
hiring/spawning. No prescribed roles: choose and revise your own strategy, and
design children's strategies around the current problem. Share evidence upward.
"""


class TaskBoard:
    def __init__(self, capacity, detail_bytes=240):
        self.capacity = capacity
        self.detail_bytes = detail_bytes
        self.items = {}
        self.version = 0

    @property
    def enabled(self):
        return self.capacity > 0

    def view(self):
        return {
            "capacity": self.capacity,
            "version": self.version,
            "tasks": deepcopy(list(self.items.values())),
        }

    def replace(self, rows):
        if not isinstance(rows, list) or len(rows) > self.capacity:
            raise ValueError("Task board exceeds its fixed capacity")
        proposed = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Board tasks must be objects")
            task_id = row.get("id")
            if not isinstance(task_id, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,23}", task_id):
                raise ValueError("Task IDs use 1–24 lowercase letters, digits, underscores or hyphens")
            if task_id in proposed:
                raise ValueError("Duplicate board task ID")
            old = self.items.get(task_id, {})
            item = {"id": task_id}
            for key, limit in (("title", 120), ("scope", self.detail_bytes), ("acceptance", self.detail_bytes), ("feedback", 180)):
                item[key] = clip(row.get(key, old.get(key, "")), limit)
                if key != "feedback" and not item[key].strip():
                    raise ValueError("Tasks need a title, scope and observable acceptance criterion")
            state = row.get("status", old.get("status", "open"))
            if state not in {"open", "claimed", "blocked", "done"}:
                raise ValueError("Invalid board task status")
            owner, team = old.get("owner"), old.get("team")
            if state == "claimed" and owner is None:
                # Models naturally describe the post-approval state in the same
                # response as a grant. Treat that as intent, not authority:
                # only claim(), after a validated grant, can assign ownership.
                state = "open"
            if state == "open" and owner is not None:
                if not item["feedback"].strip():
                    raise ValueError("Reopening a claimed task requires feedback")
                owner = team = None
            item.update(status=state, owner=owner, team=team, claimed_round=old.get("claimed_round"))
            proposed[task_id] = item
        for task_id, old in self.items.items():
            if old["owner"] and old["status"] in {"claimed", "blocked"} and task_id not in proposed:
                raise ValueError("Retain active claimed tasks; explicitly complete or reopen them first")
        self.items = proposed
        self.version += 1

    def conflict(self, owner, task_id):
        if not isinstance(task_id, str) or task_id not in self.items:
            return "Choose a listed open task_id; an unlisted idea needs a board task before funding."
        task = self.items[task_id]
        if task["status"] == "done":
            return "This task is already marked done; propose a distinct next task or verification scope."
        if task["owner"] and task["owner"] != owner:
            return f"Task {task_id} already belongs to {task['owner']} in {task['team']}. Choose complementary work or seek hiring by that owner."
        for other in self.items.values():
            if (
                other["owner"] == owner
                and other["id"] != task_id
                and other["status"] in {"claimed", "blocked"}
            ):
                return (
                    f"Finish or explicitly release your current task {other['id']} before claiming another."
                )
        if task["status"] == "blocked" and task["owner"] is None:
            return "This task is blocked; request a prerequisite or a complementary open task."
        return None

    def claim(self, task_id, owner, team, round_number):
        reason = self.conflict(owner, task_id)
        if reason:
            raise ValueError(reason)
        self.items[task_id].update(owner=owner, team=team, status="claimed", claimed_round=round_number)
        self.version += 1

    def release_team(self, team):
        released = []
        for item in self.items.values():
            if item["team"] == team:
                item.update(owner=None, team=None)
                if item["status"] != "done":
                    item["status"] = "open"
                released.append(item["id"])
        if released:
            self.version += 1
        return released

    def can_work(self, team):
        return not self.enabled or any(
            t["team"] == team and t["status"] in {"claimed", "blocked"} for t in self.items.values()
        )
