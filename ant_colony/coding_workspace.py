"""Bounded coding state, isolated commands, and deterministic checkpoint integration.

Artifact storage grows with work. Model-visible records and integration admission
remain bounded; no model receives the population's complete code or transcript.
"""

from collections import deque
import ast
import hashlib
import json
from pathlib import Path
import tempfile

from .context import clip
from .sandbox import DockerSandbox, copy_source


def sha(data):
    return hashlib.sha256(data).hexdigest()


class CodingWorkspace:
    max_files = 128
    max_file_bytes = 64_000
    max_source_bytes = 2_000_000

    def syntax_error(self, files):
        for name, data in sorted(files.items()):
            if not name.endswith(".py"):
                continue
            try:
                ast.parse(data, filename=name)
            except (SyntaxError, ValueError, RecursionError) as exc:
                return clip(f"{name}:{getattr(exc, 'lineno', '?')}: {getattr(exc, 'msg', str(exc))}", 500)
        return None

    def initialize_workspace(self):
        self.public = self.directory / "public"
        copy_source(self.seed, self.public)
        self.bases, self.progress, self.recent, self.managers = {}, {}, {}, {}
        self.public_version = 0
        self.public_check = None
        self.merge_cursor = 0
        self.automatic_selection = False

    def code_files(self, directory):
        result = {}
        for path in sorted((Path(directory) / self.package).rglob("*.py")):
            if path.is_symlink() or not path.resolve().is_relative_to(Path(directory).resolve()):
                raise ValueError("Unsafe implementation path")
            data = path.read_bytes()
            if len(data) > self.max_file_bytes:
                raise ValueError("Implementation file exceeds 64000 bytes")
            result[str(path.relative_to(directory))] = data
        if len(result) > self.max_files or sum(map(len, result.values())) > self.max_source_bytes:
            raise ValueError("Implementation artifact exceeds file or byte limit")
        return result

    def hashes(self, directory):
        return {name: sha(data) for name, data in self.code_files(directory).items()}

    def register_team(self, team, manager=None):
        self.source(team)
        if manager:
            self.managers[team] = manager

    def initialize_team(self, team, source):
        self.bases[team] = self.hashes(source)
        self.progress[team] = {
            "revision": 0,
            "reads": 0,
            "writes": 0,
            "tests": 0,
            "reads_since_edit": 0,
            "last_changed": [],
            "last_test": None,
            "tested_revision": None,
            "conflicts": [],
            "integrations": 0,
            "executions": 0,
            "recent_executions": [],
        }

    def summary(self, team):
        if team not in self.progress:
            return None
        p = self.progress[team]
        return {
            **p,
            "last_changed": p["last_changed"][-4:],
            "conflicts": p["conflicts"][:4],
            "tests_current": p["tested_revision"] == p["revision"],
            "next_step": "Implement a concrete missing behavior or run a test; avoid repeating inventory."
            if p["reads_since_edit"] >= 3
            else "Read targeted evidence, edit, test, record the next step.",
        }

    def brief(self, result):
        if result is None:
            return None
        return {
            k: result[k]
            for k in ("status", "score", "passed", "total", "errors", "scope", "report")
            if k in result
        } | {
            "failures": result.get("failures", [])[:2],
            "diagnostic": clip(result.get("diagnostic", ""), 800),
        }

    def bounded_result(self, result, budget=None):
        """Bound encoded receipts too: backslashes/quotes can expand JSON greatly."""
        if result is None:
            return None
        budget = self.output_bytes if budget is None else budget
        result = dict(result)
        while len(json.dumps(result, ensure_ascii=False).encode()) > budget:
            fields = [k for k in ("text", "output", "diagnostic") if isinstance(result.get(k), str)]
            if not fields:
                return {"truncated": True}
            key = max(fields, key=lambda k: len(result[k]))
            if not result[key]:
                return {"truncated": True}
            original = result[key]
            result["shortened_for_context"] = True
            if key == "output":
                result["output_truncated"] = True
            if key == "text" and "total_lines" in result:
                lines = original.splitlines()
                start = int(lines[0].split(":", 1)[0])
                maximum = len(lines)

                def retain(count):
                    result[key] = "\n".join(lines[:count])
                    result["next_start"] = start + count
            else:
                maximum = len(original.encode())

                def retain(count):
                    result[key] = clip(original, count)

            # Keep the largest prefix that fits. Halving discarded roughly half
            # a receipt even when only its small JSON wrapper exceeded the cap.
            low, high = 0, maximum
            retain(0)
            if len(json.dumps(result, ensure_ascii=False).encode()) > budget:
                continue
            while low < high:
                mid = (low + high + 1) // 2
                retain(mid)
                if len(json.dumps(result, ensure_ascii=False).encode()) <= budget:
                    low = mid
                else:
                    high = mid - 1
            retain(low)
        return result

    def record_change(self, agent, before, after, kind):
        changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
        if not changed:
            return []
        p = self.progress[agent.team]
        p["revision"] += 1
        p["writes"] += 1
        p["reads_since_edit"] = 0
        p["last_changed"] = changed[-4:]
        for name in changed:
            self.emit(
                "source_written",
                agent=agent.name,
                team=agent.team,
                path=name,
                before_sha256=before.get(name, "new"),
                after_sha256=after.get(name),
                kind=kind,
            )
        return changed

    def remember_tool(self, agent, action, result):
        p = self.progress[agent.team]
        if action["kind"] in {"list", "read"}:
            p["reads"] += 1
            p["reads_since_edit"] += 1
        if action["kind"] == "exec":
            p["executions"] += 1
            p["recent_executions"] = (p["recent_executions"] + [{
                "command": clip(action.get("command", ""), 240),
                "status": result.get("status"), "exit_code": result.get("exit_code"),
                "output": clip(result.get("output", ""), 600),
                "revision": p["revision"],
            }])[-2:]
        if action["kind"] == "test":
            p["tests"] += 1
            p["last_test"] = self.brief(result)
            p["tested_revision"] = p["revision"]
        history = self.recent.setdefault(agent.name, deque(maxlen=2))
        history.append(
            {
                "kind": action["kind"],
                "path": action.get("path"),
                "command": clip(action.get("command", ""), 350),
                "result": self.bounded_result(result, budget=min(8000, self.output_bytes * 2 // 3)),
            }
        )

    def execute_command(self, agent, action):
        command = action.get("command")
        if not isinstance(command, str) or not command.strip() or len(command.encode()) > 32_000:
            raise ValueError("exec requires a nonempty command of at most 32000 bytes")
        source = self.source(agent.team)
        before = self.hashes(source)
        with tempfile.TemporaryDirectory(prefix="colony-command-") as temp:
            snapshot = Path(temp) / "snapshot"
            with DockerSandbox(self.manifest["image"], timeout=45, memory="768m", cpus="1") as box:
                box.load(source)
                box.put(self.root / "SPEC.md", "/workspace/SPEC.md", readonly=True)
                box.put(self.public / self.package, "/public", readonly=True)
                for name in self.manifest["protected_paths"]:
                    box.put(self.seed / self.safe_relative(name), "/workspace/" + name, readonly=True)
                result = box.execute(command, timeout=45)
                if result["status"] != "ok" or result["exit_code"] in (124, 137):
                    return {
                        "status": result["status"],
                        "exit_code": result["exit_code"],
                        "output": clip(result["output"], self.output_bytes),
                        "output_truncated": bool(result.get("output_truncated"))
                        or len(result["output"].encode()) > self.output_bytes,
                        "changed": [],
                        "note": "Command interrupted; no edits imported.",
                    }
                box.snapshot(snapshot)
            # Validate everything before importing any source. Container programs
            # cannot persist tests, configuration, executables, symlinks or secrets.
            files = self.code_files(snapshot)
            for name, expected in self.original_hashes.items():
                if any(name == p or name.startswith(p + "/") for p in self.manifest["protected_paths"]):
                    path = snapshot / name
                    if not path.is_file() or sha(path.read_bytes()) != expected:
                        raise ValueError("Command changed protected task files; edits discarded")
            notes = snapshot / "PROJECT_NOTES.md"
            note_data = notes.read_bytes() if notes.is_file() else b""
            if len(note_data) > 16_000:
                raise ValueError("PROJECT_NOTES.md exceeds 16000 bytes; edits discarded")
            note_data.decode("utf-8")
            for name in before.keys() - files.keys():
                (source / name).unlink()
            for name, data in files.items():
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            (source / "PROJECT_NOTES.md").write_bytes(note_data)
        changed = self.record_change(agent, before, self.hashes(source), "exec")
        return {
            "status": result["status"],
            "exit_code": result["exit_code"],
            "output": clip(result["output"], self.output_bytes),
            "output_truncated": bool(result.get("output_truncated"))
            or len(result["output"].encode()) > self.output_bytes,
            "changed": changed[:8],
            "changed_count": len(changed),
            "notes_saved": bool(note_data),
        }

    def synchronize(self, team):
        """Fast-forward untouched files; retain local edits and their original bases."""
        source = self.source(team)
        local, public, base = self.code_files(source), self.code_files(self.public), self.bases[team]
        projected, next_base = dict(local), dict(base)
        conflicts = []
        for name in sorted(local.keys() | public.keys() | base.keys()):
            lh = sha(local[name]) if name in local else None
            ph = sha(public[name]) if name in public else None
            if lh == base.get(name):
                if name in public:
                    projected[name] = public[name]
                else:
                    projected.pop(name, None)
                if ph is None:
                    next_base.pop(name, None)
                else:
                    next_base[name] = ph
            elif lh == ph:
                if ph is None:
                    next_base.pop(name, None)
                else:
                    next_base[name] = ph
            elif ph != base.get(name):
                conflicts.append(name)
        if len(projected) > self.max_files or sum(map(len, projected.values())) > self.max_source_bytes:
            self.progress[team]["sync_blocked"] = "Free source capacity before receiving public changes."
            self.emit("checkpoint_sync_blocked", team=team, reason="Combined branch exceeds fixed limits")
            return
        for name in local.keys() - projected.keys():
            (source / name).unlink()
        for name, data in projected.items():
            if local.get(name) != data:
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        self.bases[team] = next_base
        self.progress[team].pop("sync_blocked", None)
        if projected != local:
            self.progress[team]["revision"] += 1
        self.progress[team]["conflicts"] = conflicts[:8]

    def finish_round(self, round_number, *, limit=2):
        """Admit a fixed number of changed teams, without choosing by test score."""
        teams = sorted(t for t in self.sources if self.hashes(self.sources[t]) != self.bases[t])
        if teams:
            offset = self.merge_cursor % len(teams)
            selected = (teams[offset:] + teams[:offset])[:limit]
            self.merge_cursor += len(selected)
        else:
            selected = []
        changed_any = False
        for team in selected:
            local = self.code_files(self.source(team))
            base, public = self.bases[team], self.hashes(self.public)
            changes = {
                n: local.get(n)
                for n in local.keys() | base.keys()
                if (sha(local[n]) if n in local else None) != base.get(n)
            }
            conflicts = [
                n
                for n, data in changes.items()
                if public.get(n) not in (base.get(n), sha(data) if data is not None else None)
            ]
            if conflicts:
                self.progress[team]["conflicts"] = sorted(conflicts)[:8]
                self.emit("checkpoint_conflict", team=team, paths=sorted(conflicts)[:8])
                continue
            public_files = self.code_files(self.public)
            projected = dict(public_files)
            for name, data in changes.items():
                if data is None:
                    projected.pop(name, None)
                else:
                    projected[name] = data
            if len(projected) > self.max_files or sum(map(len, projected.values())) > self.max_source_bytes:
                self.emit("checkpoint_rejected", team=team, reason="Combined artifact exceeds fixed limits")
                continue
            # Parsing bounded source is validation, never execution. Reject the
            # whole update transactionally and retain the team's private work.
            invalid = self.syntax_error(projected)
            if invalid:
                self.progress[team]["integration_feedback"] = (
                    "Fix Python syntax before integration: " + invalid
                )
                self.emit(
                    "checkpoint_rejected", team=team, reason="Invalid Python syntax", diagnostic=invalid
                )
                continue
            self.progress[team].pop("integration_feedback", None)
            actual_changes = {n: d for n, d in changes.items() if public_files.get(n) != d}
            for name, data in changes.items():
                target = self.public / name
                if data is None:
                    target.unlink(missing_ok=True)
                    base.pop(name, None)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    base[name] = sha(data)
            if not actual_changes:
                continue
            self.public_version += 1
            self.progress[team]["integrations"] += 1
            changed_any = True
            self.emit(
                "checkpoint_integrated",
                team=team,
                manager=self.managers.get(team),
                version=self.public_version,
                paths=sorted(changes),
                selection="rotating_conflict_checked",
            )
        if changed_any:
            self.public_check = self.check(self.public, f"checkpoint-{round_number:04d}")
            self.emit("checkpoint_test", version=self.public_version, **self.public_check)
        for team in self.sources:
            self.synchronize(team)

    def export_public(self, reason):
        destination = self.directory / "checkpoint-export"
        if not destination.exists():
            copy_source(self.public, destination)
            (self.directory / "checkpoint-export.json").write_text(
                json.dumps(
                    {
                        "reason": reason,
                        "version": self.public_version,
                        "selection": "latest integrated public checkpoint, independent of scores",
                        "source_hashes": self.hashes(self.public),
                    },
                    indent=2,
                )
                + "\n"
            )
            self.emit("checkpoint_exported", reason=reason, version=self.public_version)

    def bank_evidence(self, teams):
        return {
            "public_version": self.public_version,
            "public_test": self.brief(self.public_check),
            "applicants": [{"team": team, "progress": self.summary(team)} for team in teams],
        }
