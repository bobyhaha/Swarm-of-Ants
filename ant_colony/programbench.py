"""ProgramBench cleanroom workspaces; only terminal evaluation sees hidden tests."""

import hashlib
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import shlex

from hayekmas.base.env import BaseEnv
from .commit0 import Commit0Workbench
from .context import clip
from .sandbox import DockerSandbox, copy_source, run, safe_extract
from . import programbench_probes as probes


PIN = "27f02157c785f8da3647aa6dbbe6b9137f99f10e"
IGNORED = {".git", ".venv", "__pycache__", ".pytest_cache", "target", "build", "dist", "node_modules"}


class ProgramBenchWorkbench(Commit0Workbench):
    schema = {
        "kind": "exec|read|write|patch|list|test|probe",
        "command": "isolated shell in /workspace; read documentation at /reference-docs; prior source at /public",
        "path": "relative source path or SPEC.md",
        "start": 1, "lines": 100,
        "base_sha256": "exact file_versions/read hash, or new only for a new file",
        "content": "complete replacement source text",
        "old": "unique exact substring", "new": "replacement text",
        "args": ["--help"], "stdin": "", "files": {}, "capture": [],
    }

    def __init__(self, task_path, directory, evaluator_python, emit=None, *, output_bytes=16000):
        BaseEnv.__init__(self)
        self.root = Path(task_path).resolve().parent
        self.manifest = json.loads(Path(task_path).read_text())
        self.data = self.manifest
        if self.manifest.get("benchmark") != "programbench" or self.manifest["source_revision"] != PIN:
            raise ValueError("Expected the pinned prepared ProgramBench task")
        if hashlib.sha256((self.root / "SPEC.md").read_bytes()).hexdigest() != self.manifest["spec_sha256"]:
            raise ValueError("Specification hash mismatch")
        # Do not resolve a virtualenv interpreter symlink: it needs its venv path.
        self.evaluator_python = str(Path(evaluator_python).absolute())
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.seed, self.package = self.root / "seed", ""
        self.emit = emit or (lambda *a, **k: None)
        self.output_bytes = output_bytes
        self.results, self.actor_teams, self.sources = {}, {}, {}
        self.step_count = self.counter = 0
        self.score = self.final_check = self.initial_check = None
        self.terminated = False
        self.grade_on_budget_stop = True
        self.bank_finalization = False
        self.quality_first = False
        self.probe_bank = []
        self.initialize_workspace()
        self.original_hashes = self.hashes(self.seed)
        self._write("provenance.json", {"benchmark": "programbench", "task": self.manifest["id"],
                    "source_revision": PIN, "image": self.manifest["image"],
                    "seed_hashes": self.original_hashes, "hidden_tests": "terminal only"})

    def _write(self, name, data):
        p = self.directory / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2) + "\n")

    def writable_path(self, rel):
        return (not any(p in IGNORED for p in rel.parts)
                and rel.name not in {"SPEC.md", "PROJECT_NOTES.md", "executable"}
                and rel.suffix not in {".o", ".a", ".so", ".pyc", ".exe", ".gz", ".zip"})

    @staticmethod
    def safe_relative(value):
        if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
            raise ValueError("Invalid relative path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or any(p in IGNORED for p in path.parts):
            raise ValueError("Path must remain inside the implementation source")
        return path

    def code_files(self, directory):
        root = Path(directory)
        files = {}
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if not self.writable_path(rel):
                continue
            if p.is_symlink() or not p.resolve().is_relative_to(root.resolve()):
                raise ValueError("Unsafe implementation path")
            if not p.is_file():
                continue
            self.safe_relative(str(rel))
            data = p.read_bytes()
            data.decode("utf-8")
            if len(data) > self.max_file_bytes:
                raise ValueError("Implementation file exceeds 64000 bytes")
            files[str(rel)] = data
        if len(files) > self.max_files or sum(map(len, files.values())) > self.max_source_bytes:
            raise ValueError("Implementation artifact exceeds fixed bounds")
        return files

    def get_task_description(self):
        return "Reconstruct the program using its documentation and black-box behavior; hidden tests are final-only."

    def sandbox(self, source, *, reference=True):
        box = DockerSandbox(self.manifest["image"], programbench=reference, timeout=60, memory="1g", cpus="1")
        try:
            box.load(source)
            box.put(self.root / "SPEC.md", "/workspace/SPEC.md", readonly=True)
            evidence = self.directory / "public-probes.json"
            if evidence.exists():
                box.put(evidence, "/evidence/probes.json", readonly=True)
            box.root("if test -f /workspace/compile.sh; then chmod a+x /workspace/compile.sh; fi")
            return box
        except BaseException:
            box.close()
            raise

    def observation(self, agent):
        if self.actor_teams.get(agent.name) != agent.team:
            self.results.pop(agent.name, None)
            self.recent.pop(agent.name, None)
        self.actor_teams[agent.name] = agent.team
        source = self.source(agent.team)
        note = source / "PROJECT_NOTES.md"
        note_data = note.read_bytes() if note.exists() else b""
        return {
            "instructions": (
                "Rebuild independently from documentation in /reference-docs. Use probe with args/stdin/files/capture "
                "to query the execute-only reference normally; the reference is deliberately absent from exec/build containers. "
                "Never decompile, trace, copy or wrap the reference. No internet or original source. "
                "Implement /workspace/compile.sh which builds ./executable from source using installed offline tools. "
                "exec changes source and notes transactionally; compiled outputs and build directories do not persist. "
                "All UTF-8 source/configuration files may persist within 128 files, 64000 bytes/file, 2MB total. "
                "write/patch require an exact current hash. Use targeted reads when output is truncated. "
                "Keep PROJECT_NOTES.md below 16000 bytes: replace/compact it, do not endlessly append. "
                "Record evidence, owned interfaces, and the next action. One tool per turn; receipts arrive next turn. "
                "Team members share files. Root teams integrate conflict-free changes between rounds; /public is read-only. "
                "test builds fresh with NO reference binary and compares agent-generated reproducible probes only. "
                "A probe score is diagnostic, not the hidden benchmark score. Native hidden tests run only after final selection. "
                "Reuse the bounded public probe bank at /evidence/probes.json (when present) through targeted exec reads. "
                "Prefer small concrete implementation steps and focused verification over repeated inventory. "
                "Managers can hire/spawn when useful; any model can manage or execute. "
                + ("A root reports team readiness; only the bank can finalize the integrated public artifact. "
                   if self.bank_finalization else "A root can submit early. ")
                + "The public artifact is selected automatically at the deadline or dollar stop. "
                + ("Priority: maximize correctness and completeness within the hard episode cap. "
                   "Actual API dollars are recorded; unused budget is not an objective. "
                   if self.quality_first else "")
            ),
            "team": agent.team, "public_version": self.public_version,
            "public_test": self.brief(self.public_check), "progress": self.summary(agent.team),
            "file_versions": {n:h for n,h in list(self.hashes(source).items())[:16] if len(n.encode()) <= 240},
            "notes_bytes": len(note_data), "notes_limit_bytes": 16000,
            "project_notes": clip(note_data.decode(), 1600),
            "probe_count": len(self.probe_bank), "probe_limit": 32,
            "recent_probe_ids": [r["id"] for r in self.probe_bank[-4:]],
            "previous_tool": list(self.recent.get(agent.name, []))[-2:-1],
            "result": self.bounded_result(self.results.get(agent.name)),
        }

    def inspect_public(self, action):
        """Read-only sponsored discovery; never exposes the reference or hidden tests."""
        if not isinstance(action, dict) or set(action) - {"kind", "path", "start"}:
            raise ValueError("Inspection supports kind, path and start only")
        kind, path, start = action.get("kind"), action.get("path"), action.get("start", 1)
        if kind not in {"list", "read"} or type(start) is not int or not 1 <= start <= 100000:
            raise ValueError("Inspection requires list/read and a bounded positive start")
        if path == "SPEC.md":
            path = "/workspace/SPEC.md"
        elif isinstance(path, str) and (path in {"/reference-docs", "/public"}
                or path.startswith(("/reference-docs/", "/public/"))):
            if ".." in PurePosixPath(path).parts or "\x00" in path or "\\" in path or len(path) > 512:
                raise ValueError("Invalid inspection path")
        else:
            raise ValueError("Inspect only SPEC.md, /reference-docs or /public")
        # Resolve symlinks inside the isolated container before enforcing the roots.
        command = 'p=$(realpath -- ' + shlex.quote(path) + '); case "$p" in /reference-docs|/reference-docs/*|/public|/public/*|/workspace/SPEC.md) ;; *) exit 64;; esac; '
        command += (f'find "$p" -type f | sort | sed -n \'{start},{start + 29}p\'' if kind == "list"
                    else f'sed -n \'{start},{start + 99}p\' "$p"')
        with self.sandbox(self.public, reference=False) as box:
            box.put(self.public, "/public", readonly=True)
            result = box.execute(command, timeout=30)
        return {"path": path, "start": start, "kind": kind, "exit_code": result["exit_code"],
                "text": clip(result.get("output", ""), 6000),
                "next_start": start + (30 if kind == "list" else 100),
                "scope": "public documentation/source only; no writes imported"}

    def execute(self, agent, action):
        if isinstance(action, dict) and action.get("kind") == "probe":
            if len(self.probe_bank) >= 32:
                raise ValueError("Probe bank is full; use existing evidence and test the implementation")
            row = probes.discover(self, action)
            if not any(r["id"] == row["id"] for r in self.probe_bank):
                self.probe_bank.append(row)
                self._write("public-probes.json", self.probe_bank)
                self.emit("probe_discovered", agent=agent.name, team=agent.team, probe=row)
            result = {"status": "reference_observation", "id": row["id"], **probes.readable_observation(row["expected"])}
            self.results[agent.name] = result
            self.remember_tool(agent, action, result)
            self.emit("coding_tool", agent=agent.name, team=agent.team, kind="probe", result=result)
            return result
        return super().execute(agent, action)

    def execute_command(self, agent, action):
        command = action.get("command")
        if not isinstance(command, str) or not command.strip() or len(command.encode()) > 32000:
            raise ValueError("exec requires a nonempty command of at most 32000 bytes")
        source = self.source(agent.team)
        before = self.hashes(source)
        with tempfile.TemporaryDirectory(prefix="colony-programbench-") as temp:
            snapshot = Path(temp) / "snapshot"
            with self.sandbox(source, reference=False) as box:
                box.put(self.public, "/public", readonly=True)
                result = box.execute(command, timeout=120)
                if result["status"] != "ok" or result["exit_code"] in (124,137):
                    return {**result, "changed": [], "note": "Interrupted command; no edits imported"}
                blob = run(["docker", "exec", "--user", "65534:65534", box.name, "tar",
                            *["--exclude=" + n for n in sorted(IGNORED | {"executable"})],
                            "-C", "/workspace", "-czf", "-", "."], timeout=60).stdout
                safe_extract(blob, snapshot)
            files = self.code_files(snapshot)
            notes = snapshot / "PROJECT_NOTES.md"
            note_data = notes.read_bytes() if notes.exists() else b""
            note_data.decode("utf-8")
            if len(note_data) > 16000:
                raise ValueError("Notes exceed 16000 bytes; compact/replace them. No edits imported.")
            for name in before.keys() - files.keys():
                (source / name).unlink()
            for name, data in files.items():
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                if name == "compile.sh":
                    target.chmod(0o755)
            (source / "PROJECT_NOTES.md").write_bytes(note_data)
        changed = self.record_change(agent, before, self.hashes(source), "exec")
        return {**result, "changed": changed[:8], "changed_count": len(changed), "notes_bytes": len(note_data)}

    def check(self, source, label, *, test_path=None):
        if test_path:
            raise ValueError("ProgramBench test takes no path; hidden tests are final-only")
        report = self.directory / "checks" / label
        report.mkdir(parents=True, exist_ok=False)
        with self.sandbox(source, reference=False) as box:
            build = box.execute(self.manifest["build_command"], timeout=180)
            (report / "build.json").write_text(json.dumps(build, indent=2))
            result = {"status": "build_failed", "score": None, "exit_code": build["exit_code"],
                      "diagnostic": clip(build["output"] or "Build command failed: " + self.manifest["build_command"], 3000)}
            if build["status"] == "ok" and build["exit_code"] == 0:
                passed, failures = 0, []
                for row in self.probe_bank:
                    try:
                        observed = probes.observation(box, row["input"], "/workspace/executable")
                        ok = observed == row["expected"]
                    except (ValueError, json.JSONDecodeError):
                        ok = False
                    passed += ok
                    if not ok and len(failures) < 4:
                        failures.append({"probe": row["id"], "input": row["input"],
                                         "expected": probes.readable_observation(row["expected"], limit=300)})
                result = {"status": "public_probe_score" if self.probe_bank else "build_only", "build": True,
                          "score": passed / len(self.probe_bank) if self.probe_bank else None,
                          "passed": passed, "total": len(self.probe_bank), "failures": failures}
        result.update(report=str(report.relative_to(self.directory)), scope="agent-generated probes; not hidden tests")
        (report / "result.json").write_text(json.dumps(result, indent=2))
        return result

    def apply(self, action):
        if self.terminated:
            raise RuntimeError("Final evaluation already performed")
        source = self.public if action.author == "__public__" else self.sources[self.actor_teams[action.author]]
        root = self.directory / "submission"
        target = root / self.manifest["id"]
        target.mkdir(parents=True, exist_ok=False)
        copy_source(source, target / "source")
        with tarfile.open(target / "submission.tar.gz", "w:gz") as archive:
            for name in sorted(self.code_files(source)):
                info = archive.gettarinfo(str(source / name), arcname=name)
                info.mode = 0o755 if name == "compile.sh" else 0o644
                with (source / name).open("rb") as handle:
                    archive.addfile(info, handle)
        self._write("selected-artifact.json", {"selection": action.author, "hashes": self.hashes(source)})
        environment = {k:v for k,v in os.environ.items() if k not in {"OPENROUTER_API_KEY", "OPENAI_API_KEY"}}
        native = [self.evaluator_python, "-c", "from programbench.cli.main import app; app()"]
        # Native branch evaluation creates substantial Docker work. Serialize it
        # across episodes; an infrastructure failure must not terminate peers.
        lock_path = Path(os.environ.get("ANT_COLONY_NATIVE_LOCK", str(self.directory.parent / "native.lock")))
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.emit("native_evaluation_waiting", artifact=str(root))
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.emit("native_evaluation_started", artifact=str(root))
            result = run([*native, "eval", str(root), "--workers", "1", "--branch-workers", "1", "--docker-cpus", "1"],
                         timeout=21600, check=False, env=environment)
        (root / "native-eval.log").write_bytes(result.stdout + result.stderr)
        if result.returncode:
            self.final_check = {"status": "native_evaluator_error", "score": None}
        else:
            scorer = Path(__file__).with_name("programbench_score.py")
            script = ("import importlib.util,json; "
                      "s=importlib.util.spec_from_file_location('scorer'," + repr(str(scorer)) + "); "
                      "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                      "print(json.dumps(m.summarize(" + repr(str(root)) + "," + repr(self.manifest["id"]) + ")))")
            scored = run([self.evaluator_python, "-c", script], timeout=120, check=False, env=environment)
            (root / "scorer.log").write_bytes(scored.stdout + scored.stderr)
            self.final_check = json.loads(scored.stdout) if not scored.returncode else {"status": "native_summary_error", "score": None}
        self._write("final-result.json", self.final_check)
        self.emit("programbench_final", author=action.author, **self.final_check)
        if self.final_check["score"] is None:
            raise RuntimeError("ProgramBench final evaluation unscored: " + self.final_check["status"])
        self.score = self.final_check["score"]
        self.step_count, self.terminated = 1, True
        return self.score
