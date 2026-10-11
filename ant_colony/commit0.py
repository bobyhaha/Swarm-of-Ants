"""Paged Commit0 source access, team-local edits, and isolated visible-test grading.

No model-written program or shell command executes on the host. Only implementation
Python files are editable; every check starts from the original scaffold and tests.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import xml.etree.ElementTree as ET

from hayekmas.base.env import BaseEnv
from .context import clip
from .coding_workspace import CodingWorkspace
from .sandbox import DockerSandbox, copy_source, run


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse_junit(text, exit_code):
    try:
        root = ET.fromstring(text)
        cases = list(root.iter("testcase"))
    except ET.ParseError:
        return {"status": "missing_test_report", "score": None}
    ids = [(c.get("classname"), c.get("name")) for c in cases]
    if not cases or len(set(ids)) != len(ids):
        return {"status": "invalid_test_report", "score": None}
    passed = sum(not any(c.find(k) is not None for k in ("failure", "error", "skipped")) for c in cases)
    errors = sum(c.find("error") is not None for c in cases)
    failures, seen = [], set()
    for case in cases:
        for kind in ("error", "failure"):
            node = case.find(kind)
            if node is None:
                continue
            message = clip(node.get("message") or node.text or "No failure detail", 400)
            identity = (kind, message)
            if identity in seen:
                continue
            seen.add(identity)
            failures.append(
                {"kind": kind, "test": clip(case.get("name", "unknown"), 120), "message": message}
            )
            if len(failures) == 4:
                break
        if len(failures) == 4:
            break
    # Runtime/setup/collection failure is missing quality, not an invented zero.
    scored = exit_code in (0, 1)
    return {
        "status": "scored" if scored else "test_error",
        "score": passed / len(cases) if scored else None,
        "passed": passed,
        "total": len(cases),
        "errors": errors,
        "failures": failures,
    }


class Commit0Workbench(CodingWorkspace, BaseEnv):
    schema = {
        "kind": "exec|read|patch|write|test|list",
        "command": "shell command in isolated /workspace; inspect/edit Python and PROJECT_NOTES.md; /public is prior checkpoint package",
        "path": "relative path; test may select one original tests/test_file.py, or omit for full suite",
        "start": 1,
        "lines": 100,
        "base_sha256": "copy exact current hash from coding.file_versions or read.sha256; new ONLY for a nonexistent file",
        "content": "complete replacement Python source for write",
        "old": "exact unique source substring for patch",
        "new": "replacement source substring for patch",
    }

    def __init__(self, task_path, directory, emit=None, *, output_bytes=6000):
        super().__init__()
        self.root = Path(task_path).resolve().parent
        self.manifest = json.loads(Path(task_path).read_text())
        if self.manifest.get("benchmark") != "commit0":
            raise ValueError("Expected a prepared Commit0 task")
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.seed = self.root / "seed"
        row_path = self.root / "dataset-row.json"
        self.row = json.loads(row_path.read_text())
        self.package = self.row["src_dir"].strip("/")
        self.safe_relative(self.package + "/x.py")
        if self.row["base_commit"] != self.manifest["source_revision"]:
            raise ValueError("Scaffold base revision mismatch")
        if digest((self.root / "SPEC.md").read_bytes()) != self.manifest["spec_sha256"]:
            raise ValueError("Specification hash mismatch")
        self.emit = emit or (lambda *a, **k: None)
        self.output_bytes = output_bytes
        self.results, self.actor_teams = {}, {}
        self.step_count = 0
        self.score = None
        self.terminated = False
        self.counter = 0
        self.final_check = None
        self.initial_check = None
        self.sources = {}
        self.initialize_workspace()
        self.original_hashes = {
            str(p.relative_to(self.seed)): digest(p.read_bytes())
            for p in self.seed.rglob("*")
            if p.is_file() and not p.is_symlink()
        }
        (self.directory / "provenance.json").write_text(
            json.dumps(
                {
                    "benchmark": "commit0",
                    "task": self.manifest["id"],
                    "base_commit": self.row["base_commit"],
                    "image": self.manifest["image"],
                    "seed_hashes": self.original_hashes,
                    "evaluation": "adapted offline visible tests; not official Modal evaluator parity",
                },
                indent=2,
            )
            + "\n"
        )

    def initialize(self):
        pass

    def get_task_description(self):
        return "Implement the Commit0 library from its scaffold and visible specification/tests."

    def get_terminal_score(self):
        return self.score

    @staticmethod
    def safe_relative(value):
        if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
            raise ValueError("Invalid relative path")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or any(p.startswith(".") for p in path.parts):
            raise ValueError("Path must remain inside the public task")
        return path

    def source(self, team):
        if team not in self.sources:
            source = self.directory / "teams" / team / "source"
            copy_source(self.public, source)
            self.sources[team] = source
            self.initialize_team(team, source)
        return self.sources[team]

    def observation(self, agent):
        if self.actor_teams.get(agent.name) != agent.team:
            self.results.pop(agent.name, None)
            self.recent.pop(agent.name, None)
        self.actor_teams[agent.name] = agent.team
        self.source(agent.team)
        return {
            "instructions": "You are in a CODING WORK turn with working tools. Make a concrete implementation or verification step. "
            "Use exec for targeted shell reads, Python edit scripts, and persistent PROJECT_NOTES.md in /workspace. "
            "For write/patch, copy the exact current file hash from file_versions or a read result. "
            "Never invent a hash or use 'new' for an existing file. Exec editing scripts do not require a hash. "
            "Record implemented behavior, evidence, file ownership, and the exact NEXT action there. "
            "Recent receipts and a notes excerpt appear in context when they fit; the full notes persist on disk. "
            "When output_truncated is true, request targeted lines or symbols rather than dumping many files. "
            "Keep edits in a separate exec call from long tests; an interrupted command cannot import edits. "
            "Do not keep relisting the repository or claim tools are absent. Use patch for a unique exact replacement. "
            "Private files are shared inside your team. Divide files among workers to avoid conflicts. "
            "At round boundaries, a bounded number of teams' changed files enter the public checkpoint if base hashes match. "
            "Untouched files then fast-forward to that checkpoint; conflicting local edits are retained, never overwritten. "
            "Read /public in exec to resolve a conflict explicitly. New teams start from the public checkpoint. "
            "One tool per work turn; results arrive next turn. start and lines must be JSON integers, never strings. Read SPEC.md and original visible tests as needed. "
            f"Read pages contain at most 100 lines and {self.output_bytes} UTF-8 bytes; larger requested pages are capped. "
            "Use the returned next_start to continue reading, and retain useful findings in memory or the scratchpad. "
            "baseline describes ONLY the original untouched scaffold; it is not a current test after edits. "
            "result is your latest actual tool output; use test to check the current team source. "
            "Only Python files under "
            + self.package
            + "/ and PROJECT_NOTES.md persist from exec. No network or host access. "
            "test runs original tests in a fresh isolated container. Infrastructure errors are unscored. "
            "A root may submit early. At the deadline the integrated public checkpoint is automatically exported and graded. "
            "Closing a team returns its funds; it is NOT a submission. Prefer completing and testing code over further recruitment near the deadline. "
            "Do not claim success without evidence.",
            "baseline": self.brief(self.initial_check),
            "public_version": self.public_version,
            "public_test": self.brief(self.public_check),
            "progress": self.summary(agent.team),
            # A bounded inventory of actual versions avoids an extra read just
            # to obtain the optimistic-write token after inspecting with exec.
            # Long/unlisted paths remain available through explicit paged reads.
            "file_versions": {
                name: version
                for name, version in sorted(self.hashes(self.source(agent.team)).items())[:16]
                if len(name.encode("utf-8")) <= 240
            },
            "inventory": sorted(
                n for n in self.original_hashes if n.startswith((self.package + "/", "tests/"))
            )[:24],
            "project_notes": clip((self.source(agent.team) / "PROJECT_NOTES.md").read_text(), 1600)
            if (self.source(agent.team) / "PROJECT_NOTES.md").exists()
            else "No notes yet. Create a concrete plan and update it after edits.",
            "previous_tool": list(self.recent.get(agent.name, []))[-2:-1],
            "result_action": {
                k: v for k, v in list(self.recent.get(agent.name, []))[-1].items() if k != "result"
            }
            if self.recent.get(agent.name)
            else None,
            "result": self.bounded_result(self.results.get(agent.name)),
            "team": agent.team,
        }

    def execute(self, agent, action):
        try:
            return self._execute(agent, action)
        except (OSError, UnicodeError) as exc:
            raise ValueError("Task file unavailable or not UTF-8: " + type(exc).__name__) from None

    def writable_path(self, rel):
        return str(rel).startswith(self.package + "/") and rel.suffix == ".py"

    def _execute(self, agent, action):
        if not isinstance(action, dict):
            raise ValueError("Tool must be an object")
        self.counter += 1
        source = self.source(agent.team)
        kind = action.get("kind")
        if kind == "exec":
            result = self.execute_command(agent, action)
        elif kind == "list":
            start = action.get("start", 1)
            if type(start) is not int or start < 1:
                raise ValueError("List start must be positive")
            prefix = action.get("path")
            if prefix in (None, "", "."):
                directory = source
            else:
                directory = source / self.safe_relative(prefix)
                if not directory.resolve().is_relative_to(source.resolve()) or not directory.is_dir():
                    raise ValueError("List path must be a task directory")
            files = (["SPEC.md"] if directory == source else []) + sorted(
                str(p.relative_to(source))
                for p in directory.rglob("*")
                if p.is_file() and not any(x.startswith(".") for x in p.relative_to(source).parts)
            )
            result = {
                "files": files[start - 1 : start + 29],
                "next_start": start + 30 if start + 29 < len(files) else None,
            }
        elif kind in {"read", "write", "patch"}:
            rel = self.safe_relative(action.get("path"))
            path = self.root / "SPEC.md" if str(rel) == "SPEC.md" else source / rel
            if path.is_symlink() or (
                str(rel) != "SPEC.md" and not path.resolve().is_relative_to(source.resolve())
            ):
                raise ValueError("Unsafe source path")
            if kind == "read":
                start, lines = action.get("start", 1), action.get("lines", 100)
                if type(start) is not int or type(lines) is not int or start < 1 or lines < 1:
                    raise ValueError(
                        "Read start and lines must be positive JSON integers; pages contain at most 100 lines"
                    )
                lines = min(lines, 100)
                data = path.read_bytes()
                if len(data) > 1_000_000:
                    raise ValueError("Read file exceeds artifact limit")
                rows = data.decode("utf-8").splitlines()
                # Preserve whole lines and an exact continuation; never silently skip a clipped line.
                page, used = [], 0
                for i in range(start - 1, min(len(rows), start - 1 + lines)):
                    line = f"{i + 1}: {rows[i]}"
                    if used + len(line.encode()) + 1 > self.output_bytes:
                        break
                    page.append(line)
                    used += len(line.encode()) + 1
                if not page and start <= len(rows):
                    raise ValueError("Single line exceeds page byte cap")
                result = {
                    "path": str(rel),
                    "sha256": digest(data),
                    "text": "\n".join(page),
                    "next_start": start + len(page) if start + len(page) <= len(rows) else None,
                    "total_lines": len(rows),
                }
            else:
                if not self.writable_path(rel):
                    raise ValueError("Only implementation source files may be changed")
                before_hashes = self.hashes(source)
                before = path.read_bytes() if path.exists() else None
                current_hash = digest(before) if before is not None else "new"
                if action.get("base_sha256") != current_hash:
                    raise ValueError("Stale file hash; read the current team version before writing")
                if kind == "patch":
                    old, new = action.get("old"), action.get("new")
                    if before is None or not isinstance(old, str) or not old or not isinstance(new, str):
                        raise ValueError("patch requires an existing file, nonempty old text, and new text")
                    original = before.decode("utf-8")
                    if original.count(old) != 1:
                        raise ValueError("Patch text must match exactly once; read the current source")
                    content = original.replace(old, new, 1)
                else:
                    content = action.get("content")
                if not isinstance(content, str) or len(content.encode()) > 64_000:
                    raise ValueError("Write exceeds source size limit")
                total = (
                    sum(len(data) for data in self.code_files(source).values())
                    - len(before or b"")
                    + len(content.encode())
                )
                if total > self.max_source_bytes or (before is None and len(before_hashes) >= self.max_files):
                    raise ValueError("Implementation artifact exceeds file or byte limit")
                # Parsing generated code is unnecessary: it only executes in a Docker check.
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                result = {
                    "path": str(rel),
                    "sha256": digest(path.read_bytes()),
                    "written_bytes": len(content.encode()),
                }
                self.record_change(agent, before_hashes, self.hashes(source), kind)
        elif kind == "test":
            result = self.check(source, f"work-{self.counter:05d}", test_path=action.get("path"))
        else:
            raise ValueError("Unknown coding tool")
        self.results[agent.name] = result
        self.remember_tool(agent, action, result)
        self.emit("coding_tool", agent=agent.name, team=agent.team, kind=kind, result=result)
        return result

    def check(self, source, label, *, test_path=None):
        if test_path:
            rel = self.safe_relative(test_path)
            if not (
                str(rel).startswith("tests/")
                and rel.name.startswith("test_")
                and rel.suffix == ".py"
                and str(rel) in self.original_hashes
            ):
                raise ValueError("Select an existing original test module")
        report = self.directory / "checks" / label
        report.mkdir(parents=True, exist_ok=False)
        # Fresh scaffold: tests, build settings and all non-source files are trusted.
        prepared = report / "source"
        copy_source(self.seed, prepared)
        # Preserve original tests/assets, but do not resurrect implementation
        # files intentionally removed by an agent's command.
        for path in (prepared / self.package).rglob("*.py"):
            path.unlink()
        for p in (Path(source) / self.package).rglob("*.py"):
            if p.is_symlink() or not p.resolve().is_relative_to(Path(source).resolve()):
                raise ValueError("Unsafe submitted source")
            dest = prepared / p.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
        with DockerSandbox(self.manifest["image"], timeout=60, memory="768m", cpus="1") as box:
            box.load(prepared)
            for name in self.manifest["protected_paths"]:
                rel = self.safe_relative(name)
                box.put(self.seed / rel, "/workspace/" + str(rel), readonly=True)
            build = box.execute(self.manifest["build_command"], timeout=120)
            (report / "build.json").write_text(json.dumps(build, indent=2))
            if build["exit_code"]:
                result = {"status": "build_failed", "score": None, "diagnostic": clip(build["output"], 2500)}
            else:
                command = (
                    ("pytest " + shlex.quote(test_path))
                    if test_path
                    else self.manifest["visible_test_command"]
                )
                # Hundreds of recursive failures can exhaust the output cap while
                # pytest prints tracebacks, destroying an otherwise useful report.
                # Shorten presentation only; preserve the full original test selection.
                command += " --tb=line --junitxml=/tmp/colony-results.xml"
                test = box.execute(command, timeout=120)
                (report / "test-command.json").write_text(json.dumps(test, indent=2))
                xml = report / "junit.xml"
                copied = run(["docker", "cp", box.name + ":/tmp/colony-results.xml", str(xml)], check=False)
                raw = xml.read_text() if copied.returncode == 0 and xml.stat().st_size < 8_000_000 else ""
                result = parse_junit(raw, test["exit_code"])
                result["diagnostic"] = clip(test["output"][-3500:], 3500)
        result["report"] = str(report.relative_to(self.directory))
        result["scope"] = test_path or "full original visible suite"
        (report / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result

    def apply(self, action):
        if self.terminated:
            raise RuntimeError("Final benchmark evaluation already performed")
        team = self.actor_teams[action.author] if action.author != "__public__" else "public"
        source = self.sources[team] if team != "public" else self.public
        # Save the selected artifact BEFORE grading; never choose by hidden scores.
        copy_source(source, self.directory / "submission")
        self.final_check = self.check(source, "final")
        self.emit("commit0_final", team=team, author=action.author, **self.final_check)
        if self.final_check["score"] is None:
            raise RuntimeError("Commit0 final evaluation unscored: " + self.final_check["status"])
        self.score = self.final_check["score"]
        self.step_count = 1
        self.terminated = True
        return self.score
