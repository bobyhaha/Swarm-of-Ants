"""Grant-funded replacement for HayekMAS._run_auction_action_loop.

No model sees this object, a full population, an environment reference answer,
or the ledger mutators. Models receive bounded JSON and propose typed actions.
"""

from collections import deque
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
import json
import random

from hayekmas.base.population import Population

from .context import build_context, byte_size, clip, messages as context_messages
from .board import TaskBoard, WORKER_BOARD_RULES
from .environment import ExactEnv, evaluate, research_env
from .ledger import Ledger, amount
from .protocol import RULES, HIERARCHY_RULES, SCHEMAS, peer_graph
from .funding import FundingProtocol
from .providers import BudgetStop, UncertainCall
from .state import ColonyAgent, Team
from .quality import QualityDiscovery, rules as quality_rules
from .adaptation import RENEWAL_SCHEMA, RENEWAL_RULES


class ColonyMAS(QualityDiscovery, FundingProtocol):
    def __init__(self, config, task, provider, emit=None, *, workbench=None):
        self.config, self.task, self.provider = config, task, provider
        if hasattr(provider, "request_pricing"):
            provider.request_pricing = config.request_pricing
        self.workbench = workbench
        if (
            hasattr(provider, "models")
            and (set(config.worker_models) | {config.bank_model}) - provider.models.keys()
        ):
            raise ValueError("Missing configured bank/worker model alias")
        if len(task.problem.encode("utf-8")) > config.task_bytes:
            raise ValueError("Task exceeds task_bytes; configure a larger constant before starting")
        self.sink = emit or (lambda *args, **kwargs: None)
        self.round = 0
        self.phase = "initialize"
        if self.workbench is not None:
            self.workbench.emit = self.emit
            self.workbench.quality_first = config.quality_first
        self.population = Population()
        self.agents = {}
        self.teams = {}
        self.next_agent = self.next_team = 1
        self.ledger = Ledger(config.initial_bank, self.emit)
        self.trails = deque(maxlen=config.trail_size)
        self.bank_memory = ""
        self.board = TaskBoard(config.bank_board_size, detail_bytes=900 if config.quality_first else 240)
        self.format_errors = {}
        self.bank_brief = "Propose concrete, verifiable work with a budget and stopping rule."
        self.application_spent = 0
        self.discovery_spent = 0
        self.proposal_cursor = 0
        self.calls = self.output_tokens = self.estimated_tokens = 0
        self.max_context_seen = 0
        self.final_answer = self.final_author = None
        self.submission_mode = None
        self.score = None
        self.status = "ready"
        self.uncertain_credits = 0
        self.eval_reserve = self.call_budget(config.bank_model) if task.kind == "research" else 0
        self.env = (
            workbench
            if workbench is not None
            else (research_env(task, self.judge) if task.kind == "research" else ExactEnv(task))
        )
        if task.kind in {"commit0", "programbench"} and workbench is None:
            raise ValueError("Coding tasks require an isolated coding workbench")
        if task.kind == "research":
            from hayekmas.adapters.researchworld.prompts import RUBRIC_JUDGE_PROMPT

            minimum = (
                RUBRIC_JUDGE_PROMPT.replace("<<<problem>>>", task.problem)
                .replace("<<<rubric>>>", task.answer)
                .replace("<<<actual_output>>>", "")
            )
            # Reject a rubric that cannot even fit before making the first paid call.
            build_context(
                "Evaluate using the rubric. Follow the requested SCORE/REASON format.",
                {"evaluation": minimum},
                [],
                config.context_bytes,
            )
        initial = list(config.initial_models) if config.initial_models else [
            config.worker_models[i % len(config.worker_models)] for i in range(config.initial_agents)
        ]
        if config.initial_models:
            random.Random(config.population_seed).shuffle(initial)
        for model in initial:
            self.new_agent(
                model,
                ("Find useful work, investigate uncertainty, collaborate and improve the integrated solution."
                 if config.quality_first else "Find useful work; report uncertainty; recruit only when it improves expected net reward."),
            )

    def emit(self, event, **data):
        self.sink(event, **{"round": self.round, "phase": self.phase, **data})

    def new_agent(self, model, strategy, parent=None, allowance=0, assignment=""):
        cfg = self.config
        if len(self.agents) >= cfg.max_agents or model not in cfg.worker_models:
            raise ValueError("Population cap or model allowlist violation")
        if parent:
            self.check_child_slot(parent)
            team = self.teams[parent.team]
            amount(allowance)
            if not team.funded or allowance > team.allowances[parent.name]:
                raise ValueError("Spawning requires an existing funded allowance")
            if allowance < self.working_minimum(model) * (2 if cfg.funding_recovery else 1):
                raise ValueError("Child must receive enough rights for one bounded call")
        agent = ColonyAgent(f"a{self.next_agent:04d}", model, clip(strategy, cfg.strategy_bytes), self.round)
        self.next_agent += 1
        if assignment:
            agent.assignment = clip(assignment, cfg.text_bytes)
        self.agents[agent.name] = agent
        self.population.add_agent(agent)
        if parent:
            agent.parent = parent.name
            agent.team = parent.team
            parent.children.append(agent.name)
            team.members.add(agent.name)
            team.allowances[parent.name] -= allowance
            team.allowances[agent.name] = allowance
            agent.parent_agent_id, agent.parent_agent_name = parent.id, parent.name
            agent.father_agent_id, agent.father_agent_name = parent.id, parent.name
            agent.spawn_method = "team_funded"
            agent.root_ancestor_class = parent.root_ancestor_class
            self.recruitment_completed(parent)
        self.emit(
            "spawn",
            agent=agent.name,
            parent=agent.parent,
            team=agent.team,
            model=model,
            strategy=agent.trainable_system_prompt,
            allowance=allowance,
        )
        return agent

    def depth(self, agent):
        depth, seen = 0, set()
        while agent.parent:
            if agent.name in seen:
                raise AssertionError("Recruitment cycle")
            seen.add(agent.name)
            agent = self.agents[agent.parent]
            depth += 1
        return depth

    def check_child_slot(self, parent):
        if len(parent.children) >= self.config.max_children or self.depth(parent) >= self.config.max_depth:
            raise ValueError("Recruitment degree/depth cap reached")

    def ensure_team(self, agent):
        if agent.team:
            return self.teams[agent.team]
        team = Team(f"t{self.next_team:04d}", agent.name, {agent.name}, {agent.name: 0})
        self.next_team += 1
        self.teams[team.id] = team
        agent.team = team.id
        self.ledger.balances[team.id] = 0
        if self.workbench is not None:
            self.workbench.register_team(team.id, agent.name)
        self.emit("team_created", team=team.id, manager=agent.name)
        return team

    def hire(self, parent, child, allowance, assignment):
        if self.phase != "formation" or parent is child or child.team is not None:
            raise ValueError("Hiring requires consent by an unteamed peer during formation")
        self.check_child_slot(parent)
        amount(allowance)
        if not parent.team or not self.teams[parent.team].funded or allowance < self.working_minimum(child.model):
            raise ValueError("Hiring needs funded working capital")
        team = self.teams[parent.team]
        if allowance > team.allowances[parent.name]:
            raise ValueError("Hiring cannot mint spending rights")
        child.team, child.parent = team.id, parent.name
        parent.children.append(child.name)
        team.members.add(child.name)
        team.allowances[child.name] = allowance
        team.allowances[parent.name] -= allowance
        child.assignment = clip(assignment, self.config.text_bytes)
        self.clear_introduction(child, "hired")
        self.recruitment_completed(parent)
        # Clear old team/discovery input at a membership boundary.
        child.inbox, child.read_keys, child.memory = [], [], ""
        child.wake_round = self.round
        self.emit("hire", parent=parent.name, child=child.name, team=team.id, allowance=allowance)

    def leave(self, agent):
        if self.phase != "formation":
            raise ValueError("Membership is frozen during work")
        if agent.children:
            raise ValueError("Only leaves may leave; a manager can close its team")
        if not agent.team:
            return
        team = self.teams[agent.team]
        if team.manager == agent.name:
            self.close_team(team)
            return
        parent = self.agents[agent.parent]
        parent.children.remove(agent.name)
        team.allowances[parent.name] += team.allowances.pop(agent.name)
        team.members.remove(agent.name)
        team.scratchpad.pop(agent.name, None)
        agent.team = agent.parent = None
        agent.application, agent.opening = None, None
        agent.standing_grant = None
        agent.recruitment_plan = None
        agent.inbox, agent.read_keys, agent.memory = [], [], ""
        self.emit("leave", agent=agent.name, team=team.id)

    def close_team(self, team):
        released = self.board.release_team(team.id)
        if released:
            self.emit("board_released", team=team.id, tasks=released)
        value = self.ledger.available(team.id)
        self.ledger.transfer(team.id, "bank", value, "grant_return")
        del self.ledger.balances[team.id]
        for name in sorted(team.members):
            agent = self.agents[name]
            agent.team = agent.parent = None
            agent.application, agent.opening = None, None
            agent.standing_grant = None
            agent.recruitment_plan = None
            agent.children, agent.inbox, agent.read_keys, agent.memory = [], [], [], ""
            agent.wake_round = self.round
        del self.teams[team.id]
        self.emit("team_closed", team=team.id, returned=value)

    def grant(self, agent, value, task_id=None):
        amount(value)
        if self.board.enabled:
            reason = self.board.conflict(agent.name, task_id)
            if reason:
                raise ValueError(reason)
        if not 0 < value <= self.config.max_grant or value > self.bank_grantable():
            raise ValueError("Grant exceeds request limits or spendable treasury")
        team = self.ensure_team(agent)
        if team.manager != agent.name:
            raise ValueError("Only team managers can obtain grants")
        self.ledger.transfer("bank", team.id, value, "grant")
        team.allowances[agent.name] += value
        team.ready = None
        team.funded = True
        team.expires = self.round + self.config.grant_rounds - 1
        if self.board.enabled:
            self.board.claim(task_id, agent.name, team.id, self.round)
            team.board_task_id = task_id
            item = self.board.items[task_id]
            agent.assignment = clip(
                item["scope"] + " Acceptance: " + item["acceptance"], self.config.text_bytes
            )
            self.emit("board_claimed", task_id=task_id, owner=agent.name, team=team.id)
        agent.wake_round = self.round
        self.emit("grant", team=team.id, manager=agent.name, amount=value, expires=team.expires)

    def bank_spendable(self):
        return max(0, self.ledger.available("bank") - self.eval_reserve)

    def neighbors(self, agent):
        return ([agent.parent] if agent.parent else []) + list(agent.children)

    def card(self, name):
        agent = self.agents[name]
        return {
            "id": name,
            "team": agent.team,
            "model": agent.model,
            "profile": clip(agent.trainable_system_prompt, self.config.text_bytes),
        }

    def objective(self):
        return {
            "maximum_terminal_reward": self.config.reward,
            "reward_rule": "One payment to the bank: maximum_terminal_reward * final normalized score [0,1].",
            "score_rule": "Fraction of original visible tests passed; missing evaluation earns no reward."
            if self.task.kind == "commit0"
            else "The task evaluator supplies the final normalized score.",
            "pricing": self.pricing_view(),
            "cost_per_generated_token": self.config.token_price if self.config.cost_mode == "output_tokens" else None,
            "rent_per_round_per_agent_and_bank": self.config.rent_per_round,
            "transfers_create_wealth": False,
        }

    def observation(self, agent, phase, *, peers=(), may_propose=False, extra=None):
        cfg = self.config
        coding_context = []
        team = self.teams.get(agent.team)
        required = {
            "phase": phase,
            "round": self.round,
            "max_rounds": cfg.max_rounds,
            "self": agent.name,
            "model": agent.model,
            "task": self.task.problem,
            "society_wealth": self.ledger.wealth,
            "objective": self.objective(),
            "team": agent.team,
            "parent": agent.parent,
            "children": list(agent.children),
            "depth": self.depth(agent),
            "manager": bool(team and team.manager == agent.name),
            "funded": bool(team and team.funded),
            "allowance": team.allowances[agent.name] if team else 0,
            "team_balance": self.ledger.available(agent.team),
            "worker_models": cfg.worker_models,
            "limits": {
                "children": cfg.max_children,
                "depth": cfg.max_depth,
                "population_cap": cfg.max_agents,
                "spawn_slots_remaining": cfg.max_agents - len(self.agents),
                "output_tokens": self.output_limit(self.work_kind(agent) if phase == "work" else phase),
                "token_price": cfg.token_price if cfg.cost_mode == "output_tokens" else None,
                "message_bytes": cfg.text_bytes,
                "sleep_rounds": cfg.max_sleep_rounds,
                "max_grant": cfg.max_grant,
                "minimum_working_budget": self.working_minimum(agent.model),
                "recommended_runway": self.call_budget(agent.model) * 4,
                "manager_contacts": cfg.manager_contacts,
                "peer_contacts": cfg.peer_contacts,
                "max_revisions": cfg.max_revisions,
                "grant_rounds": cfg.grant_rounds,
            },
            "assignment": agent.assignment,
            "strategy": agent.trainable_system_prompt,
            "bank_brief": self.bank_brief,
            "application": agent.application,
            "last_feedback": agent.last_feedback,
            "last_protocol_error": self.format_errors.get(agent.name),
            "opening": agent.opening,
            "schema": SCHEMAS[phase],
        }
        required["schema"] = deepcopy(SCHEMAS[phase])
        schema = required["schema"]
        if cfg.bank_finalization and phase == "work":
            schema.pop("submit", None)
            schema["ready"] = "root only: evidence and remaining gaps for YOUR TEAM, after seeing tool receipts"
            required["team_readiness"] = team.ready if team else None
        if self.board.enabled:
            required["bank_board"] = self.board.view()
            required["team_task_id"] = team.board_task_id if team else None
            required["work_authorized"] = bool(team and self.board.can_work(team.id))
            if "proposal" in schema and not agent.parent:
                schema["proposal"]["task_id"] = "listed open task ID, or your existing claim for a top-up"
        if "proposal" in schema:
            schema["proposal"]["amount"] = min(cfg.max_grant, self.call_budget(agent.model) * (16 if cfg.quality_first else 8))
        if "opening" in schema:
            schema["opening"]["allowance"] = (min(self.working_minimum(m) for m in cfg.worker_models) * 2
                                               if cfg.quality_first else self.call_credits * 3)
        if "recruitment_request" in schema:
            cheapest = min(cfg.worker_models, key=self.call_budget)
            schema["recruitment_request"].update(model=cheapest, allowance=2 * self.call_budget(cheapest))
        if "spawn" in schema:
            schema["spawn"]["allowance"] = self.call_credits * 4
        if "allocate" in schema:
            schema["allocate"][0]["amount"] = self.call_credits * 2
        if "decisions" in schema:
            schema["decisions"][0]["amount"] = self.call_credits * 2
            if cfg.standing_grants:
                schema["decisions"][0]["renewal"] = RENEWAL_SCHEMA
                required["standing_grant_rounds"] = cfg.standing_grant_rounds
        if extra:
            required.update(extra)
            if "proposals" in extra and extra["proposals"] and "decisions" in schema:
                schema["decisions"][0]["amount"] = min(schema["decisions"][0]["amount"], extra["proposals"][0]["amount"])
        if self.workbench is not None and team and phase != "work":
            required["verified_progress"] = self.workbench.summary(team.id)
        if self.workbench is not None and phase == "work":
            required["coding"] = self.workbench.observation(agent)
            coding_context = [{"previous_tool": item} for item in required["coding"].pop("previous_tool")] + [
                {"project_notes": required["coding"].pop("project_notes")}
            ]
            tool_schema = deepcopy(self.workbench.schema)
            if cfg.progress_control and required["coding"].get("probe_count", 0) >= required["coding"].get("probe_limit", 1):
                tool_schema["kind"] = tool_schema["kind"].replace("|probe", "")
                required["coding"]["probe_status"] = "Full: use existing /evidence/probes.json to fix and test; new probes are unavailable."
            required["schema"] = dict(required["schema"], tool=tool_schema)
        if cfg.progress_control:
            required["stalled_rounds"] = self.stalled_rounds()
        if cfg.funding_recovery and team:
            required["funding_status"] = self.funding_receipt(agent)
            required["standing_grant"] = agent.standing_grant
        if team:
            required["recruitment"] = {
                "direct_slots": cfg.max_children - len(agent.children),
                "can_delegate": self.depth(agent) < cfg.max_depth,
                "introductions": self.introduction_views(agent),
                "budget_options": self.recruitment_costs(agent.model),
                "plan": agent.recruitment_plan,
                "funding_retry_round": agent.funding_retry_round,
            }
            required["local_status"] = [
                {
                    "id": name,
                    "allowance": team.allowances[name],
                    "model": self.agents[name].model,
                    "wake_round": self.agents[name].wake_round,
                    "child_slots": cfg.max_children - len(self.agents[name].children),
                }
                for name in self.neighbors(agent)
            ]
        optional = [{"memory": agent.memory}, *coding_context, *[{"message": item} for item in agent.inbox]]
        if phase == "application" and cfg.quality_first:
            required["discovery_evidence"] = agent.discovery_result
        if phase == "formation":
            required.update(peers=[self.card(p) for p in peers], may_propose=may_propose)
            optional += [{"public_trail": item} for item in list(self.trails)[-cfg.scratch_reads :]]
        elif team:
            if phase == "work":
                required["work_peers"] = [self.card(p) for p in peers]
            # No full-team roster or shared transcript. Exact-key retrieval is O(reads).
            keys = list(dict.fromkeys(agent.read_keys + self.neighbors(agent) + [agent.name]))
            optional += [
                {"scratch": {"author": key, **team.scratchpad[key]}}
                for key in keys[: cfg.scratch_reads]
                if key in team.scratchpad
            ]
        if cfg.quality_first:
            turn_rules = quality_rules(cfg, phase)
            required["objective"]["primary"] = "Maximize final correctness and completeness within the hard cap; cost is a recorded constraint."
            required["directory"] = agent.directory
            required["limits"]["messages"] = cfg.message_contacts
            if phase in {"formation", "work"}:
                schema["lookup"] = "topic to find relevant collaborators (at most 240 bytes)"
            if "messages" in schema:
                schema["messages"][0]["to"] = "any known live worker ID, including other teams"
            if phase == "formation":
                required["collaboration_offers"] = self.offers_for(agent)
            if phase == "work":
                required["call_mode"] = self.work_kind(agent)
            required["schema"] = dict(required["schema"], **{k:v for k,v in schema.items() if k == "lookup"})
        else:
            turn_rules = RULES + f"\nYOUR CURRENT PHASE IS {phase.upper()}. Make that decision now."
            if cfg.hierarchy_bias:
                turn_rules += HIERARCHY_RULES
            if self.board.enabled:
                turn_rules += WORKER_BOARD_RULES
            if cfg.standing_grants:
                turn_rules += RENEWAL_RULES
            if cfg.progress_control:
                turn_rules += (" If repeated work stalls, change approach, propose a scoped collaborator or "
                               "a stronger model, or stop. Model tier does not determine anyone's role. "
                               "Use existing failing evidence before requesting more probes; preserve regression checks.")
            if cfg.funding_recovery:
                turn_rules += (
                    "\nLow liquidity creates a runtime renewal request to your direct manager; "
                    "you still choose work, and your manager decides funding. Pending requests need no duplicate proposal. "
                    "Keep several calls of reserve for yourself AND children. Hire/spawn with at least two "
                    "maximum-call reserves of the CHILD'S model; advertised examples use the cheapest model. "
                    "Board ownership prevents conflicting edits, not useful collaboration: decompose locally, "
                    "open scoped jobs, or spawn a child with an independently useful assignment. "
                    "Any model may manage any other model. Escalate a reproduced unresolved problem when "
                    "the expected improvement justifies the higher price; request delegation funds if needed. "
                    "Accepting an advertised job needs only accept=true; its assignment and allowance are the default proposal."
                )
            if cfg.bank_finalization:
                turn_rules += (
                    "\nYour team cannot end society work. Use ready with evidence AFTER seeing the last tool result "
                    "and without a tool in that same reply. The bank decides final submission of the integrated artifact. "
                    "Marking ready is a local completion report, not a claim that other teams are finished."
                )
            if phase != "work":
                turn_rules += "\nNo coding tools in this phase. Use only this phase's schema; work resumes in the work phase."
        if self.workbench is not None and phase == "work":
            remaining = required.get("work_calls_left_this_round", cfg.work_turns)
            turn_rules += (
                f" You have {remaining} work calls left this round INCLUDING THIS CALL."
                " Each inspection, edit, test, or message consumes a call."
                + (" Leave verified implementation, investigation, integration or team-progress evidence;"
                   if cfg.quality_first else " Leave a concrete source change or verified test result before the round ends;")
                + " use targeted reads when output is truncated."
                " Do not simulate future turns or tool results. Return one JSON action object only."
            )
            if remaining == 1:
                turn_rules += " THIS IS YOUR LAST WORK CALL THIS ROUND."
            # The result is JSON inside a JSON message: escaping can grow it a
            # second time. Fit the receipt to actual remaining serialized space
            # while preserving the task, policy, schema and required state.
            while required["coding"]["result"]:
                excess = (
                    byte_size(context_messages(turn_rules, {**required, "context": []})) - self.context_limit(self.work_kind(agent))
                )
                if excess <= 0:
                    break
                result = required["coding"]["result"]
                budget = max(64, len(json.dumps(result, ensure_ascii=False).encode()) - excess)
                shortened = self.workbench.bounded_result(result, budget=budget)
                if shortened == result:
                    break  # Other required data cannot fit; build_context rejects it before inference.
                required["coding"]["result"] = shortened
        return build_context(turn_rules, required, optional, self.context_limit(self.work_kind(agent) if phase == "work" else phase))

    def call(self, model, messages, *, agent=None, wallet="bank", kind="decision", payer=None):
        cfg = self.config
        priced_kind = self.work_kind(agent) if kind == "work" and agent else kind
        reserve = self.call_budget(model, priced_kind, messages)
        output_limit = self.output_limit(priced_kind)
        payer = payer if payer is not None else (agent if wallet != "bank" else None)
        team = self.teams.get(payer.team) if payer else None
        available = self.ledger.available(wallet)
        if wallet == "bank" and kind != "judge":
            available = self.bank_spendable()
        if team and wallet == team.id:
            available = min(available, team.allowances[payer.name])
        if available < reserve:
            self.emit("call_skipped", agent=agent.name if agent else "bank", reason="insufficient allowance")
            return None
        size = byte_size(messages)
        if size > cfg.context_bytes:
            raise AssertionError("Unbounded context")
        self.max_context_seen = max(size, self.max_context_seen)
        self.calls += 1
        call_id = f"c{self.calls:07d}"
        self.emit(
            "call_reserved",
            call_id=call_id,
            actor=agent.name if agent else kind,
            model=model,
            wallet=wallet,
            payer=payer.name if payer else "bank",
            kind=kind,
            credits=reserve,
            input_bytes=size,
            max_output_tokens=output_limit,
            messages=messages,
        )
        try:
            result = self.provider.complete(model, messages, output_limit, call_id)
        except BudgetStop:
            self.emit("call_cancelled", call_id=call_id)
            raise
        except Exception:
            # Never pretend a possibly billed failure was free. This is a marked
            # conservative hold, not an assertion about actual tokens generated.
            self.charge(payer, wallet, reserve, "uncertain_call_reserve")
            self.uncertain_credits += reserve
            self.emit("call_uncertain", call_id=call_id, credits=reserve)
            raise
        if (
            type(result.output_tokens) is not int
            or result.output_tokens < 0
            or result.output_tokens > output_limit
        ):
            self.charge(payer, wallet, reserve, "invalid_usage_reserve")
            self.uncertain_credits += reserve
            self.emit("call_uncertain", call_id=call_id, credits=reserve)
            raise UncertainCall("Provider output count violated its contract")
        try:
            credits = self.generation_credits(result)
        except (ValueError, TypeError, ArithmeticError):
            self.charge(payer, wallet, reserve, "invalid_usage_reserve")
            self.uncertain_credits += reserve
            self.emit("call_uncertain", call_id=call_id, credits=reserve)
            raise UncertainCall("Provider did not supply a valid charge") from None
        if credits > reserve:
            self.charge(payer, wallet, reserve, "invalid_usage_reserve")
            self.uncertain_credits += reserve
            self.emit("call_uncertain", call_id=call_id, credits=reserve)
            raise UncertainCall("Provider cost exceeded the credit reservation")
        self.charge(payer, wallet, credits, "generated_tokens" if cfg.cost_mode == "output_tokens" else "api_cost")
        self.output_tokens += result.output_tokens
        if result.estimated:
            self.estimated_tokens += result.output_tokens
        self.emit(
            "call",
            call_id=call_id,
            actor=agent.name if agent else kind,
            kind=kind,
            model=model,
            wallet=wallet,
            text=result.text,
            output_tokens=result.output_tokens,
            input_tokens=result.input_tokens,
            cost_usd=result.cost_usd,
            charged_credits=credits,
            estimated=result.estimated,
        )
        if agent:
            agent.inbox = []
        return result.text

    def charge(self, agent, wallet, value, reason):
        self.ledger.burn(wallet, value, reason)
        if agent and agent.team == wallet:
            self.teams[wallet].allowances[agent.name] -= value

    def ask(self, agent, phase, **kwargs):
        messages, dropped = self.observation(agent, phase, **kwargs)
        if dropped:
            self.emit("context_omitted", agent=agent.name, items=dropped)
        team = self.teams.get(agent.team)
        if not team or not team.funded or self.rights(agent) < self.call_budget(agent.model, self.work_kind(agent) if phase == "work" else phase, messages):
            return {}
        wallet = team.id
        text = self.call(agent.model, messages, agent=agent, wallet=wallet, kind=phase)
        allowed = set(SCHEMAS[phase])
        if self.config.quality_first and phase in {"formation", "work"}:
            allowed.add("lookup")
        if self.config.bank_finalization and phase == "work":
            allowed.add("ready")
        if self.workbench is not None and phase == "work":
            allowed.add("tool")
        return self.parse(text, allowed, agent.name)

    def parse(self, text, allowed, actor):
        if text is None:
            return {}
        try:

            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ValueError("Duplicate JSON key")
                    result[key] = value
                return result

            def invalid_number(_):
                raise ValueError("Nonfinite JSON number")

            if len(text.encode("utf-8")) > self.config.max_output_tokens * 32:
                raise ValueError("Oversized reply")
            reply = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_number)
            if not isinstance(reply, dict) or set(reply) - allowed:
                raise ValueError("Unexpected decision fields")
            self.format_errors.pop(actor, None)
            return reply
        except (ValueError, TypeError, RecursionError) as exc:
            self.emit("invalid_reply", actor=actor, reason=clip(str(exc), 300))
            self.format_errors[actor] = (
                "Your previous response executed no actions: "
                + clip(str(exc), 180)
                + ". Return exactly one JSON object, no prose, code fences, duplicate keys or additional objects."
            )
            return {}

    def attempt(self, agent, action, fn):
        try:
            fn()
        except (ValueError, TypeError, KeyError) as exc:
            self.emit("action_rejected", agent=agent.name, action=action, reason=clip(str(exc), 300))
            agent.inbox.append(
                {
                    "from": "runtime",
                    "text": f"{action} rejected: {clip(str(exc), 300)}",
                    "scope": "receipt",
                    "round": self.round,
                }
            )
            agent.inbox = agent.inbox[-self.config.inbox_size :]

    def remember_and_sleep(self, agent, reply):
        if self.config.quality_first and "lookup" in reply:
            self.attempt(agent, "lookup", lambda: self.lookup_collaborators(agent, reply["lookup"]))
        if "strategy" in reply:

            def update_strategy():
                value = clip(reply["strategy"], self.config.strategy_bytes)
                if not value.strip():
                    raise ValueError("Strategy must be nonempty")
                agent.trainable_system_prompt = value
                self.emit("strategy_changed", agent=agent.name, strategy=value)

            self.attempt(agent, "strategy", update_strategy)
        if "memory" in reply:
            self.attempt(
                agent,
                "memory",
                lambda: setattr(agent, "memory", clip(reply["memory"], self.config.text_bytes)),
            )
        if "sleep" in reply:

            def sleep():
                rounds = reply["sleep"]
                if type(rounds) is not int or not 0 <= rounds <= self.config.max_sleep_rounds:
                    raise ValueError("Invalid sleep duration")
                if rounds:
                    agent.wake_round = self.round + rounds + 1
                    self.emit("sleep", agent=agent.name, wake_round=agent.wake_round)

            self.attempt(agent, "sleep", sleep)

    def deliver(self, deliveries):
        for source, target, text, scope in deliveries:
            agent = self.agents.get(target)
            if not agent:
                continue
            agent.inbox.append({"from": source, "text": text, "scope": scope, "round": self.round})
            agent.inbox = agent.inbox[-self.config.inbox_size :]
            agent.wake_round = self.round
            self.emit("message", source=source, target=target, text=text, scope=scope)

    def outbound(self, agent, reply, allowed, scope):
        result = []
        if "messages" not in reply:
            return result
        data = reply["messages"]
        if self.config.quality_first:
            allowed = {n for n,a in self.agents.items() if not a.retired and n != agent.name}
        limit = self.config.message_contacts if self.config.quality_first else len(allowed)
        if not isinstance(data, list) or len(data) > limit:
            raise ValueError("Too many messages")
        seen = set()
        for item in data:
            if not isinstance(item, dict) or item.get("to") not in allowed or item["to"] in seen:
                raise ValueError("Recipient outside allowed neighborhood or duplicated")
            seen.add(item["to"])
            result.append((agent.name, item["to"], clip(item["text"], self.config.text_bytes), scope))
        return result

    def allocate(self, parent, data):
        if not isinstance(data, list) or len(data) > self.config.max_children:
            raise ValueError("Allocation list exceeds local degree")
        team = self.teams[parent.team]
        seen = set()
        for item in data:
            if (
                not isinstance(item, dict)
                or item.get("agent") not in parent.children
                or item["agent"] in seen
            ):
                raise ValueError("Allocate only once to each direct child")
            child = self.agents[item["agent"]]
            value = item["amount"]
            if (
                type(value) is not int
                or not -team.allowances[child.name] <= value <= team.allowances[parent.name]
            ):
                raise ValueError("Allocation exceeds existing rights")
            assignment = clip(item.get("assignment", child.assignment), self.config.text_bytes)
            team.allowances[parent.name] -= value
            team.allowances[child.name] += value
            child.assignment = assignment
            child.wake_round = self.round
            seen.add(child.name)
            self.emit("allocate", parent=parent.name, child=child.name, amount=value, assignment=assignment)

    def work(self):
        self.phase = "work"
        submissions, publications = [], []
        # Reuse bounded, symmetric matching among funded workers. Keep contacts
        # fixed for the round so replies can return across the next work barrier.
        # Sleeping contacts may be woken by a paid message; newborns wait a round.
        contacts = peer_graph(
            [a.name for a in self.agents.values()
             if a.born_round < self.round and not a.retired and a.team
             and self.teams[a.team].funded and self.board.can_work(a.team)],
            self.round, self.config.peer_contacts,
        )
        for turn in range(self.config.work_turns):
            active = [
                a
                for a in self.agents.values()
                if a.born_round < self.round
                and a.team
                and self.teams[a.team].funded
                and self.board.can_work(a.team)
                and a.match_wakeup_condition(self)
            ]
            # A barrier between calls and actions prevents same-turn reading order advantages.
            replies = [
                (a, self.ask(a, "work", peers=contacts.get(a.name, ()),
                             extra={"work_calls_left_this_round": self.config.work_turns - turn}))
                for a in active
            ]
            deliveries = []
            for agent, reply in replies:
                self.remember_and_sleep(agent, reply)
                team = self.teams[agent.team]
                if "tool" in reply and self.workbench is not None:
                    self.attempt(agent, "tool", lambda: (
                        self.execute_with_progress(agent, reply["tool"])
                        if self.config.progress_control or self.config.standing_grants
                        else self.workbench.execute(agent, reply["tool"])))
                if self.config.bank_finalization and ("ready" in reply or "submit" in reply):
                    self.attempt(agent, "ready", lambda: self.mark_ready(agent, reply))
                if "note" in reply:
                    self.attempt(
                        agent,
                        "note",
                        lambda: team.scratchpad.__setitem__(
                            agent.name,
                            {"text": clip(reply["note"], self.config.text_bytes), "round": self.round},
                        ),
                    )
                allowed = set(self.neighbors(agent)) | set(contacts.get(agent.name, ()))
                self.attempt(
                    agent, "messages", lambda: deliveries.extend(self.outbound(agent, reply, allowed, "work"))
                )
                if "allocate" in reply:
                    self.attempt(agent, "allocate", lambda: self.allocate(agent, reply["allocate"]))
                if "spawn" in reply:

                    def spawn():
                        data = reply["spawn"]
                        if not isinstance(data, dict):
                            raise ValueError("Spawn needs model, strategy, assignment and allowance")
                        self.new_agent(
                            data["model"],
                            data["strategy"],
                            parent=agent,
                            allowance=data["allowance"],
                            assignment=data["assignment"],
                        )

                    self.attempt(agent, "spawn", spawn)
                if "read_keys" in reply:

                    def read():
                        keys = reply["read_keys"]
                        if not isinstance(keys, list) or len(keys) > self.config.scratch_reads:
                            raise ValueError("Too many scratchpad reads")
                        if any(not isinstance(k, str) or k not in team.members for k in keys):
                            raise ValueError("Scratchpad keys must belong to your team")
                        agent.read_keys = list(dict.fromkeys(keys))

                    self.attempt(agent, "read_keys", read)
                for key, destination in (("publish", publications), ("submit", submissions)):
                    if key == "submit" and self.config.bank_finalization:
                        continue
                    if key in reply:

                        def publish():
                            if agent.name != team.manager:
                                raise ValueError("Send findings to your parent; only managers publish/submit")
                            limit = self.config.answer_bytes if key == "submit" else self.config.text_bytes
                            value = reply[key]
                            if not isinstance(value, str) or not value.strip():
                                raise ValueError("Nonempty text required")
                            if key == "submit" and len(value.encode("utf-8")) > limit:
                                raise ValueError("Final answer exceeds answer_bytes; summarize explicitly")
                            destination.append(
                                {
                                    "author": agent.name,
                                    "team": team.id,
                                    "text": clip(value, limit),
                                    "round": self.round,
                                }
                            )

                        self.attempt(agent, key, publish)
            self.deliver(deliveries)
            self.check()
            if submissions:
                break
        self.phase = "publish"
        for item in publications:
            self.trails.append(item)
            self.emit("published", **item)
        if submissions:
            # At most one candidate is graded. Selection never consults hidden reward.
            by_team = {s["team"]: s for s in submissions}
            order = sorted(self.teams)
            start = (self.round - 1) % len(order)
            winner = next(by_team[t] for t in order[start:] + order[:start] if t in by_team)
            self.final_answer, self.final_author = winner["text"], winner["author"]
            self.submission_mode = "explicit"
            self.emit("submission", **winner, candidates=len(by_team), selection="rotating_team_priority")

    def mark_ready(self, agent, reply):
        team = self.teams[agent.team]
        if agent.name != team.manager or "tool" in reply:
            raise ValueError("Only a root may report readiness, after receiving tool results in a prior turn")
        text = reply.get("ready", reply.get("submit"))
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Readiness requires evidence and remaining gaps")
        progress = self.workbench.summary(team.id) if self.workbench else {}
        team.ready = {"text": clip(text, self.config.text_bytes), "round": self.round,
                      "revision": progress.get("revision")}
        self.emit("team_ready", team=team.id, agent=agent.name, **team.ready)

    def readiness_receipts(self):
        receipts = []
        for item in self.board.items.values():
            team = self.teams.get(item["team"])
            progress = self.workbench.summary(team.id) if self.workbench and team else {}
            ready = team.ready if team else None
            receipts.append({"task_id": item["id"], "status": item["status"],
                             "ready": ready,
                             "current": bool(ready and ready["revision"] == progress.get("revision")),
                             "conflicts": progress.get("conflicts", [])})
        return receipts

    def finalize_society(self, reason):
        receipts = self.readiness_receipts()
        check = self.workbench.public_check if self.workbench else None
        if (not isinstance(reason, str) or not reason.strip() or not receipts
                or any(not r["current"] or r["conflicts"] or r["status"] not in {"claimed", "done"}
                       for r in receipts)
                or not check or check.get("status") not in {"build_only", "public_probe_score"}):
            self.emit("finalization_deferred", reason="Need current team readiness and a successful integrated build")
            return
        # A locally ready branch must have made it into the public artifact.
        if any(self.workbench.hashes(self.workbench.sources[t]) != self.workbench.hashes(self.workbench.public)
               for t in {i["team"] for i in self.board.items.values()}):
            self.emit("finalization_deferred", reason="Pending source integration")
            return
        self.final_answer, self.final_author = clip(reason, self.config.answer_bytes), "__public__"
        self.submission_mode = "bank_approved_public"
        self.emit("submission", author="__public__", team="public", text=self.final_answer,
                  candidates=1, selection="bank_approved_public")

    def judge(self, prompt):
        # The protected rubric enters only this evaluator call, never bank/worker context.
        messages, _ = build_context(
            "Evaluate using the rubric. Follow the requested SCORE/REASON format.",
            {"evaluation": prompt},
            [],
            self.config.context_bytes,
        )
        text = self.call(self.config.bank_model, messages, kind="judge")
        if text is None:
            raise BudgetStop("Reserved evaluation funds unavailable")
        return text

    def check(self):
        self.ledger.check()
        if not 0 <= self.discovery_spent <= self.config.discovery_budget:
            raise AssertionError("Sponsored discovery exceeded the fixed pool")
        if not 0 <= self.application_spent <= self.config.application_budget:
            raise AssertionError("Sponsored bank applications exceeded the fixed pool")
        if len(self.agents) != len(self.population) or len(self.agents) > self.config.max_agents:
            raise AssertionError("Population mismatch")
        for team in self.teams.values():
            if set(team.allowances) != team.members or sum(team.allowances.values()) != self.ledger.available(
                team.id
            ):
                raise AssertionError("Spending rights do not match team money")
            if any(type(v) is not int or v < 0 for v in team.allowances.values()):
                raise AssertionError("Negative/invalid spending rights")
            if team.manager not in team.members or self.agents[team.manager].parent:
                raise AssertionError("Invalid root manager")
        for agent in self.agents.values():
            if len(agent.introductions) > self.config.manager_contacts or len(set(agent.introductions)) != len(agent.introductions):
                raise AssertionError("Unbounded or duplicate introductions")
            for name in agent.introductions:
                candidate = self.agents.get(name)
                if not candidate or not candidate.referral or candidate.referral["target"] != agent.name:
                    raise AssertionError("Introduction routing mismatch")
            if agent.referral:
                owner = self.agents.get(agent.referral["target"])
                if owner and agent.name not in owner.introductions:
                    raise AssertionError("Missing introduction queue entry")
            if len(agent.children) > self.config.max_children or self.depth(agent) > self.config.max_depth:
                raise AssertionError("Unbounded recruitment")
            if agent.team:
                team = self.teams[agent.team]
                if agent.name not in team.members:
                    raise AssertionError("Missing team membership")
                if not agent.parent and team.manager != agent.name:
                    raise AssertionError("Multiple team roots")
            elif agent.parent or agent.children:
                raise AssertionError("Unteamed agent has recruitment edges")
            for child in agent.children:
                if self.agents[child].parent != agent.name or self.agents[child].team != agent.team:
                    raise AssertionError("Recruitment edge mismatch")

    def snapshot(self):
        return {
            "round": self.round,
            "status": self.status,
            "wealth": self.ledger.wealth,
            "bank": self.ledger.available("bank"),
            "burned": self.ledger.burned,
            "rewarded": self.ledger.rewarded,
            "bank_brief": self.bank_brief,
            "bank_board": self.board.view(),
            "application_spent": self.application_spent,
            "discovery_spent": self.discovery_spent,
            "teams": [t.to_dict() for t in self.teams.values()],
            "agents": [
                {
                    "id": a.name,
                    "team": a.team,
                    "parent": a.parent,
                    "children": a.children[:],
                    "model": a.model,
                    "wake_round": a.wake_round,
                    "born_round": a.born_round,
                    "assignment": a.assignment,
                    "strategy": a.trainable_system_prompt,
                    "application": a.application,
                    "last_feedback": a.last_feedback,
                    "opening": a.opening,
                    "referral": a.referral,
                    "introductions": list(a.introductions),
                    "recruitment_plan": a.recruitment_plan,
                }
                for a in self.agents.values()
            ],
        }

    def run(self):
        if self.status != "ready":
            raise RuntimeError("An engine instance runs once; automatic paid retries are disabled")
        self.status = "running"
        self.emit(
            "start",
            config=self.config.to_dict(),
            task={"id": self.task.id, "problem": self.task.problem, "kind": self.task.kind},
            provider="scripted_demo" if self.provider.scripted else "live",
        )
        try:
            for r in range(1, self.config.max_rounds + 1):
                self.round = r
                if r == 1 and self.board.enabled:
                    self.fund([], planning=True)
                self.phase = "formation"
                for team in list(self.teams.values()):
                    if team.funded and team.expires < r and not self.config.persistent_teams:
                        self.close_team(team)
                while self.trails and r - self.trails[0]["round"] > self.config.trail_ttl:
                    self.trails.popleft()
                self.discover()
                proposals = self.formation()
                self.fund(proposals)
                if self.final_answer is None:
                    self.work()
                if self.workbench is not None:
                    self.phase = "checkpoint"
                    self.workbench.finish_round(r, limit=self.config.bank_contacts)
                    self.assess_progress()
                    if r == self.config.max_rounds and self.final_answer is None:
                        self.final_answer = (
                            "Automatically exported integrated public checkpoint at the round limit."
                        )
                        self.final_author = "__public__"
                        self.submission_mode = "automatic_checkpoint"
                        self.emit(
                            "submission",
                            author=self.final_author,
                            team="public",
                            text=self.final_answer,
                            candidates=1,
                            selection="public_checkpoint_at_deadline",
                        )
                if self.final_answer is not None:
                    self.phase = "evaluate"
                    self.score = evaluate(self.env, self.final_answer, self.final_author)
                    reward = int(
                        (Decimal(str(self.score)) * self.config.reward).quantize(
                            Decimal(1), rounding=ROUND_HALF_UP
                        )
                    )
                    self.ledger.reward_once(self.task.id, reward)
                    self.status = "completed"
                if self.config.rent_per_round:
                    self.apply_rent()
                self.check()
                self.emit("snapshot", state=self.snapshot())
                if self.status == "completed":
                    break
            if self.status == "running":
                self.status = "round_limit"
        except BudgetStop as exc:
            self.status = "budget_stopped"
            self.emit("stopped", reason=str(exc))
            if self.workbench is not None and getattr(self.workbench, "grade_on_budget_stop", False):
                # No further model calls. Select the already-integrated source,
                # then evaluate once; never select among candidates by hidden scores.
                self.phase = "evaluate"
                self.final_answer = "Automatically selected public checkpoint at the dollar stop."
                self.final_author = "__public__"
                self.submission_mode = "automatic_budget_checkpoint"
                self.emit("submission", author=self.final_author, team="public", text=self.final_answer,
                          candidates=1, selection="public_checkpoint_at_budget_stop")
                try:
                    self.score = evaluate(self.env, self.final_answer, self.final_author)
                    reward = int((Decimal(str(self.score)) * self.config.reward).quantize(Decimal(1), rounding=ROUND_HALF_UP))
                    self.ledger.reward_once(self.task.id, reward)
                    self.status = "completed"
                except Exception as grading_error:
                    self.status = "error"
                    self.emit("stopped", reason=clip(str(grading_error),500), error_type=type(grading_error).__name__)
                    raise
        except Exception as exc:
            self.status = "error"
            self.emit("stopped", reason=clip(str(exc), 500), error_type=type(exc).__name__)
            self.emit("snapshot", state=self.snapshot())
            raise
        finally:
            self.phase = "settle"
            if self.workbench is not None:
                self.workbench.export_public(self.status)
            for team in list(self.teams.values()):
                self.close_team(team)
            self.check()
            self.emit("finished", report=self.report())
        return self.report()

    def apply_rent(self):
        self.phase = "rent"
        value = min(self.config.rent_per_round, self.bank_spendable())
        self.ledger.burn("bank", value, "bank_rent")
        for agent in self.agents.values():
            team = self.teams.get(agent.team)
            wallet = team.id if team and team.funded else "bank"
            available = team.allowances[agent.name] if wallet != "bank" else self.bank_spendable()
            self.charge(agent, wallet, min(available, self.config.rent_per_round), "agent_rent")

    def report(self):
        return {
            "status": self.status,
            "task_id": self.task.id,
            "rounds": self.round,
            "final_answer": self.final_answer,
            "author": self.final_author,
            "submission_mode": self.submission_mode,
            "score": self.score,
            "initial_wealth": self.ledger.initial,
            "final_wealth": self.ledger.wealth,
            "bank": self.ledger.available("bank"),
            "burned": self.ledger.burned,
            "rewarded": self.ledger.rewarded,
            "net_gain": self.ledger.wealth - self.ledger.initial,
            "calls": self.calls,
            "output_tokens": self.output_tokens,
            "estimated_tokens": self.estimated_tokens,
            "uncertain_credits": self.uncertain_credits,
            "application_spent": self.application_spent,
            "discovery_spent": self.discovery_spent,
            "application_budget": self.config.application_budget,
            "max_context_bytes": self.max_context_seen,
            "context_cap_bytes": self.config.context_bytes,
            "population": len(self.agents),
            "bank_board": self.board.view(),
            "total_births": self.next_agent - 1,
            "api_spent_usd": str(self.provider.spent),
            "api_reserved_usd": str(self.provider.reserved),
            "scripted_demo": self.provider.scripted,
            "usage_verified": not self.provider.scripted
            and self.estimated_tokens == 0
            and self.uncertain_credits == 0
            and self.provider.reserved == 0,
            "accounting_identity": self.ledger.wealth
            == self.ledger.initial + self.ledger.rewarded - self.ledger.burned,
        }
