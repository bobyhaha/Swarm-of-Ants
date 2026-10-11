"""Sponsored admission and the same bounded review contract at every tree level."""

from copy import deepcopy

from .board import BOARD_RULES, BOARD_SCHEMA
from .context import build_review_context, clip
from .ledger import amount
from .protocol import BANK_RULES, HIERARCHY_RULES, SCHEMAS, peer_graph
from .pricing import Pricing
from .adaptation import AdaptiveFunding, RENEWAL_SCHEMA, RENEWAL_RULES
from .recruitment import Recruitment
from .delegation import DelegationFunding
from .quality import rules as quality_rules


class FundingProtocol(Recruitment, DelegationFunding, AdaptiveFunding, Pricing):
    def bank_grantable(self):
        if not self.config.funding_recovery:
            return self.bank_spendable()
        # Retain one full next-call reservation plus a declining operating
        # envelope (20% of initial capital across remaining rounds). These are
        # existing credits, not extra authorization or assumed future income.
        rounds_left = max(0, self.config.max_rounds - self.round)
        operating = self.config.initial_bank * rounds_left // (5 * self.config.max_rounds)
        if self.config.event_driven_bank:
            operating = self.call_budget(self.config.bank_model, "bank")
        return max(0, self.bank_spendable() - self.call_budget(self.config.bank_model, "bank") - operating)

    def funding_request(self, agent):
        """A bounded local liquidity signal; no generation or transfer of money."""
        reserve = self.working_minimum(agent.model)
        child_requests = sum(
            c.application["amount"]
            for name in agent.children
            if (c := self.agents[name]).application
            and c.application.get("target") == agent.name
            and c.application.get("status") == "pending"
        )
        if self.rights(agent) >= reserve * 2 + child_requests:
            return 0
        return min(self.config.max_grant, max(reserve * 2, reserve * 4 + child_requests - self.rights(agent)))

    def queue_funding_recovery(self, active):
        # Each node reads at most max_children requests. Requests propagate up
        # over rounds; no ancestor receives a full subtree or a global roster.
        for agent in active:
            if agent.funding_retry_round > self.round:
                continue
            if not agent.team or not self.board.can_work(agent.team):
                continue
            ready = self.teams[agent.team].ready
            if ready and (not self.workbench or ready["revision"] == self.workbench.summary(agent.team)["revision"]):
                continue
            target = agent.parent or "bank"
            if not self.application_open(agent, target) or agent.application:
                continue
            value = self.funding_request(agent)
            if not value:
                continue
            data = {"amount": value,
                    "plan": clip("Runtime liquidity request for existing assignment: " + agent.assignment
                                 + ". Continue only useful work; manager may reject or revise this request.",
                                 self.config.text_bytes),
                    "task_id": self.teams[agent.team].board_task_id}
            self.record_proposal(agent, data, target)
            agent.application["runtime_recovery"] = True
            self.emit("funding_needed", agent=agent.name, target=target, amount=value,
                      allowance=self.rights(agent), minimum_call=self.call_budget(agent.model))

    def funding_receipt(self, agent):
        return {"agent": agent.name, "model": agent.model, "allowance": self.rights(agent),
                "minimum_call": self.call_budget(agent.model),
                "can_afford_call": self.rights(agent) >= self.call_budget(agent.model),
                "pending_request": (agent.application or {}).get("amount", 0),
                "children": [{"agent": name, "model": self.agents[name].model,
                              "allowance": self.rights(self.agents[name]),
                              "minimum_call": self.call_budget(self.agents[name].model),
                              "pending_request": (self.agents[name].application or {}).get("amount", 0),
                              "request_purpose": clip((self.agents[name].application or {}).get("plan", ""), 240)}
                             for name in agent.children]}

    @property
    def call_credits(self):
        return min(self.call_budget(model) for model in self.config.worker_models)

    def rights(self, agent):
        team = self.teams.get(agent.team)
        return team.allowances[agent.name] if team and team.funded else 0

    def application_open(self, agent, target):
        app = agent.application
        if target == "bank" and agent.bank_closed:
            return False
        return not app or app["target"] != target or app["status"] not in {"reject", "approved"}

    def sponsored_application(self, agent, target, invitation=None):
        """Sponsor authorizes this exact call; never transfers free personal money."""
        cfg = self.config
        reserve = self.call_budget(agent.model, "application")
        sponsor = self.agents.get(target) if target != "bank" else None
        if target == "bank":
            if self.application_spent + reserve > cfg.application_budget:
                self.emit(
                    "application_skipped",
                    agent=agent.name,
                    target=target,
                    reason="application pool exhausted",
                )
                return {}
            if self.bank_spendable() < reserve:
                return {}
            wallet = "bank"
        else:
            if not sponsor or self.rights(sponsor) < reserve:
                return {}
            wallet = sponsor.team
        messages, dropped = self.observation(
            agent,
            "application",
            may_propose=True,
            extra={
                "application_target": target,
                "opportunity": invitation,
                "sponsored": True,
                "may_decline": True,
            },
        )
        if dropped:
            self.emit("context_omitted", agent=agent.name, items=dropped)
        self.emit(
            "application_sponsored",
            agent=agent.name,
            target=target,
            wallet=wallet,
            maximum_credits=self.call_budget(agent.model, "application", messages),
        )
        before = self.ledger.burned
        try:
            text = self.call(
                agent.model, messages, agent=agent, wallet=wallet, payer=sponsor, kind="application"
            )
        finally:
            if target == "bank":
                self.application_spent += self.ledger.burned - before
        if target == "bank":
            agent.bank_attempts += 1
        reply = self.parse(text, set(SCHEMAS["application"]), agent.name)
        if isinstance(reply.get("memory"), str):
            agent.memory = clip(reply["memory"], self.config.text_bytes)
        return reply

    def record_proposal(self, agent, data, target, *, consent=False):
        if agent.team and agent.funding_retry_round > self.round:
            raise ValueError("Wait until funding_retry_round before another funding request")
        if not self.application_open(agent, target):
            raise ValueError("Application closed; no automatic resubmission after rejection")
        if not isinstance(data, dict) or not 0 < amount(data["amount"]) <= self.config.max_grant:
            raise ValueError("Invalid requested amount")
        plan = clip(data["plan"], self.config.text_bytes)
        if not plan.strip():
            raise ValueError("A proposal needs a deliverable")
        old = agent.application
        attempts = old["attempts"] if old and old["target"] == target else 0
        if old and old["target"] == target and old["status"] == "pending":
            raise ValueError("Await review before revising")
        if attempts >= 1 + self.config.max_revisions:
            raise ValueError("Revision limit reached")
        agent.application = {
            "agent": agent.name,
            "target": target,
            "plan": plan,
            "amount": data["amount"],
            "attempts": attempts + 1,
            "status": "pending",
            "feedback": old["feedback"] if old else "",
            "consent": consent,
            "round": self.round,
        }
        if self.board.enabled and target == "bank":
            agent.application["task_id"] = clip(data.get("task_id", ""), 24)
        self.emit("proposal", **agent.application)

    def proposal_view(self, agent):
        app = agent.application
        view = {k: app[k] for k in ("agent", "plan", "amount", "attempts", "feedback")} | {
            "team": agent.team,
            "current_allowance": self.rights(agent),
            "model": agent.model,
            "minimum_working_budget": self.working_minimum(agent.model),
            "minimum_grant": 1 if agent.team and app.get("recruitment") else self.working_minimum(agent.model) * (2 if self.config.funding_recovery else 1),
            "delegation_options": self.recruitment_costs(agent.model),
            "runtime_recovery": app.get("runtime_recovery", False),
            "recruitment": app.get("recruitment"),
        }
        if self.board.enabled and app["target"] == "bank":
            view["task_id"] = app.get("task_id", "")
            view["claim_feedback"] = self.board.conflict(agent.name, view["task_id"])
        return view

    def rotate(self, names, count, cursor):
        names = sorted(names)
        if not names:
            return [], cursor
        start = cursor % len(names)
        selected = (names[start:] + names[:start])[:count]
        return selected, (start + len(selected)) % len(names)

    def formation(self):
        self.phase = "formation"
        cfg = self.config
        self.prune_introductions()
        if cfg.quality_first:
            for a in self.agents.values():
                if a.discovery_offer and not a.team:
                    owner = self.agents.get(a.discovery_offer["target"])
                    if (self.round - a.discovery_offer["round"] >= cfg.grant_rounds
                            or (a.discovery_offer["target"] != "bank" and (not owner or not owner.team))):
                        a.discovery_offer = None
        self.prune_recruitment_plans()
        active = [
            a for a in self.agents.values() if a.born_round < self.round and a.match_wakeup_condition(self)
        ]
        active_names = {a.name for a in active}
        graph = peer_graph(active_names, self.round, cfg.peer_contacts)
        # Invalidated sponsorship cannot survive a closed team or changed parent.
        for a in active:
            if a.opening and a.opening["expires"] < self.round:
                a.opening = None
            app = a.application
            if app and app["target"] != "bank":
                sponsor = self.agents.get(app["target"])
                unavailable = not sponsor or not sponsor.team or (a.team and a.parent != sponsor.name)
                if sponsor and not a.team:
                    unavailable = (
                        unavailable
                        or not sponsor.opening
                        or (
                            sponsor.opening["expires"] < self.round
                            or len(sponsor.children) >= cfg.max_children
                        )
                    )
                if unavailable:
                    a.last_feedback = {
                        "from": app["target"],
                        "text": "Previous sponsorship is no longer available.",
                    }
                    a.application = None
        if cfg.standing_grants:
            self.standing_renewals(active)
        if cfg.funding_recovery:
            self.queue_funding_recovery(active)
        eligible = [
            a.name
            for a in active
            if (not a.team or not a.parent)
            and not (cfg.quality_first and a.discovery_offer and a.discovery_offer["target"] != "bank")
            and not a.referral
            and not (a.application and a.application["target"] != "bank"
                     and a.application["status"] in {"pending", "revise"})
            and self.application_open(a, "bank")
            and (
                not a.application or a.application["target"] == "bank" or a.application["status"] == "reject"
            )
            and (
                a.bank_attempts < 1 + cfg.max_revisions
                or (a.application and a.application["status"] == "pending")
            )
        ]
        urgent = [n for n in eligible if self.agents[n].team
                  and (self.agents[n].application or {}).get("runtime_recovery")]
        if cfg.funding_recovery and urgent:
            # Reserve capacity for existing blocked teams without starving discovery.
            count = cfg.bank_contacts if len(urgent) == len(eligible) else max(1, cfg.bank_contacts // 2)
            urgent_slots, self.recovery_cursor = self.rotate(
                urgent, count, getattr(self, "recovery_cursor", 0))
            rest, self.proposal_cursor = self.rotate(
                [n for n in eligible if n not in urgent_slots], cfg.bank_contacts - len(urgent_slots),
                self.proposal_cursor)
            slots = urgent_slots + rest
        else:
            slots, self.proposal_cursor = self.rotate(eligible, cfg.bank_contacts, self.proposal_cursor)
        slots = set(slots)
        self.emit("proposal_slots", agents=sorted(slots))
        # Every manager receives at most manager_contacts child applications/reviews.
        local_slots = {}
        for parent in active:
            candidates = [
                n
                for n in parent.children
                if n in active_names and self.application_open(self.agents[n], parent.name)
            ]
            chosen, parent.review_cursor = self.rotate(candidates, cfg.manager_contacts, parent.review_cursor)
            local_slots[parent.name] = chosen
        replies = {}
        for a in active:
            may = a.name in slots or (a.parent and a.name in local_slots.get(a.parent, []))
            if self.rights(a) >= self.call_budget(a.model, "formation"):
                replies[a.name] = self.ask(a, "formation", peers=graph[a.name], may_propose=bool(may))
            elif (a.name in slots and not (a.application and a.application["status"] == "pending")
                  and not (cfg.quality_first and a.discovery_calls < cfg.discovery_turns
                           and self.discovery_spent + self.call_budget(a.model, "discovery") <= cfg.discovery_budget
                           and self.workbench and hasattr(self.workbench, "inspect_public"))):
                replies[a.name] = self.sponsored_application(a, "bank")
            # Everyone else waits without a model call. No bank-funded social polling.
        deliveries = []
        for a in active:
            reply = replies.get(a.name, {})
            self.remember_and_sleep(a, reply)
            if reply.get("close") is True:

                def close(agent=a):
                    if not agent.team or agent.parent:
                        raise ValueError("Only a root manager may close its team")
                    self.close_team(self.teams[agent.team])

                self.attempt(a, "close", close)
            if reply.get("leave") is True or reply.get("retire") is True:
                self.attempt(a, "leave", lambda a=a: self.leave(a))
            if reply.get("retire") is True and not a.team:
                a.retired = True
                self.population.remove_agent(a)
                del self.agents[a.name]
                self.emit("retire", agent=a.name)
                continue
            # Payment authorized before generation permits the resulting bounded actions,
            # even when the generation consumes the last available credits.
            allowed = set(graph[a.name] + self.neighbors(a)) & self.agents.keys()
            self.attempt(
                a,
                "messages",
                lambda a=a, r=reply, ok=allowed: deliveries.extend(self.outbound(a, r, ok, "formation")),
            )
            if "opening" in reply:
                self.attempt(a, "opening", lambda a=a, r=reply: self.open_job(a, r["opening"]))
            if "introductions" in reply:
                self.attempt(a, "introductions", lambda a=a, r=reply: self.forward_introductions(a, r["introductions"]))
            if "recruitment_request" in reply:
                self.attempt(a, "recruitment_request", lambda a=a, r=reply: self.request_recruitment(a, r["recruitment_request"]))
            if "proposal" in reply:
                target = a.parent or "bank"
                admitted = a.name in slots if target == "bank" else a.name in local_slots.get(target, [])
                if admitted:
                    self.attempt(
                        a,
                        "proposal",
                        lambda a=a, r=reply, t=target: self.record_proposal(a, r["proposal"], t),
                    )
            if a.name in slots and not a.team and a.bank_attempts >= 1 + cfg.max_revisions:
                if not a.application or a.application["status"] != "pending":
                    a.bank_closed = True
        self.deliver(deliveries)
        index = self.candidate_index(active)
        # Local review batches: existing child requests first; remaining room admits
        # at most one explicitly sponsored external applicant per manager per round.
        batches, claimed = {}, set(slots)
        pending_by_target = {}
        for a in active:
            if not a.team and a.application and a.application["status"] in {"pending", "revise"}:
                pending_by_target.setdefault(a.application["target"], []).append(a.name)
        for parent in active:
            if parent.name not in self.agents or not parent.team:
                continue
            applicants = []
            for name in local_slots[parent.name]:
                child = self.agents.get(name)
                if not child or child.parent != parent.name:
                    continue
                app = child.application
                if (
                    app
                    and app["target"] == parent.name
                    and app["status"] == "revise"
                    and self.rights(child) < self.call_budget(child.model)
                ):
                    reply = self.sponsored_application(child, parent.name)
                    if "proposal" in reply:
                        self.attempt(
                            child,
                            "proposal",
                            lambda c=child, r=reply: self.record_proposal(c, r["proposal"], parent.name),
                        )
                if (
                    child.application
                    and child.application["target"] == parent.name
                    and child.application["status"] == "pending"
                ):
                    applicants.append(child)
            if (
                len(applicants) < cfg.manager_contacts
                and parent.opening
                and len(parent.children) < cfg.max_children
            ):
                pending = pending_by_target.get(parent.name, [])
                choices = self.recruitment_candidates(parent, pending[:cfg.manager_contacts], index, claimed)
                for name in choices:
                    child = self.agents.get(name)
                    if (
                        not child
                        or child.team
                        or child.retired or child.born_round >= self.round
                        or (child.referral and child.referral["target"] != parent.name)
                        or parent.opening["allowance"] < self.working_minimum(child.model) * (2 if cfg.funding_recovery or parent.opening.get("trial") else 1)
                        or name in claimed
                        or not self.application_open(child, parent.name)
                    ):
                        continue
                    if (
                        not child.application
                        or child.application["target"] != parent.name
                        or child.application["status"] != "pending"
                    ):
                        # Reserve enough rights for both screening and the manager review.
                        if self.rights(parent) < self.hiring_reserve(parent, child, parent.opening):
                            self.emit("recruitment_waiting", agent=parent.name, candidate=name,
                                      reason="Insufficient allowance for offer, screening, and retained runway")
                            continue
                        claimed.add(name)
                        reply = self.sponsored_application(child, parent.name, parent.opening)
                        if reply.get("accept") is True:
                            # The existing offer already specifies the work and ceiling.
                            # Explicit acceptance is sufficient consent; a custom
                            # proposal remains optional and still goes through review.
                            proposal = reply.get("proposal", {
                                "plan": parent.opening["assignment"],
                                "amount": parent.opening["allowance"],
                            })
                            self.attempt(
                                child,
                                "proposal",
                                lambda c=child, p=proposal: self.record_proposal(
                                    c, p, parent.name, consent=True
                                ),
                            )
                            if child.application and child.application.get("target") == parent.name and child.application.get("status") == "pending":
                                if self.accept_trial(parent, child, reply):
                                    break
                        elif reply.get("accept") is False:
                            self.emit("application_declined", agent=child.name, target=parent.name)
                            # A refusal cannot cause repeated screening from this opening.
                            child.application = {
                                "agent": name,
                                "target": parent.name,
                                "status": "reject",
                                "feedback": "Applicant declined",
                                "attempts": 1,
                            }
                        else:
                            child.last_feedback = {"from": parent.name,
                                                   "text": "Hiring reply needs accept=true or accept=false; no refusal recorded."}
                            self.emit("application_incomplete", agent=child.name, target=parent.name)
                    claimed.add(name)
                    if (
                        child.application
                        and child.application["target"] == parent.name
                        and child.application["status"] == "pending"
                    ):
                        applicants.append(child)
                    break
            if applicants:
                batches[parent.name] = [self.proposal_view(a) for a in applicants]
        reviewed = []
        for name, proposals in batches.items():
            parent = self.agents[name]
            reply = self.ask(parent, "review", extra={"proposals": proposals, "review_target": name})
            reviewed.append((name, proposals, reply))
        for target, proposals, reply in reviewed:
            self.apply_review(target, proposals, reply)
        return [
            self.proposal_view(self.agents[n])
            for n in sorted(slots)
            if n in self.agents
            and self.agents[n].application
            and self.agents[n].application["target"] == "bank"
            and self.agents[n].application["status"] == "pending"
            and not self.agents[n].parent
        ]

    def open_job(self, parent, data):
        if data is None:
            parent.opening = None
            return
        self.check_child_slot(parent)
        if not parent.team or not self.teams[parent.team].funded:
            raise ValueError("A job opening needs a funded manager")
        value = amount(data["allowance"])
        trial = data.get("trial", False)
        if type(trial) is not bool:
            raise ValueError("trial must be boolean")
        candidate = data.get("candidate")
        if candidate is not None and (not isinstance(candidate, str) or candidate not in self.agents
                                      or self.agents[candidate].team or self.agents[candidate].retired):
            raise ValueError("Named candidate must be an available free agent")
        models = ([self.agents[candidate].model] if candidate else self.config.worker_models)
        affordable = [model for model in models if value >= self.working_minimum(model)
                      * (2 if trial or self.config.funding_recovery else 1)]
        if not affordable:
            raise ValueError("Opening must fund working capital for an eligible model")
        screening = min(self.call_budget(m, "application") for m in affordable)
        review = 0 if trial else self.call_budget(parent.model, "review")
        if value + screening + review + self.call_budget(parent.model, "formation") > self.rights(parent):
            raise ValueError("Opening needs working capital plus application and review reserves")
        assignment = clip(data["assignment"], self.config.text_bytes)
        if not assignment.strip():
            raise ValueError("Opening needs an assignment")
        parent.opening = {
            "assignment": assignment,
            "allowance": value,
            "expires": self.round + self.config.opening_rounds - 1,
            "trial": trial,
        }
        if candidate:
            parent.opening["candidate"] = candidate
        self.emit("job_opened", agent=parent.name, **parent.opening)

    def update_board(self, rows):
        before = self.board.view()
        self.board.replace(rows)
        for old in before["tasks"]:
            owner = self.agents.get(old["owner"])
            if not owner:
                continue
            item = self.board.items.get(old["id"])
            if item != old:
                owner.last_feedback = {
                    "from": "bank_board",
                    "text": clip(
                        f"Task {old['id']}: "
                        + (item["status"] + ". " + item["feedback"] if item else "removed"),
                        self.config.text_bytes,
                    ),
                }
                owner.wake_round = self.round
        self.emit("board_updated", board=self.board.view())

    def fund(self, proposals, *, planning=False):
        self.phase = "fund"
        if not self.bank_review_due(proposals, planning):
            return
        if not proposals and not self.board.enabled:
            return
        if len(proposals) > self.config.bank_contacts:
            raise ValueError("Bank batch exceeds contact cap")
        required = {
            "phase": "bank",
            "round": self.round,
            "max_rounds": self.config.max_rounds,
            "task": self.task.problem,
            "society_wealth": self.ledger.wealth,
            "objective": self.objective(),
            "bank_spendable": self.bank_spendable(),
            "bank_grantable": self.bank_grantable(),
            "max_grant": self.config.max_grant,
            "proposals": proposals,
            "minimum_working_budget": self.call_credits,
            "application_budget_remaining": self.config.application_budget - self.application_spent,
            "max_revisions": self.config.max_revisions,
            "brief": self.bank_brief,
            "schema": deepcopy(SCHEMAS["bank"]),
            "last_protocol_error": self.format_errors.get("bank"),
            "introduction_targets": self.bank_introduction_targets(),
        }
        rules = BANK_RULES
        if self.config.hierarchy_bias:
            rules += HIERARCHY_RULES
        if self.config.standing_grants:
            required["schema"]["decisions"][0]["renewal"] = RENEWAL_SCHEMA
            required["standing_grant_rounds"] = self.config.standing_grant_rounds
            rules += RENEWAL_RULES
        if self.config.progress_control:
            required["stalled_rounds"] = self.stalled_rounds()
            rules += (" When work stalls, request a different approach, a scoped collaborator of any model, "
                      "or a concrete failing test. Do not renew identical unsuccessful plans indefinitely.")
        if proposals:
            required["schema"]["decisions"][0]["amount"] = min(
                proposals[0]["amount"], proposals[0]["minimum_working_budget"] * 3
            )
        evidence_teams = [self.agents[p["agent"]].team for p in proposals]
        if self.board.enabled:
            required["bank_board"] = self.board.view()
            required["initial_planning"] = planning
            required["schema"]["tasks"] = BOARD_SCHEMA
            rules += BOARD_RULES
            # A fixed number of compact progress receipts, never a global roster.
            owned = sorted({item["team"] for item in self.board.items.values() if item["team"]})
            start = (self.round - 1) * self.config.bank_contacts % max(1, len(owned))
            evidence_teams = list(dict.fromkeys(evidence_teams + owned[start:] + owned[:start]))
            evidence_teams = [team for team in evidence_teams if team][: self.config.bank_contacts]
            if planning:
                rules += (
                    "\nBefore the first applications, publish a small initial board; decisions must be empty."
                )
        if self.workbench is not None:
            required["verified_evidence"] = self.workbench.bank_evidence(evidence_teams)
            if planning:
                required["source_inventory"] = sorted(self.workbench.hashes(self.workbench.public))[:32]
        if self.config.funding_recovery:
            required["funding_status"] = [
                self.funding_receipt(self.agents[self.teams[t].manager])
                for t in dict.fromkeys(evidence_teams) if t in self.teams
            ][:self.config.bank_contacts]
            rules += (
                "\nA runtime_recovery request reports blocked or low liquidity, not a new task. "
                "Use funding_status to distinguish inability to pay from unwillingness to work. "
                "Fund at least minimum_grant and preferably several calls plus delegated budgets. "
                "Total grants in this batch must fit bank_grantable, which retains bank operating cash. "
                "Keeping treasury idle has no value when a useful next experiment has positive expected net reward. "
                "Board entries are broad portfolios: recruit complementary subwork inside the owner, "
                "including stronger-model escalation for unresolved failures. Do not prescribe model roles. "
                "Shell execution receipts are evidence of commands, not necessarily test success; "
                "zero dedicated tests does not imply no shell tests ran."
            )
            required["last_review_error"] = getattr(self, "last_review_error", None)
        if self.config.bank_finalization:
            required["team_readiness"] = self.readiness_receipts()
            required["schema"].update(finish=False, finish_reason="Why the whole integrated artifact is ready")
            rules += (
                "\nOnly you may finalize society work. Workers' ready reports concern their own teams. "
                "finish=true requires all board outcomes claimed/done with current team readiness, "
                "a successful integrated build, and no integration conflicts. Assess broader documented "
                "coverage before stopping; a few public probes are not full coverage. "
                "Use finish_reason and finalize the integrated public artifact."
            )
        if self.config.quality_first:
            rules = quality_rules(self.config, "bank")
            if planning:
                rules += " Initial planning: publish broad complementary outcomes with interfaces; decisions must be empty."
            required["objective"]["primary"] = "Maximize correctness and completeness within the hard cap."
            if proposals:
                required["schema"]["decisions"][0]["amount"] = proposals[0]["amount"]
        messages, _ = build_review_context(
            rules, required, [{"memory": self.bank_memory}], self.context_limit("bank")
        )
        text = self.call(self.config.bank_model, messages, kind="bank")
        if text is not None:
            self.last_bank_review = self.round
            self.force_bank_review = False
        reply = self.parse(text, set(required["schema"]), "bank")
        if self.board.enabled and "tasks" in reply:
            try:
                self.update_board(reply["tasks"])
            except (ValueError, TypeError, KeyError) as exc:
                self.format_errors["bank"] = "Board update rejected: " + clip(str(exc), 250)
                self.emit("board_rejected", reason=self.format_errors["bank"])
                # Do not grant on the basis of a board update that did not happen.
                return
        if isinstance(reply.get("brief"), str):
            self.bank_brief = clip(reply["brief"], self.config.text_bytes)
            self.emit("bank_brief", text=self.bank_brief)
        if isinstance(reply.get("memory"), str):
            self.bank_memory = clip(reply["memory"], self.config.text_bytes)
        self.apply_review("bank", proposals, reply)
        if self.config.bank_finalization and reply.get("finish") is True:
            self.finalize_society(reply.get("finish_reason", ""))

    def apply_review(self, target, proposals, reply):
        decisions = reply.get("decisions", [])
        if not isinstance(decisions, list) or len(decisions) > len(proposals):
            self.emit("invalid_reply", actor=target, reason="Review exceeds admitted proposal count")
            return
        if target != "bank" and isinstance(reply.get("memory"), str):
            self.agents[target].memory = clip(reply["memory"], self.config.text_bytes)
        presented = {p["agent"]: p for p in proposals}
        seen = set()
        for decision in decisions:
            try:
                name = decision["agent"]
                if name not in presented or name in seen:
                    raise ValueError("Review requires a unique presented applicant")
                seen.add(name)
                agent = self.agents[name]
                app = agent.application
                if not app or app["target"] != target or app["status"] != "pending":
                    raise ValueError("Application is stale or already decided")
                status = decision["decision"]
                if status not in {"approve", "revise", "reject"}:
                    raise ValueError("Unknown review decision")
                feedback = clip(decision.get("feedback", ""), self.config.text_bytes)
                if not feedback.strip():
                    raise ValueError("Every decision requires short feedback")
                if "introduce_to" in decision:
                    if target != "bank" or status not in {"reject", "revise"}:
                        raise ValueError("Bank introductions require a revised/rejected free applicant")
                    self.introduce(agent, decision["introduce_to"], "bank", feedback)
                if status == "approve" and target == "bank" and self.board.enabled:
                    conflict = self.board.conflict(agent.name, app.get("task_id"))
                    if conflict:
                        status, feedback = "revise", clip(conflict, self.config.text_bytes)
                        self.emit("proposal_overlap", agent=name, task_id=app.get("task_id"), reason=feedback)
                if status == "approve":
                    value = amount(decision["amount"])
                    renewal = (self.renewal_spec(decision["renewal"], app["amount"], value, agent)
                               if "renewal" in decision else None)
                    minimum = (1 if agent.team and app.get("recruitment") else
                               self.working_minimum(agent.model) * (2 if self.config.funding_recovery else 1))
                    if not minimum <= value <= app["amount"]:
                        raise ValueError(
                            "Approval needs minimum_grant working capital and cannot exceed request"
                        )
                    if target == "bank":
                        if agent.parent:
                            raise ValueError("Only roots can receive bank grants")
                        # The bank's own response has just incurred a charge.
                        # An approval authorizes up to this amount, never a
                        # transfer above the post-call treasury envelope.
                        affordable = min(value, self.bank_grantable())
                        if affordable < minimum:
                            raise ValueError("Bank operating cash leaves less than minimum_grant available")
                        if affordable < value:
                            self.emit("grant_limited", agent=name, approved_maximum=value, transferred=affordable)
                            value = affordable
                            decision = {**decision, "amount": value}
                        self.grant(agent, value, task_id=app.get("task_id"))
                        agent.bank_attempts = 0
                    else:
                        parent = self.agents[target]
                        if agent.parent == target:
                            allocation = {"agent": name, "amount": value}
                            if not app.get("runtime_recovery"):
                                allocation["assignment"] = app["plan"]
                            self.allocate(parent, [allocation])
                        elif not agent.team and app["consent"] and parent.opening:
                            if value > parent.opening["allowance"] or parent.opening["expires"] < self.round:
                                raise ValueError("Approval exceeds current job offer")
                            if value + self.call_budget(parent.model, "formation") > self.rights(parent):
                                raise ValueError("Hiring must retain the owner's next coordination call")
                            self.hire(parent, agent, value, parent.opening["assignment"])
                            parent.opening = None
                        else:
                            raise ValueError("No valid local proposal or consenting applicant")
                    if target == "bank":
                        self.last_review_error = None
                    if renewal:
                        self.authorize_renewal(agent, target, renewal)
                elif status == "revise" and app["attempts"] >= 1 + self.config.max_revisions:
                    status = "reject"
                    feedback = clip("Revision limit reached. " + feedback, self.config.text_bytes)
                app["status"] = "approved" if status == "approve" else status
                app["feedback"] = feedback
                agent.application = app
                defer = status == "reject" and (app.get("recruitment") or (agent.team and self.config.funding_recovery) or
                                                (app.get("runtime_recovery") and self.config.standing_grants))
                if status == "reject" and target == "bank" and not defer:
                    agent.bank_closed = True
                agent.wake_round = self.round
                if app.get("recruitment"):
                    self.finish_recruitment_review(agent, status)
                self.emit(
                    "proposal_review",
                    reviewer=target,
                    agent=name,
                    decision=status,
                    amount=decision.get("amount", 0) if status == "approve" else 0,
                    feedback=feedback,
                )
                if status == "approve":
                    # Preserve latest feedback while permitting later top-up proposals.
                    agent.application = None
                    agent.last_feedback = {"from": target, "text": feedback}
                elif defer:
                    agent.application = None
                    agent.standing_grant = None
                    agent.funding_retry_round = self.round + self.config.bank_review_interval
                    agent.last_feedback = {"from": target, "text": feedback, "retry_round": agent.funding_retry_round}
                    self.emit("funding_deferred", agent=name, target=target,
                              retry_round=agent.funding_retry_round)
            except (ValueError, TypeError, KeyError) as exc:
                if target == "bank":
                    self.last_review_error = clip(str(exc), 300)
                self.emit("review_rejected", actor=target, reason=clip(str(exc), 300))
