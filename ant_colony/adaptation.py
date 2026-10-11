"""Bounded standing authorizations and progress-driven review at every tree level."""

from .ledger import amount


RENEWAL_SCHEMA = {
    "total": "optional additional credits within the applicant's requested amount after the initial grant",
    "tranche": "credits per renewal, at least two maximum work-call reserves",
    "rounds": "authorization lifetime, no more than the configured standing_grant_rounds",
}

RENEWAL_RULES = (
    " Optional renewal authorizes conditional transfers for this assignment only: total is ADDITIONAL "
    "to the immediate amount, and amount + total cannot exceed the proposal. Renewals require new "
    "source changes or a fresh test of a changed revision by the recipient or its descendants, "
    "low liquidity, available sponsor funds, and an unexpired authorization. Each renewal has a "
    "fixed tranche; no money or inference is free. Omit renewal to require manual review. "
    "A root and every nested manager have the same contract. Roles are chosen by agents."
)


class AdaptiveFunding:
    def renewal_spec(self, spec, request, immediate, agent):
        if not self.config.standing_grants or not isinstance(spec, dict):
            raise ValueError("Standing grants are not enabled or renewal is not an object")
        total, tranche, rounds = amount(spec["total"]), amount(spec["tranche"]), spec["rounds"]
        if not 2 * self.call_budget(agent.model) <= tranche <= total <= request - immediate:
            raise ValueError("Renewal needs two-call tranches within the remaining requested ceiling")
        if type(rounds) is not int or not 1 <= rounds <= self.config.standing_grant_rounds:
            raise ValueError("Renewal lifetime exceeds its fixed limit")
        return {"remaining": total, "tranche": tranche, "expires": self.round + rounds}

    def authorize_renewal(self, agent, target, spec):
        agent.standing_grant = {**spec, "target": target, "team": agent.team,
                                "assignment": agent.assignment, "progress": agent.subtree_progress}
        self.emit("standing_grant_authorized", agent=agent.name, target=target, **spec)

    def aggregate_progress(self):
        # Runtime accounting may scan the population; each agent receives only
        # a scalar summary from each of its bounded number of direct children.
        for agent in sorted(self.agents.values(), key=self.depth, reverse=True):
            agent.subtree_progress = agent.productive_steps + sum(
                self.agents[n].subtree_progress for n in agent.children)

    def standing_renewals(self, active):
        self.aggregate_progress()
        roots = [a.name for a in active if not a.parent and a.standing_grant]
        admitted, self.standing_cursor = self.rotate(
            roots, self.config.bank_contacts, getattr(self, "standing_cursor", 0))
        local_counts = {}
        for a in active:
            line = a.standing_grant
            if not line:
                continue
            target = a.parent or "bank"
            if (line["expires"] < self.round or line["team"] != a.team
                    or line["target"] != target or line["assignment"] != a.assignment):
                a.standing_grant = None
                continue
            if target == "bank" and a.name not in admitted:
                continue
            team = self.teams.get(a.team)
            if (not team or not team.funded or not self.board.can_work(a.team) or team.ready
                    or self.rights(a) >= 2 * self.call_budget(a.model)
                    or a.subtree_progress <= line["progress"]
                    or (a.application and not a.application.get("runtime_recovery"))):
                continue
            value = line["tranche"]
            if value > line["remaining"]:
                continue
            if target == "bank":
                if value > self.bank_grantable():
                    continue
                self.grant(a, value, task_id=team.board_task_id)
            else:
                if local_counts.get(target, 0) >= self.config.manager_contacts:
                    continue
                parent = self.agents[target]
                if value > self.rights(parent) - self.call_budget(parent.model, "review"):
                    continue
                self.allocate(parent, [{"agent": a.name, "amount": value}])
                local_counts[target] = local_counts.get(target, 0) + 1
            line["remaining"] -= value
            line["progress"] = a.subtree_progress
            if a.application and a.application.get("runtime_recovery"):
                a.application = None
            self.emit("standing_grant_used", agent=a.name, target=target, amount=value,
                      remaining=line["remaining"], progress=line["progress"])

    def bank_review_due(self, proposals, planning):
        if not self.config.event_driven_bank:
            return True
        due = (planning or proposals or getattr(self, "force_bank_review", False)
               or self.round - getattr(self, "last_bank_review", 0) >= self.config.bank_review_interval
               or (self.config.bank_finalization and any(
                   t.ready and t.ready["round"] > getattr(self, "last_bank_review", 0)
                   for t in self.teams.values() if t.board_task_id)))
        if not due:
            self.emit("bank_review_skipped", reason="No new proposal, readiness, or scheduled review")
        return bool(due)

    def execute_with_progress(self, agent, action):
        before = self.workbench.summary(agent.team) or {}
        self.workbench.execute(agent, action)
        after = self.workbench.summary(agent.team) or {}
        fresh_test = (after.get("tests", 0) > before.get("tests", 0)
                      and after.get("tested_revision") != agent.last_tested_revision)
        changed = after.get("writes", 0) > before.get("writes", 0)
        if fresh_test:
            agent.last_tested_revision = after.get("tested_revision")
        if changed or fresh_test:
            agent.productive_steps += 1
            self.last_productive_round = self.round
            self.force_bank_review = False
            self.stall_review_round = None
            self.emit("productive_step", agent=agent.name, changed=changed, fresh_test=fresh_test)

    def stalled_rounds(self):
        return max(0, self.round - getattr(self, "last_productive_round", 0))

    def assess_progress(self):
        if not self.config.progress_control or not self.workbench or self.final_answer is not None:
            return
        if self.stalled_rounds() < self.config.stagnation_rounds:
            return
        if getattr(self, "stall_review_round", None) is None:
            self.stall_review_round = self.round
            self.force_bank_review = True
            self.emit("stagnation_review_requested", stalled_rounds=self.stalled_rounds())
            return
        if self.config.quality_first:
            # Quality-first runs keep the hard round/dollar deadline; stagnation
            # requests a strategy review rather than automatically ending work.
            return
        if self.round - self.stall_review_round >= self.config.bank_review_interval:
            self.final_answer = "Selected integrated public artifact after a stalled-work review and grace period."
            self.final_author = "__public__"
            self.submission_mode = "automatic_stagnation_checkpoint"
            self.emit("submission", author="__public__", team="public", text=self.final_answer,
                      candidates=1, selection="public_checkpoint_after_stagnation")
