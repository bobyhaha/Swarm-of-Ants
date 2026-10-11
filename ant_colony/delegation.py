"""Agent-chosen recruitment funding, reviewed through the existing local queue."""

from .context import clip
from .ledger import amount


class DelegationFunding:
    def request_recruitment(self, agent, data):
        if not agent.team or not self.teams[agent.team].funded or not self.board.can_work(agent.team):
            raise ValueError("Recruitment funding requires an existing funded outcome")
        if data is None:
            if agent.application and agent.application.get("recruitment"):
                agent.application = None
            agent.recruitment_plan = None
            self.emit("recruitment_request_cancelled", agent=agent.name)
            return
        if agent.funding_retry_round > self.round:
            raise ValueError("Wait until funding_retry_round before another funding request")
        self.check_child_slot(agent)
        if not isinstance(data, dict) or set(data) - {"mode", "model", "assignment", "allowance", "reason", "candidate", "trial"}:
            raise ValueError("Use the bounded recruitment_request schema")
        mode, model = data["mode"], data["model"]
        if mode not in {"hire", "spawn"} or model not in self.config.worker_models:
            raise ValueError("Choose hire or spawn and an allowed model")
        if mode == "spawn" and len(self.agents) >= self.config.max_agents:
            raise ValueError("Population cap prevents this spawn")
        trial = data.get("trial", True)
        if type(trial) is not bool:
            raise ValueError("trial must be boolean")
        child_capital = amount(data["allowance"])
        if child_capital < 2 * self.working_minimum(model):
            raise ValueError("Recruitment funding needs two child work-call reserves")
        candidate = data.get("candidate")
        if candidate is not None:
            child = self.agents.get(candidate) if isinstance(candidate, str) else None
            if (mode != "hire" or not child or child.team or child.retired or child.model != model
                    or (child.referral and child.referral["target"] != agent.name)):
                raise ValueError("Candidate must be an available matching applicant for this owner")
        limit = min(450, self.config.text_bytes)
        assignment, reason = clip(data["assignment"], limit), clip(data["reason"], limit)
        if not assignment.strip() or not reason.strip():
            raise ValueError("Recruitment needs a distinct outcome and evidence/cost justification")
        commitments = sum(c.application["amount"] for n in agent.children
                          if (c := self.agents[n]).application
                          and c.application.get("target") == agent.name
                          and c.application.get("status") == "pending")
        quote = {"child_capital": child_capital, "own_working_capital": 2 * self.call_budget(agent.model),
                 "coordination_buffer": self.call_budget(agent.model, "formation"),
                 "screening": self.call_budget(model, "application") if mode == "hire" else 0,
                 "review": self.call_budget(agent.model, "review") if mode == "hire" and not trial else 0,
                 "pending_child_commitments": commitments}
        required = sum(quote.values())
        shortfall = max(0, required - self.rights(agent))
        if shortfall > self.config.max_grant:
            raise ValueError("Recruitment shortfall exceeds max_grant; propose a smaller scope")
        target = agent.parent or "bank"
        # Never replace a pending decision with a larger request behind its reviewer.
        if agent.application and agent.application["status"] == "pending":
            raise ValueError("Await the pending funding decision before requesting recruitment")
        plan = {"mode": mode, "model": model, "assignment": assignment, "allowance": child_capital,
                "reason": reason, "trial": trial, "team": agent.team, "target": target,
                "owner_assignment": agent.assignment,
                "expires": self.round + self.config.grant_rounds - 1,
                "status": "pending" if shortfall else "affordable",
                "required_balance": required, "requested_topup": shortfall, "costs": quote}
        if candidate is not None:
            plan["candidate"] = candidate
        if shortfall:
            self.record_proposal(agent, {"plan": clip("Delegate outcome: " + assignment + ". Justification: " + reason,
                                                     self.config.text_bytes),
                                         "amount": shortfall, "task_id": self.teams[agent.team].board_task_id}, target)
            agent.application["runtime_recovery"] = True
            # The proposal already contains outcome and justification. Do not
            # duplicate prose or expose descendants in the review context.
            agent.application["recruitment"] = {k: plan[k] for k in
                ("mode", "model", "allowance", "required_balance", "requested_topup", "costs")}
        elif agent.application and agent.application.get("recruitment"):
            agent.application = None
        agent.recruitment_plan = plan
        self.emit("recruitment_funding_requested" if shortfall else "recruitment_already_affordable",
                  agent=agent.name, target=target, **{k: v for k, v in plan.items() if k != "target"})

    def prune_recruitment_plans(self):
        for agent in self.agents.values():
            plan = agent.recruitment_plan
            if not plan:
                continue
            if (plan["expires"] < self.round or plan["team"] != agent.team
                    or plan["target"] != (agent.parent or "bank") or plan["owner_assignment"] != agent.assignment):
                if agent.application and agent.application.get("recruitment"):
                    agent.application = None
                agent.recruitment_plan = None
                self.emit("recruitment_request_expired", agent=agent.name)

    def finish_recruitment_review(self, agent, status):
        plan = agent.recruitment_plan
        if not plan:
            return
        plan["status"] = ("funded" if self.rights(agent) >= plan["required_balance"] else "partial") if status == "approve" else status
        self.emit("recruitment_funding_reviewed", agent=agent.name, status=plan["status"],
                  allowance=self.rights(agent), required_balance=plan["required_balance"])

    def recruitment_completed(self, agent):
        if agent.recruitment_plan:
            self.emit("funded_recruitment_executed", agent=agent.name,
                      funding_status=agent.recruitment_plan["status"])
            if agent.application and agent.application.get("recruitment"):
                agent.application = None
            agent.recruitment_plan = None
