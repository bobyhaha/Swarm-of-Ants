"""Bounded referrals, indexed candidate discovery, and priced recruitment options."""

from collections import defaultdict, deque
from itertools import islice
import re

from .context import clip


def keywords(text):
    stop = {"with", "from", "this", "that", "will", "have", "work", "task", "team",
            "your", "only", "when", "then", "their", "into", "them", "should"}
    return list(dict.fromkeys(w for w in re.findall(r"[a-z][a-z0-9_]{3,}", text.lower())
                              if w not in stop))[:24]


class CandidateIndex:
    """One round's host-side index; no agent receives the population roster."""

    def __init__(self, agents, round_number, degree):
        self.agents = {a.name: a for a in agents}
        names = sorted(self.agents)
        offset = (round_number * degree) % max(1, len(names))
        names = names[offset:] + names[:offset]
        self.rank = {name: i for i, name in enumerate(names)}
        self.by_model, self.by_term, self.terms = defaultdict(deque), defaultdict(deque), {}
        for name in names:
            a = self.agents[name]
            words = keywords((a.application or {}).get("plan", "") + " " + (a.discovery_offer or {}).get("plan", "") + " " + a.trainable_system_prompt)
            self.terms[name] = set(words)
            self.by_model[a.model].append(name)
            for word in words:
                self.by_term[a.model, word].append(name)

    def match(self, assignment, models, claimed, limit):
        words = set(keywords(assignment)[:12])
        candidates = set()
        for model in models:
            queues = [self.by_term[model, w] for w in sorted(words)] + [self.by_model[model]]
            for queue in queues:
                # A claimed candidate is removed once per posting, amortizing
                # queue cleanup across all managers in this formation phase.
                while queue and queue[0] in claimed:
                    queue.popleft()
                candidates.update(n for n in islice(queue, limit) if n not in claimed)
        return sorted(candidates, key=lambda n: (-len(words & self.terms[n]), self.rank[n]))[:limit]


class Recruitment:
    def recruitment_costs(self, model):
        own = self.call_budget(model) * 2
        review = self.call_budget(model, "review")
        return [{"model": child, "child_working_capital": self.working_minimum(child) * 2,
                 "spawn_total": own + self.working_minimum(child) * 2,
                 "trial_hire_total": own + self.working_minimum(child) * 2 + self.call_budget(child, "application"),
                 "reviewed_hire_total": own + self.working_minimum(child) * 2 + self.call_budget(child, "application") + review}
                for child in self.config.worker_models]

    def clear_introduction(self, candidate, reason):
        referral = candidate.referral
        if not referral:
            return
        owner = self.agents.get(referral["target"])
        if owner and candidate.name in owner.introductions:
            owner.introductions.remove(candidate.name)
        candidate.referral = None
        self.emit("introduction_closed", agent=candidate.name, target=referral["target"], reason=reason)

    def prune_introductions(self):
        for candidate in self.agents.values():
            referral = candidate.referral
            if not referral:
                continue
            owner = self.agents.get(referral["target"])
            if (candidate.team or candidate.retired or referral["expires"] < self.round
                    or not owner or owner.retired or owner.team != referral["team"]
                    or not owner.team or not self.teams[owner.team].funded):
                self.clear_introduction(candidate, "expired or membership changed")

    def introduce(self, candidate, target, source, reason):
        owner = self.agents.get(target)
        if candidate.team or candidate.retired or not owner or owner.retired or not owner.team:
            raise ValueError("Introduction requires a free applicant and a funded owner")
        if not self.teams[owner.team].funded or not self.board.can_work(owner.team):
            raise ValueError("Introduction target cannot work")
        old = candidate.referral
        if source not in {"bank", "offer"}:
            manager = self.agents.get(source)
            if (not manager or not old or old["target"] != source or old["expires"] < self.round
                    or target not in manager.children or owner.team != manager.team):
                raise ValueError("Forward only your current introduction to a direct child")
        elif old:
            raise ValueError("Applicant already has a pending introduction")
        if len(owner.introductions) >= self.config.manager_contacts:
            raise ValueError("Owner introduction queue is full")
        # Capture one proposal, not its history. Routing is bookkeeping; the
        # owner's future screening call must still be explicitly sponsored.
        referral = {"target": target, "team": owner.team, "source": source,
                    "expires": old["expires"] if old else self.round + self.config.opening_rounds,
                    "plan": old["plan"] if old else clip((candidate.application or candidate.discovery_offer or {}).get("plan", ""), self.config.text_bytes),
                    "reason": clip(reason, self.config.text_bytes)}
        self.clear_introduction(candidate, "forwarded")
        candidate.referral = referral
        owner.introductions.append(candidate.name)
        owner.wake_round = self.round
        self.emit("introduction_routed", agent=candidate.name, **referral)

    def introduction_views(self, owner):
        return [{"agent": name, "model": self.agents[name].model,
                 **self.agents[name].referral,
                 "minimum_trial_allowance": self.working_minimum(self.agents[name].model) * 2}
                for name in owner.introductions if self.agents[name].referral
                and self.agents[name].referral["target"] == owner.name]

    def forward_introductions(self, owner, data):
        if not isinstance(data, list) or len(data) > self.config.manager_contacts:
            raise ValueError("Introduction forwarding exceeds local review cap")
        seen = set()
        for row in data:
            name = row["agent"]
            if name in seen or name not in owner.introductions:
                raise ValueError("Forward only unique presented introductions")
            seen.add(name)
            self.introduce(self.agents[name], row["to"], owner.name, row.get("reason", "Scoped collaboration"))

    def bank_introduction_targets(self):
        roots = sorted(t.manager for t in self.teams.values() if t.funded and self.board.can_work(t.id))
        start = (self.round - 1) * self.config.bank_contacts % max(1, len(roots))
        return [{"agent": name, "team": self.agents[name].team,
                 "child_slots": self.config.max_children - len(self.agents[name].children),
                 "introduction_slots": self.config.manager_contacts - len(self.agents[name].introductions)}
                for name in (roots[start:] + roots[:start])[:self.config.bank_contacts]]

    def candidate_index(self, active):
        return CandidateIndex([a for a in active if not a.team and not a.retired and not a.referral
                               and not (a.application and a.application["target"] != "bank"
                                        and a.application["status"] in {"pending", "revise"})],
                              self.round, self.config.peer_contacts)

    def recruitment_candidates(self, owner, pending, index, claimed):
        opening = owner.opening
        if opening.get("candidate"):
            return [opening["candidate"]]
        models = [m for m in self.config.worker_models if opening["allowance"] >= self.working_minimum(m)
                  * (2 if opening.get("trial") or self.config.funding_recovery else 1)]
        matched = index.match(opening["assignment"], models, claimed, self.config.peer_contacts)
        # Every collection here has a configured cap; no full roster reaches an LLM.
        return list(dict.fromkeys(pending + ([o["agent"] for o in self.offers_for(owner)] if self.config.quality_first else []) + owner.introductions + matched))

    def hiring_reserve(self, owner, child, opening):
        return (opening["allowance"] + self.call_budget(child.model, "application")
                + (0 if opening.get("trial") else self.call_budget(owner.model, "review"))
                + self.call_budget(owner.model, "formation"))

    def accept_trial(self, owner, child, reply):
        opening = owner.opening
        if not opening or not opening.get("trial") or reply.get("accept") is not True:
            return False
        proposal = reply.get("proposal")
        # Changed scope or price goes through ordinary review, never auto-consent.
        if proposal is not None and (not isinstance(proposal, dict)
                or proposal.get("plan") != opening["assignment"]
                or proposal.get("amount") != opening["allowance"]
                or set(proposal) - {"plan", "amount"}):
            return False
        if self.rights(owner) < opening["allowance"] + self.call_budget(owner.model, "formation"):
            return False
        self.apply_review(owner.name, [self.proposal_view(child)], {"decisions": [{
            "agent": child.name, "decision": "approve", "amount": opening["allowance"],
            "feedback": "Accepted the preapproved trial. Deliver this scope within the allowance; report evidence and stop or request a justified renewal."}]})
        if child.parent != owner.name:
            return False
        self.emit("trial_hired", agent=child.name, parent=owner.name,
                  allowance=opening["allowance"], assignment=opening["assignment"])
        return True
