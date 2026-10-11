"""EoM agent identities with a separate cooperative budget and recruitment tree."""

from dataclasses import dataclass, field
from hayekmas.base.agent import BaseAgent, BaseAction
from .protocol import RULES


class ColonyAction(BaseAction):
    def __init__(self, text, author):
        self.text = text
        self.author = author


class ColonyAgent(BaseAgent):
    ROLE = "colony"
    FROZEN_SYSTEM_PROMPT = RULES

    def __init__(self, name, model, strategy, born_round=0):
        super().__init__(name=name, initial_wealth=0)
        self.model = model
        self.trainable_system_prompt = strategy
        self.team = None
        self.parent = None
        self.children = []
        self.born_round = born_round
        self.wake_round = born_round
        self.retired = False
        self.memory = ""
        self.assignment = "Explore the task; recruit or propose only when useful."
        self.inbox = []
        self.read_keys = []
        # One current application, never an accumulating conversation history.
        self.application = None
        self.last_feedback = None
        self.discovery_calls = 0
        self.discovery_result = None
        self.discovery_offer = None
        self.directory = []
        self.bank_attempts = 0
        self.bank_closed = False
        self.opening = None
        self.referral = None
        self.introductions = []
        self.recruitment_plan = None
        self.review_cursor = 0
        self.standing_grant = None
        self.funding_retry_round = 0
        self.productive_steps = self.subtree_progress = 0
        self.last_tested_revision = None

    def match_wakeup_condition(self, env):
        # No model call: the scheduler knows when the agent requested a wakeup.
        return not self.retired and self.wake_round <= env.round

    def act(self, env):
        return ColonyAction(self.memory, self.name)


@dataclass
class Team:
    id: str
    manager: str
    members: set = field(default_factory=set)
    allowances: dict = field(default_factory=dict)
    scratchpad: dict = field(default_factory=dict)
    expires: int = 0
    funded: bool = False
    board_task_id: str | None = None
    ready: dict | None = None

    def to_dict(self):
        return {
            "id": self.id,
            "manager": self.manager,
            "members": sorted(self.members),
            "allowances": self.allowances.copy(),
            "scratchpad": self.scratchpad.copy(),
            "expires": self.expires,
            "funded": self.funded,
            "board_task_id": self.board_task_id,
            "ready": self.ready,
        }
