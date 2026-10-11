"""Terminal evaluation on the original EoM BaseEnv/ResearchEnv interfaces."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
import re

from hayekmas.base.env import BaseEnv
from hayekmas.base.config import RewardConfig
from hayekmas.adapters.researchworld.agent import ResearchAction
from hayekmas.adapters.researchworld.env import ResearchEnv


@dataclass(frozen=True)
class Task:
    id: str
    problem: str
    answer: str
    kind: str = "exact"

    def __post_init__(self):
        if any(not isinstance(v, str) or not v.strip() for v in (self.id, self.problem, self.answer)):
            raise ValueError("Task id, problem and answer/rubric must be nonempty strings")
        if self.kind not in ("exact", "research", "commit0", "programbench"):
            raise ValueError("Task kind must be exact, research, commit0 or programbench")

    @classmethod
    def from_dict(cls, data):
        ref = data.get("rubric", data.get("answer", ""))
        if isinstance(ref, bool) or not isinstance(ref, (str, int, float)):
            raise ValueError("Invalid answer/rubric")
        if isinstance(ref, float) and not math.isfinite(ref):
            raise ValueError("Nonfinite reference answer")
        return cls(
            str(data.get("id", data.get("task_group_id", ""))),
            data["problem"],
            str(ref),
            data.get("kind", "research" if "rubric" in data else "exact"),
        )


class ExactEnv(BaseEnv):
    def __init__(self, task):
        super().__init__()
        self.task = task
        self.initialize()

    def initialize(self):
        self.terminated = False
        self.step_count = 0
        self.action_history = {}
        self.score = None

    def get_task_description(self):
        return self.task.problem

    def get_terminal_score(self):
        return self.score

    def apply(self, action):
        if self.terminated:
            raise RuntimeError("Task already evaluated")
        value = action.text.strip()
        expected = self.task.answer.strip()
        try:
            a, b = Decimal(value), Decimal(expected)
            correct = a.is_finite() and b.is_finite() and a == b
        except InvalidOperation:
            correct = value.casefold() == expected.casefold()
        self.score = float(correct)
        self.terminated = True
        self.step_count = 1
        self.action_history[1] = {"author": action.author, "text": value}
        return self.score


def research_env(task, judge):
    """Use upstream grading prompt/parser, but fail closed on invalid judge output.

    Original score_answer_against_rubric swallows judge exceptions. We record and
    re-raise them after apply rather than reporting an infrastructure error as 0.
    """
    env = ResearchEnv(
        task.problem,
        task.answer,
        reward_config=RewardConfig(terminal_output_bonus_scale=1, center_env_reward=False),
    )
    env.colony_judge_errors = []

    def checked(prompt):
        try:
            result = judge(prompt)
            matches = re.findall(r"(?m)^SCORE:\s*([^\s]+)\s*$", result)
            if len(matches) != 1:
                raise ValueError("Judge must emit exactly one SCORE line")
            score = float(matches[0])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Judge score must lie in [0,1]")
            return result
        except Exception as exc:
            env.colony_judge_errors.append(exc)
            raise

    env.llm_fn = checked
    return env


def evaluate(env, text, author):
    # Avoid interpolating final-answer tags: the model's text is opaque data.
    action = ResearchAction(text, author, role="answer")
    action.is_final = True
    action.final_answer_text = text
    env.apply(action)
    if getattr(env, "colony_judge_errors", []):
        raise env.colony_judge_errors[0]
    score = env.get_path_reward_score()
    if score is None or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Environment must return a finite terminal contribution in [0,1]")
    return score
