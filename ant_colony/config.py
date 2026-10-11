"""All scalability limits are constants chosen before an experiment."""

from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class ColonyConfig:
    initial_agents: int = 6
    max_agents: int = 128
    max_rounds: int = 8
    bank_contacts: int = 2
    manager_contacts: int = 2
    application_budget: int = 12_000
    max_revisions: int = 2
    opening_rounds: int = 3
    peer_contacts: int = 2
    max_children: int = 2
    max_depth: int = 8
    work_turns: int = 2
    initial_bank: int = 100_000
    token_price: int = 1
    max_grant: int = 20_000
    grant_rounds: int = 6
    reward: int = 50_000
    max_output_tokens: int = 768
    # Serialized messages, including immutable rules and the response schema.
    context_bytes: int = 24_000
    coding_output_bytes: int = 6_000
    bank_board_size: int = 0
    task_bytes: int = 6_000
    text_bytes: int = 900
    strategy_bytes: int = 900
    answer_bytes: int = 6_000
    scratch_reads: int = 3
    inbox_size: int = 8
    trail_size: int = 64
    trail_ttl: int = 4
    max_sleep_rounds: int = 20
    rent_per_round: int = 0
    bank_model: str = "bank"
    worker_models: tuple[str, ...] = ("worker",)
    initial_models: tuple[str, ...] = ()
    population_seed: int = 0
    cost_mode: str = "output_tokens"
    credits_per_usd: int = 1_000_000
    funding_recovery: bool = False
    persistent_teams: bool = False
    bank_finalization: bool = False
    request_pricing: bool = False
    coordination_max_output_tokens: int = 4096
    coordination_context_bytes: int = 32000
    event_driven_bank: bool = False
    bank_review_interval: int = 4
    standing_grants: bool = False
    standing_grant_rounds: int = 6
    progress_control: bool = False
    hierarchy_bias: bool = False
    quality_first: bool = False
    discovery_contacts: int = 8
    discovery_turns: int = 3
    discovery_budget: int = 0
    message_contacts: int = 4
    trial_max_output_tokens: int = 2048
    stagnation_rounds: int = 6

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in {"funding_recovery", "persistent_teams", "bank_finalization",
                              "request_pricing", "event_driven_bank", "standing_grants", "progress_control", "hierarchy_bias", "quality_first"}:
                if type(value) is not bool:
                    raise ValueError(field.name + " must be boolean")
                continue
            if field.name in {"worker_models", "bank_model", "initial_models", "cost_mode"}:
                continue
            if type(value) is not int or value < (
                0
                if field.name in {"rent_per_round", "application_budget", "discovery_budget", "max_revisions", "bank_board_size", "population_seed"}
                else 1
            ):
                raise ValueError(
                    f"{field.name} must be a positive integer (rent, application budget and revisions may be zero)"
                )
        if self.initial_agents > self.max_agents:
            raise ValueError("initial_agents exceeds max_agents")
        if self.cost_mode not in {"output_tokens", "api_usd"}:
            raise ValueError("Unknown cost_mode")
        if not isinstance(self.initial_models, tuple) or (
            self.initial_models and (
                len(self.initial_models) != self.initial_agents
                or any(m not in self.worker_models for m in self.initial_models)
            )
        ):
            raise ValueError("initial_models must exactly name the initial population using allowed aliases")
        if self.context_bytes < self.task_bytes + self.strategy_bytes + 8_000:
            raise ValueError("context_bytes must leave 8000 bytes for rules and structural context")
        if self.coding_output_bytes > 32_000:
            raise ValueError("coding_output_bytes cannot exceed 32000")
        if self.bank_board_size > 8:
            raise ValueError("bank_board_size cannot exceed 8")
        if not isinstance(self.bank_model, str) or not self.bank_model:
            raise ValueError("bank_model must name a model alias")
        if (
            not isinstance(self.worker_models, tuple)
            or not self.worker_models
            or any(not isinstance(m, str) or not m for m in self.worker_models)
        ):
            raise ValueError("worker_models must contain model aliases")
        if self.initial_bank < self.max_output_tokens * self.token_price:
            raise ValueError("Treasury cannot afford even one bounded call")

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key in ("worker_models", "initial_models"):
            if key in data:
                if not isinstance(data[key], list):
                    raise ValueError(key + " must be an array")
                data[key] = tuple(data[key])
        return cls(**data)

    def to_dict(self):
        return asdict(self)
