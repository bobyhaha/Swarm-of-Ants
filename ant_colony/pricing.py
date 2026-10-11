"""Model-specific, conservative grants; API receipts remain the dollar authority."""

from decimal import Decimal, ROUND_CEILING
from .reservations import request_reservation


class Pricing:
    def output_limit(self, kind="work"):
        cfg = self.config
        if kind == "trial":
            return min(cfg.max_output_tokens, cfg.trial_max_output_tokens)
        return (min(cfg.max_output_tokens, cfg.coordination_max_output_tokens)
                if cfg.request_pricing and kind in {"bank", "formation", "review", "application", "discovery", "trial"}
                else cfg.max_output_tokens)

    def context_limit(self, kind="work"):
        cfg = self.config
        return (min(cfg.context_bytes, cfg.coordination_context_bytes)
                if cfg.request_pricing and kind in {"bank", "formation", "review", "application", "discovery", "trial"}
                else cfg.context_bytes)

    def call_budget(self, model, kind="work", messages=None):
        cfg = self.config
        if cfg.cost_mode == "output_tokens":
            return self.output_limit(kind) * cfg.token_price
        settings = self.provider.models[model]
        _, dollars = request_reservation(settings, self.output_limit(kind), enabled=cfg.request_pricing,
                                         messages=messages, context_bytes=self.context_limit(kind))
        return self.dollar_credits(dollars)

    def working_minimum(self, model):
        return self.call_budget(model, "trial" if self.config.quality_first else "work")

    def work_kind(self, agent):
        return ("trial" if self.config.quality_first and self.rights(agent) < self.call_budget(agent.model)
                else "work")

    def dollar_credits(self, dollars):
        value = Decimal(str(dollars))
        if not value.is_finite() or value < 0:
            raise ValueError("Invalid API charge")
        return int((value * self.config.credits_per_usd).to_integral_value(rounding=ROUND_CEILING))

    def generation_credits(self, generation):
        if self.config.cost_mode == "output_tokens":
            return generation.output_tokens * self.config.token_price
        if generation.estimated:
            raise ValueError("Dollar-priced credits require verified API receipts")
        return self.dollar_credits(generation.cost_usd)

    def pricing_view(self):
        if self.config.cost_mode == "output_tokens":
            return {"mode": "output_tokens", "credits_per_output_token": self.config.token_price}
        return {
            "mode": "api_usd",
            "credits_per_usd": self.config.credits_per_usd,
            "rule": "Pay ceil(actual API cost * credits_per_usd), including billed input and reasoning. Transfers do not create wealth.",
            "models": {
                name: {
                    "id": self.provider.models[name]["id"],
                    "input_usd_per_million_ceiling": self.provider.models[name]["input_usd_per_million"],
                    "output_usd_per_million_ceiling": self.provider.models[name]["output_usd_per_million"],
                    "reasoning_effort": self.provider.models[name].get("reasoning_effort"),
                    "maximum_call_credits": self.call_budget(name),
                }
                for name in dict.fromkeys((self.config.bank_model, *self.config.worker_models))
            },
        }
