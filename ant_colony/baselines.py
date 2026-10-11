"""Matched tool access without paid social orchestration."""

from dataclasses import replace

from .engine import ColonyMAS


def solo_config(config, model):
    """One permitted actor, with the same work/context limits and no bank inference."""
    if model not in config.worker_models:
        raise ValueError("Solo model must be an allowed worker model")
    return replace(config, initial_agents=1, max_agents=1, initial_models=(model,),
                   worker_models=(model,), bank_model=model, bank_board_size=0,
                   max_grant=config.initial_bank, funding_recovery=False,
                   bank_finalization=False)


class SoloMAS(ColonyMAS):
    """One model works directly; the passive treasury funds its sole workspace."""

    def formation(self):
        self.phase = "formation"
        agent = next(iter(self.agents.values()))
        if not agent.team:
            self.grant(agent, self.bank_spendable())
        return []

    def fund(self, proposals, **kwargs):
        return None
