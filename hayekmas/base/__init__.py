"""EoM interfaces; original engine and pipelines are loaded only on request."""
from .agent import BaseAction, BaseAgent, AgentStatus, set_agent_id_counter
from .env import BaseEnv


def __getattr__(name):
    from importlib import import_module
    modules = {"HayekMAS": "mas", "TerminationReason": "mas", "Population": "population",
               "Trainer": "pipeline", "Evaluator": "pipeline"}
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(f"hayekmas.base.{modules[name]}"), name)
