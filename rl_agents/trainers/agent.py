from rl_agents.agent import Agent, BaseAgent
from rl_agents.service import AgentService
from rl_agents.memory.memory import Memory
from rl_agents.utils.check import assert_is_instance

from abc import ABCMeta
from typing import Protocol, runtime_checkable

@runtime_checkable
class AgentTrainer(Protocol):
    def set_up_and_check(self, agent : BaseAgent):...
    def train_agent(self): ...


class BaseAgentTrainer(AgentService, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._hidden_module = {}

    @property
    def agent(self) -> BaseAgent : return self._hidden_module["agent"]


    def set_up_and_check(self, agent : BaseAgent):
        self._hidden_module["agent"] = agent
        assert_is_instance(self.agent, Agent)