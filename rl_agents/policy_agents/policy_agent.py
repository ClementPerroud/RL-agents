from rl_agents.agent import AbstractAgent
from rl_agents.replay_memory.replay_memory import ReplayMemory

import torch
from abc import ABC, abstractmethod


class AbstractPolicyAgent(AbstractAgent, ABC):

    _rollout_memory = None
    @property
    def rollout_memory(self) -> int:
        if self._rollout_memory is None: raise AttributeError(f"{self.__class__.__name__}.rollout_memory is not set")
        return self._rollout_memory
    @rollout_memory.setter
    def rollout_memory(self, val): self._rollout_memory = val
