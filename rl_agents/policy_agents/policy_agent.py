from rl_agents.agent import BaseAgent
from rl_agents.memory.replay_memory import ReplayMemory

import torch
from abc import ABC, abstractmethod


class AbstractPolicyAgent(BaseAgent, ABC):
    
    _rollout_memory = None
    @property
    def rollout_memory(self) -> ReplayMemory:
        if self._rollout_memory is None: raise AttributeError(f"Please set a rollout_memory for {self.__class__.__name__}")
        return self._rollout_memory
    
    @rollout_memory.setter
    def rollout_memory(self, val) -> None:
        self._rollout_memory = val
