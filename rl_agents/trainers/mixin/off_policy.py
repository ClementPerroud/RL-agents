from rl_agents.trainers.agent import BaseAgentTrainer
from rl_agents.memory.sampler import Sampler, RandomSampler, UpdatableSampler
from rl_agents.memory.memory import Memory
from rl_agents.memory.replay_memory import ReplayMemory
from rl_agents.memory.experience import ExperienceLike
from rl_agents.utils.hidden_modules import HiddenModulesUtilsMixin
from rl_agents.utils.assert_check import assert_is_instance

from abc import ABCMeta, abstractmethod
import torch

class OffPolicyTrainerMixin(
        HiddenModulesUtilsMixin,
        BaseAgentTrainer,
        metaclass=ABCMeta
    ):
    def __init__(self, sampler : RandomSampler, train_every : int, batch_size : int, **kwargs): 
        super().__init__(**kwargs)
        self.train_every = train_every
        self.batch_size = batch_size
        self.sampler = sampler

    @property
    def replay_memory(self) -> ReplayMemory: 
        """Hidden reference to avoid infinite recursion with torch recursive method (like .to(), .train() ...)"""
        return self.hidden.replay_memory

    def set_up_and_check(self, agent):
        super().set_up_and_check(agent)
        assert_is_instance(self.agent.memory, Memory)
        self.hidden.replay_memory = self.agent.memory

        assert_is_instance(self.sampler, Sampler)
    
    def train_agent(self) -> float:
        # advantage : [batch]
        if self.agent.nb_step % self.train_every == 0 and len(self.agent.memory) > self.batch_size:
                self.sample_pre_hook()
                batch = self.sampler.sample(self.batch_size)
                experience = self.replay_memory[batch]
                return self.train_step(experience)
        return None
    
    @abstractmethod
    def train_step(self): ...

    def sample_pre_hook(self): ...