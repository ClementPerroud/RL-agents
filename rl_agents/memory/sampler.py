from rl_agents.service import AgentService
from rl_agents.utils.sumtree import SumTree

from abc import ABC, abstractmethod
import numpy as np
import torch
from dataclasses import dataclass, asdict
from functools import partial

from typing import TYPE_CHECKING, Protocol, runtime_checkable, Sized
from rl_agents.memory.replay_memory import BaseExperienceMemory
from rl_agents.memory.memory import Memory, Experience

if TYPE_CHECKING:
    from rl_agents.agent import BaseAgent
    from rl_agents.actor_critic_agent import ActorCriticAgent


@runtime_checkable
class Sampler(Protocol):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    replay_memory : Memory[Experience]
    def sample(self, batch_size : int) -> torch.Tensor: ...


@runtime_checkable
class UpdatableSampler(Protocol):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def store(self, **kwargs): ...
    def update_experiences(self, **kwargs): ...


@runtime_checkable
class WeightableSampler(Protocol):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def apply_weights(self, loss : torch.Tensor, indices : torch.Tensor) -> torch.Tensor:...


class RandomSampler(AgentService, Sampler):
    def __init__(self, replay_memory : "BaseExperienceMemory", **kwargs):
        super().__init__(**kwargs)
        self.replay_memory = replay_memory

    @torch.no_grad()
    def sample(self, batch_size : int):
        return torch.randint(0, len(self.replay_memory), (batch_size,), device=self.replay_memory.device)

    

class LastSampler(AgentService, Sampler):
    def __init__(self, replay_memory : "BaseExperienceMemory", **kwargs):
        super().__init__(**kwargs)
        self.replay_memory = replay_memory
    
    @torch.no_grad()
    def sample(self, batch_size : int): 
        n = len(self.replay_memory)
        return torch.arange(start = n - batch_size, end = n)


class PrioritizedReplaySampler(AgentService, UpdatableSampler, WeightableSampler, Sampler):
    def __init__(self, replay_memory: "BaseExperienceMemory", alpha=0.65, beta_0=0.5, duration=150_000, **kwargs):
        super().__init__(**kwargs)
        self.replay_memory = replay_memory
        self.alpha = alpha
        self.beta_0 = beta_0
        self.duration = duration
        self.priorities = SumTree(size=self.replay_memory.max_length)
        self.random_sampler = RandomSampler(replay_memory=self.replay_memory)
        self.nb_step = 0
    
    def update(self, agent : "BaseAgent", **kwargs):
        self.nb_step = agent.nb_step

    @torch.no_grad()
    def sample(self, batch_size : int):
        if self.nb_step>=self.duration: return self.random_sampler.sample(batch_size=batch_size)

        indices = self.priorities.sample(batch_size=batch_size)
        indices = torch.tensor(indices, device=self.replay_memory.device).long()
        return indices
    
    # Protocol : WeightableSampler
    def apply_weights(self, loss : torch.Tensor, indices : torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            beta = min(1, self.beta_0 + (1 - self.beta_0) * self.nb_step / self.duration)
            weights : torch.Tensor= (len(self.replay_memory) * self.priorities[indices] / (self.priorities.sum() + 1E-8)) ** (-beta)
            weights = weights / (weights.max() + 1E-6)
        return loss * weights


    # Protocol : UpdatableSampler
    @torch.no_grad()
    def update_experiences(self, indices : torch.Tensor, weights : torch.Tensor = None, **kwargs):
        self.priorities[indices] = (weights + 1e-6) ** self.alpha



