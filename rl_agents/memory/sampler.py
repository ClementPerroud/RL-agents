from rl_agents.value_functions.value import Trainable
from rl_agents.value_agents.value_agent import AbstractValueAgent
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
        return torch.randint(0, len(self.replay_memory), (batch_size,))

    

class LastSampler(AgentService, Sampler):
    def __init__(self, replay_memory : "BaseExperienceMemory", **kwargs):
        super().__init__(**kwargs)
        self.replay_memory = replay_memory
    
    @torch.no_grad()
    def sample(self, batch_size : int): 
        n = len(self.replay_memory)
        return torch.arange(start = n - batch_size, end = n)


class PrioritizedReplaySampler(AgentService, UpdatableSampler, WeightableSampler, Sampler):
    def __init__(self, replay_memory: "BaseExperienceMemory", service : Trainable, alpha=0.65, beta_0=0.5, duration=150_000):
        assert isinstance(service, Trainable), "Provided service must implement Trainable."
        super().__init__()
        self.replay_memory = replay_memory
        self.service = service
        self.alpha = alpha
        self.beta_0 = beta_0
        self.duration = duration
        self.priorities = SumTree(size=self.replay_memory.max_length)
        self.random_sampler = RandomSampler(replay_memory=self.replay_memory)
        self.nb_step = 0
    
    @torch.no_grad()
    def sample(self, batch_size : int):
        if self.nb_step>=self.duration: return self.random_sampler.sample(batch_size=batch_size)

        indices = self.priorities.sample(batch_size=batch_size)
        indices = torch.tensor(indices).long()
        return indices
    
    # Protocol : WeightableSampler
    def apply_weights(self, loss : torch.Tensor, indices : torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            beta = min(1, self.beta_0 + (1 - self.beta_0) * self.nb_step / self.duration)
            weights : np.ndarray= (len(self.replay_memory) * self.priorities[indices] / (self.priorities.sum() + 1E-8)) ** (-beta)
            weights = weights / (weights.max() + 1E-6)
        return loss * weights


    # Protocol : UpdatableSampler
    @torch.no_grad()
    def update_experiences(self, indices : torch.Tensor, td_errors : torch.Tensor = None, **kwargs):
        td_errors = td_errors.abs()
        self.priorities[indices] = (td_errors + 1e-6) ** self.alpha

    @torch.no_grad()
    def update(self, agent : AgentService, **kwargs):
        experience = self.replay_memory.experience_dataclass_generator(
            **{name : torch.as_tensor(value, dtype = self.replay_memory.fields_by_name[name].dtype) for name, value in kwargs.items()}
        )

        loss_input = self.service.compute_loss_input(experience=experience)
        loss_target = self.service.compute_loss_target(experience=experience)
        td_errors = self.service.compute_td_errors(loss_input=loss_input, loss_target=loss_target).abs().cpu().numpy()

        new_priorities = (td_errors + 1e-6) ** self.alpha 
        self.priorities.add(new_priorities)
        self.nb_step += new_priorities.size # = nb of experience stored