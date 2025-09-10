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

if TYPE_CHECKING:
    from rl_agents.agent import BaseAgent
    from rl_agents.memory.replay_memory import BaseExperienceMemory


@runtime_checkable
class Sampler(Protocol):
    replay_memory : "BaseExperienceMemory"
    def sample(self, batch_size : int) -> torch.Tensor: ...


@runtime_checkable
class UpdatableSampler(Protocol):
    def store(self, **kwargs): ...
    def update_experiences(self, **kwargs): ...


@runtime_checkable
class WeightableSampler(Protocol):
    def apply_weights(self, loss : torch.Tensor, indices : torch.Tensor) -> torch.Tensor:...


class RandomSampler(AgentService):
    def __init__(self, replay_memory : "BaseExperienceMemory", **kwargs):
        super().__init__(**kwargs)
        self.replay_memory = replay_memory

    @torch.no_grad()
    def sample(self, batch_size : int): 
        return torch.randint(0, len(self.replay_memory), (batch_size,))

    

class LastSampler(AgentService):
    def __init__(self, replay_memory : "BaseExperienceMemory", **kwargs):
        super().__init__(**kwargs)
        self.replay_memory = replay_memory
    
    @torch.no_grad()
    def sample(self, batch_size : int): 
        n = len(self.replay_memory)
        return torch.arange(start = n - batch_size, end = n)


class PrioritizedReplaySampler(AgentService):
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
        self.step = 0
    
    @torch.no_grad()
    def sample(self, batch_size : int):
        if self.step>=self.duration: return self.random_sampler.sample(batch_size=batch_size)

        indices = self.priorities.sample(batch_size=batch_size)
        indices = torch.tensor(indices).long()
        return indices
    
    # Protocol : WeightableSampler
    def apply_weights(self, loss : torch.Tensor, indices : torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            beta = min(1, self.beta_0 + (1 - self.beta_0) * self.step / self.duration)
            weights : np.ndarray= (len(self.replay_memory) * self.priorities[indices] / (self.priorities.sum() + 1E-8)) ** (-beta)
            weights = weights / (weights.max() + 1E-6)
        return loss * weights


    # Protocol : UpdatableSampler
    @torch.no_grad()
    def update_experiences(self, indices : torch.Tensor, td_errors : torch.Tensor = None, **kwargs):
        td_errors = td_errors.abs()
        self.priorities[indices] = (td_errors + 1e-6) ** self.alpha

    @torch.no_grad()
    def store(self, experience = None, **kwargs):
        if experience is None:
            experience = self.replay_memory.experience_dataclass_generator(
                **{name : torch.as_tensor(value, dtype = self.replay_memory.dtypes[name]) for name, value in kwargs.items()}
            )

        loss_input = self.service.compute_loss_input(experience=experience)
        loss_target = self.service.compute_loss_target(experience=experience)
        td_errors = self.service.compute_td_errors(loss_input=loss_input, loss_target=loss_target).abs().cpu().numpy()

        new_priorities = (td_errors + 1e-6) ** self.alpha 
        self.priorities.add(new_priorities)
        self.step += new_priorities.size # = nb of experience stored