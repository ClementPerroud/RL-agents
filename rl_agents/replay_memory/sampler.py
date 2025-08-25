from rl_agents.value_agents.value_agent import AbstractValueAgent
from rl_agents.service import AgentService
from rl_agents.utils.sumtree import SumTree

from abc import ABC, abstractmethod
import numpy as np
import torch
from dataclasses import dataclass
from functools import partial

from typing import TYPE_CHECKING, Optional, Callable

if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent
    from rl_agents.replay_memory.replay_memory import AbstractReplayMemory


class AbstractSampler(AgentService, torch.utils.data.Sampler, ABC):
    def update(self, agent: "AbstractAgent"): ...

    def store(self, agent: "AbstractAgent", **kwargs): ...

    def compute_weights_from_indices(self, indices : torch.Tensor):...

    def update_experiences(self, agent : "AbstractAgent", **kwargs): ...

class RandomSampler(AbstractSampler):
    def __init__(self, replay_memory : "AbstractReplayMemory"):
        super().__init__()
        self.replay_memory = replay_memory

    @torch.no_grad()
    def __len__(self) -> int:
        return len(self.replay_memory)

    @torch.no_grad()
    def __iter__(self):
        while True:
            n = len(self.replay_memory)
            # with replacement is OK for off-policy
            yield from torch.randint(0, n, (16,)).tolist()
        

class PrioritizedReplaySampler(AbstractSampler):

    def __init__(self, replay_memory: "AbstractReplayMemory", alpha=0.65, beta_0=0.5, duration=150_000):
        super().__init__()
        self.replay_memory = replay_memory
        self.alpha = alpha
        self.beta_0 = beta_0
        self.duration = duration
        self.priorities = SumTree(size=self.replay_memory.max_length)
        self.random_sampler = RandomSampler(replay_memory=self.replay_memory)
        self.step = 0

    @torch.no_grad()
    def __len__(self) -> int:
        return len(self.replay_memory)
    
    @torch.no_grad()
    def __iter__(self):
        for _ in range(len(self)):
            if self.step >= self.duration:
                yield from self.random_sampler.__iter__()
            else:
                indices = self.priorities.sample(16)
                indices = torch.tensor(indices).long()
                yield from indices
        
    def compute_weights_from_indices(self, indices : torch.Tensor):
        beta = min(1, self.beta_0 + (1 - self.beta_0) * self.step / self.duration)
        weights : np.ndarray= (len(self) * self.priorities[indices] / (self.priorities.sum() + 1E-8)) ** (-beta)
        weights = weights / (weights.max() + 1E-6)
        return weights


    @torch.no_grad()
    def update_experiences(self, agent : "AbstractAgent", indices : torch.Tensor, td_errors : torch.Tensor = None, **kwargs):
        td_errors = td_errors.abs().cpu().numpy()
        self.priorities[indices.cpu().numpy()] = (td_errors + 1e-6) ** self.alpha

    @torch.no_grad()
    def store(self, agent: "AbstractAgent", **kwargs):
        assert isinstance(
            agent, AbstractValueAgent
        ), "PrioritizedReplaySampler can only be used with QAgents"

        experience = agent.replay_memory.experience_dataclass_generator(
            **{name : torch.as_tensor(value, dtype = agent.replay_memory.dtypes[name]) for name, value in kwargs.items()}
        )
        loss_inputs = agent.q_function.compute_loss_inputs(experience=experience)

        td_errors = agent.q_function.compute_td_errors(loss_inputs).abs().cpu().numpy()
        new_priorities = (td_errors + 1e-6) ** self.alpha 
        self.priorities.add(new_priorities)
        self.step += 1
