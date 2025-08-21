from rl_agents.q_agents.value_agent import AbstractValueAgent
from rl_agents.service import AgentService
from rl_agents.utils.sumtree import SumTree

from abc import ABC, abstractmethod
import numpy as np
import torch

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent


class AbstractSampler(AgentService, ABC):
    @abstractmethod
    def sample(self, agent : 'AbstractAgent', batch_size: int, size: int) -> tuple[np.ndarray, np.ndarray]: ...

    def update(self, agent: "AbstractAgent"): ...

    def train_callback(self, **kwargs): ...

    def store(self, agent: "AbstractAgent", **kwargs): ...


class RandomSampler(AbstractSampler):
    def __init__(self):
        pass

    def sample(self, agent : "AbstractAgent", batch_size: int, size: int):
        batch = torch.from_numpy(np.random.choice(size, size=batch_size, replace=False))
        return batch, None


class PrioritizedReplaySampler(AbstractSampler):

    def __init__(self, max_length: int, alpha=0.65, beta_0=0.5, duration=150_000):
        self.max_length = int(max_length)
        self.alpha = alpha
        self.beta_0 = beta_0
        self.duration = duration
        self.priorities = SumTree(size=self.max_length)
        self.last_batch = None
        self.random_sampler = RandomSampler()

    def sample(self, agent : "AbstractAgent", batch_size: int, size: int):

        # if self.step < batch_size:
        #     raise ValueError("Cannot sample batch_size")
        
        if agent.step >= self.duration:
            return self.random_sampler.sample(batch_size=batch_size, size=size)

        beta = min(1, self.beta_0 + (1 - self.beta_0) * agent.step / self.duration)

        batch = self.priorities.sample(batch_size)
        weights : np.ndarray= (size * self.priorities[batch] / (self.priorities.sum() + 1E-8)) ** (-beta)
        weights = weights / (weights.max() + 1E-6)

        return torch.tensor(batch).long(), torch.from_numpy(weights)

    def train_callback(self, batch : torch.Tensor, td_errors: torch.Tensor, **kwargs):
        td_errors = td_errors.abs().cpu().numpy()
        self.priorities[batch.cpu().numpy()] = (td_errors + 1e-6) ** self.alpha

    @torch.no_grad()
    def store(self, agent: "AbstractAgent", **kwargs):
        assert isinstance(
            agent, AbstractValueAgent
        ), "PrioritizedReplaySampler can only be used with QAgents"
        state = torch.as_tensor(kwargs["state"])
        action = torch.as_tensor(kwargs["action"])
        reward = torch.as_tensor(kwargs["reward"])
        next_state = torch.as_tensor(kwargs["next_state"])
        done = torch.as_tensor(kwargs["done"])

        y_true, y_pred = agent.q_function.compute_loss_inputs(
            state=state, action=action, reward=reward, next_state=next_state, done=done
        )
        td_errors = agent.q_function.compute_td_errors(y_true=y_true, y_pred=y_pred).abs().cpu().numpy()
        new_priorities = (td_errors + 1e-6) ** self.alpha 
        self.priorities.add(new_priorities)
