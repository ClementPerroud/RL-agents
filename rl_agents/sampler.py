from rl_agents.q_agent import AbstractQAgent 
from rl_agents.service import AgentService
from rl_agents.utils.sumtree import SumTree

from abc import ABC, abstractmethod
import numpy as np
import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.replay_memory import ReplayMemory
    from rl_agents.agent import AbstractAgent
    

class AbstractSampler(AgentService, ABC):
    @abstractmethod
    def sample(self, batch_size : int, size : int) -> tuple[np.ndarray, np.ndarray]:
        ...
    
    def update(self, agent : 'AbstractAgent'):
        ...

    def train_callback(self, agent : 'AbstractAgent', infos : dict):
        ...
    
    def store(self, agent : 'AbstractAgent', **kwargs):
        ...

class RandomSampler(AbstractSampler):
    def __init__(self):
        pass
    
    def sample(self, batch_size : int, size : int):
        batch = torch.from_numpy(np.random.choice(size, size = batch_size,replace=False))
        return batch, 1



class PrioritizedReplaySampler(AbstractSampler):
        
    def __init__(self, length : int, alpha = 0.65, beta_0 = 0.5, duration = 150_000):
        self.length = int(length)
        self.alpha = alpha
        self.beta_0 = 0.5
        self.duration = 150_000
        self.priorities = SumTree(size= self.length)
        self.last_batch = None
        self.step = 0
        self.random_sampler = RandomSampler()
    
    def sample(self, batch_size : int, size : int):
        if self.step >= self.duration: 
            return self.random_sampler.sample(batch_size=batch_size)
        
        beta = min(1, self.beta_0 + (1 - self.beta_0)*self.step/150_000)

        batch = self.priorities.sample(batch_size)
        weights = (size * self.priorities[batch]/self.priorities.sum() ) ** (- beta)
        weights = weights / np.amax(weights)

        self.last_batch = batch
        return torch.Tensor(batch).long(), torch.from_numpy(weights)

    def train_callback(self, agent : 'AbstractAgent', infos : dict):
        td_errors = (infos['y_true'] - infos['y_pred']).abs().numpy()
        self.priorities[self.last_batch] = (td_errors+ 1E6)**self.alpha

    @torch.no_grad()
    def store(self, agent : 'AbstractAgent', **kwargs):
        assert isinstance(agent, AbstractQAgent), "PrioritizedReplaySampler can only be used with QAgents"
        
        state = torch.Tensor([kwargs["state"]])
        action = torch.Tensor([kwargs["action"]])
        reward = torch.Tensor([kwargs["reward"]])
        next_state = torch.Tensor([kwargs["next_state"]])
        done = torch.BoolTensor([kwargs["done"]])

        y_true, y_pred = agent.compute_td_errors(state= state, action = action, reward = reward, next_state = next_state, done = done)
        self.priorities.add(
            float((y_true - y_pred)[0].abs().numpy())
        )