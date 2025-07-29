from rl_agents.replay_memory import ReplayMemory

from  collections.abc import Callable
from abc import ABC, abstractmethod
import numpy as np

class Sampler(ABC):
    def __init__(self):
        self._replay_memory : ReplayMemory = None

    @abstractmethod
    def sample(self, batch_size : int) -> tuple[np.ndarray, np.ndarray]:
        ...
    
    def update(self, infos : dict):
        ...

    def connect(self, replay_memory : 'ReplayMemory'):
        self._replay_memory = replay_memory
        return self

    @property
    def replay_memory(self) -> ReplayMemory:
        assert self._replay_memory is not None, "Please use .connect(replay_memory) to connect the sampler to the replay memory"
        return self._replay_memory

class ClassicSampler(Sampler):
    def __init__(self):
        pass
    
    def sample(self, batch_size):
        batch = np.random.choice(self.replay_memory.size(), size = batch_size,
                replace=False)
        return batch, 1


from rl_agents.utils.sumtree import SumTree 
class PrioritizedReplaySampler(Sampler):
        
    def __init__(self, length, beta_function : Callable, alpha = 0.65):
        self.length = length
        self.alpha = alpha
        self.beta_function = beta_function
        self.priorities = SumTree(size= self.length)
        self.last_batch = None

    def sample(self, batch_size : int):
        beta = self.beta_function()
        if beta >= 1: 
            self.replay_memory.sampler = ClassicSampler()

        batch = self.priorities.sample(batch_size)
        weights = (self.replay_memory.size() * self.priorities[batch]/self.priorities.sum() ) ** (- beta)
        weights = weights / np.amax(weights)

        self.last_batch = batch
        return batch, weights

    def update(self, infos : dict):
        td_errors = infos["td_errors"]
        self.priorities[self.batch] = (np.abs(td_errors)+ 1)**self.alpha