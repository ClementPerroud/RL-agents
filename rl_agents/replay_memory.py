from rl_agents.sampler import AbstractSampler, RandomSampler
from rl_agents.service import AgentService

from abc import ABC, abstractmethod
from collections import namedtuple, deque
import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class AbstractReplayMemory(AgentService, ABC):
    @abstractmethod
    def store(self):
        ...


class BaseReplayMemory(
        torch.nn.Module, # Make methode .to(device) available
        AgentService):
    def __init__(self,
            length : int,
            names : list[str],
            sizes : list[tuple],
            dtypes : list[str],
            sampler : AbstractSampler = RandomSampler(),
            device : torch.DeviceObjType = None
        ):
        torch.nn.Module.__init__(self)
        AgentService.__init__(self)

        self.length = length
        self.tensor_memories = {}
        self.names = names
        self.sizes = [(length, ) + size for size in sizes]
        self.dtypes = dtypes
        self.device = device
        self.sampler = sampler.connect(self)
        self.n = len(names)

        assert self.n == len(sizes) and self.n == len(dtypes), f"names ({len(names)}), sizes ({len(sizes)}), dtypes ({len(dtypes)}) must have the same length"

        for i in range(self.n):
            self.tensor_memories[names[i]] = torch.zeros(size = sizes[i], dtype= dtypes[i], device= device)

        self.i = 0

    def size(self):
        return min(self.i, self.length)
            
    def store(self, **kwargs):
        """Save a transition"""
        assert len(kwargs) == self.n, f"Detected some missing elements as kwargs. Names must match : {self.names}. Currently : {self.kwargs.keys()}"

        i = self.i % self.length

        for key, val in kwargs.items():
            try:
                self.tensor_memories[key][i] = val
            except KeyError:
                raise KeyError(f"Key {key} must be in {self.names}")


    def sample(self, batch_size : int):
        batch, weights = self.sampler.sample(batch_size = batch_size, size = self.size())
        elements = {}
        for name in self.names:
            elements[name] = self.tensor_memories[name][batch]
        
        return elements, weights


class ReplayMemory(BaseReplayMemory):
    def __init__(self,
            length : int,
            nb_features : int,
            sampler : AbstractSampler = RandomSampler(),
            device : torch.DeviceObjType = None
        ):
        super().__init__(
            length = length, 
            names =["state", "action", "next_state", "reward"], 
            sizes = [(nb_features, ), (), (nb_features,), ()], 
            dtypes = [torch.float32, torch.long, torch.float32, torch.float32], 
            sampler = sampler,
            device = device
        )