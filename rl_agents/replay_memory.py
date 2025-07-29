from rl_agents.sampler import Sampler

from collections import namedtuple, deque
import torch

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))



class BaseReplayMemory(torch.Module):

    def __init__(self, length : int, names : list[str], sizes : list[tuple], dtypes : list[str], sampler : Sampler, device : torch.DeviceObjType):
        self.length = length
        self.tensor_memories = {}
        self.names = names
        self.sizes = [(length, ) + size for size in sizes]
        self.dtypes = dtypes
        self.device = device
        self.sampler = sampler.connect(replay_memory=self)
        self.n = len(names)

        assert self.n == len(sizes) and self.n == len(dtypes), f"names ({len(names)}), sizes ({len(sizes)}), dtypes ({len(dtypes)}) must have the same length"

        for i in range(self.n):
            self.tensor_memories[names[i]] = torch.zeros(size = sizes[i], dtype= dtypes[i], device= device)

        self.i = 0
    def update(self, infos : dict):
        self.sampler.update(infos)

    def size(self):
        return min(self.i, self.length)
            
    def add(self, **kwargs):
        """Save a transition"""
        assert len(kwargs) == self.n, f"Detected some missing elements as kwargs. Names must match : {self.names}. Currently : {self.kwargs.keys()}"

        i = self.i % self.length

        for key, val in kwargs.items():
            try:
                self.tensor_memories[key][i] = val
            except KeyError:
                raise KeyError(f"Key {key} must be in {self.names}")


    def sample(self, batch_size : int):
        batch, weights = self.sampler.sample(batch_size = batch_size)
        elements = {}
        for name in self.names:
            elements[name] = self.tensor_memories[name][batch]
        
        return elements, weights



    def __len__(self):
        return len(self.memory)
    


class ReplayMemory(BaseReplayMemory):
    def __init__(self, length : int, nb_features : int, sampler : Sampler, device : torch.DeviceObjType):
        super().__init__(
            length = length, 
            names =["state", "action", "next_state", "reward"], 
            sizes = [(nb_features, ), (), (nb_features), ()], 
            dtypes = [torch.float32, torch.long, torch.float32, torch.float32], 
            sampler = sampler,
            device = device
        )