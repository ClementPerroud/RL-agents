from rl_agents.replay_memory.sampler import AbstractSampler, RandomSampler
from rl_agents.service import AgentService


from abc import ABC, abstractmethod
from collections import deque
import torch
import numpy as np
from gymnasium.spaces import Space, Box
from functools import partial
import warnings

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent


class AbstractReplayMemory(torch.utils.data.Dataset, AgentService, ABC):
    @abstractmethod
    def store(self, agent: "AbstractAgent", **kwargs): ...

    @abstractmethod
    def sample(self, batch_size: int): ...

    @abstractmethod
    def reset(self): ...

    def train_callback(self, agent: "AbstractAgent", infos: torch.Tensor): ...


class BaseReplayMemory(AbstractReplayMemory):
    def __init__(
        self,
        max_length: int,
        names: list[str],
        sizes: list[tuple],
        dtypes: list[torch.dtype],
        sampler: AbstractSampler,
        device: torch.DeviceObjType = None,
    ):
        torch.nn.Module.__init__(self)
        AgentService.__init__(self)
        self.max_length = int(max_length)
        self.names = names
        self.sizes = {name:(self.max_length,) + size for name, size in zip(self.names, sizes)}
        self.dtypes = {name:dtype for name, dtype in zip(self.names, dtypes)}
        self.device = device
        self.sampler = sampler
        self.n = len(names)

        assert self.n == len(sizes) and self.n == len(
            dtypes
        ), f"names ({len(names)}), sizes ({len(sizes)}), dtypes ({len(dtypes)}) must have the same max_length"
        self.reset()

    def reset(self):
        self.tensor_memories = {}
        for name in self.names:
            self.tensor_memories[name] = torch.zeros(
                size=self.sizes[name], dtype=self.dtypes[name], device=self.device
            )

        self.i: int = 0

    def __len__(self):
        return min(self.i, self.max_length)

    @torch.no_grad()
    def store(self, agent: "AbstractAgent", **kwargs):
        """Save a transition"""
        assert (
            len(kwargs) == self.n
        ), f"Detected some missing elements as kwargs. Names must match : {self.names}. Currently : {kwargs.keys()}"

        i = int(self.i % self.max_length)

        nb_env = next(iter(kwargs.values())).size(0)
        for name, val in kwargs.items():
            try:
                val_tensor = torch.as_tensor(val, dtype = self.dtypes[name])
                if i+nb_env <= self.max_length: self.tensor_memories[name][i:i+nb_env] = val_tensor
                else:
                    n_end = self.max_length - i
                    n_start = nb_env - n_end
                    self.tensor_memories[name][i:i+n_end] = val_tensor[:n_end]
                    self.tensor_memories[name][0:n_start] = val_tensor[n_end:n_end + n_start]

            except KeyError:
                raise KeyError(f"{name} does not exist as a field in {self.__class__.__name__}. Current fields : {', '.join(self.names)}")
        
        self.i += nb_env
        self.sampler.store(agent=agent, **kwargs)

    @torch.no_grad()
    def sample(self, agent : "AbstractAgent", batch_size: int):
        if self.__len__() < batch_size:
            return

        batch, weights = self.sampler.sample(agent=agent, batch_size=batch_size, size=self.__len__())
        elements = self[batch]
        elements["weight"] = weights
        elements["replay_memory_callbacks"] = [partial(self.sampler.train_callback, batch = batch)]
        return elements

    
    def __getitem__(self, loc):
        index : int | np.ndarray
        name :str
        if isinstance(loc, tuple):
            name, index= loc
            return self.tensor_memories[name][index]
        elif isinstance(loc, str):
            name = loc
            return self.tensor_memories[name][:self.__len__()]
        else:
            index = loc
            elements = {}
            for name in self.names:
                elements[name] = self.tensor_memories[name][index]
            return elements

    def __setitem__(self, loc, val):
        index : int | np.ndarray
        name :str
        if isinstance(loc, tuple):
            name, index= loc
            self.tensor_memories[name][index] = val
        elif isinstance(loc, str):
            name = loc
            self.tensor_memories[name][:self.__len__()] = val



    @torch.no_grad()
    def train_callback(self, agent: "AbstractAgent", infos: dict):
        self.sampler.train_callback(agent=agent, infos=infos)


class ReplayMemory(BaseReplayMemory):
    def __init__(
        self,
        max_length: int,
        observation_space: Space,
        sampler: AbstractSampler = RandomSampler(),
        device: torch.DeviceObjType = None,
    ):
        assert isinstance(
            observation_space, Box
        ), "ReplayMemory only supports gymnasium.spaces.Box as observation_space"
        self.observation_space = observation_space

        super().__init__(
            max_length=max_length,
            names=["state", "action", "next_state", "reward", "done", "truncated"],
            sizes=[
                self.observation_space.shape,
                (),
                self.observation_space.shape,
                (),
                (),
                (),
            ],
            dtypes=[
                torch.float32,
                torch.long,
                torch.float32,
                torch.float32,
                torch.bool,
                torch.bool,
            ],
            sampler=sampler,
            device=device,
        )


class MultiStepReplayMemory(BaseReplayMemory):
    """
    n-step replay pour plusieurs environnements parallèles.
    Les arguments d'entrée de .store doivent avoir shape (nb_env, …).
    """

    def __init__(
        self,
        max_length: int,
        observation_space: Box,
        nb_env: int,
        gamma: float,
        multi_step: int,
        sampler: AbstractSampler,
        device=None,
    ):
        warnings.warn(f"When using {self.__class__.__name__}, please provide the multi_step parameter for the services that support it (e.g : DQNFunction, DistributionalDQNFunction ...)")
        assert isinstance(observation_space, Box)
        self.multi_step, self.gamma, self.nb_env = multi_step, gamma, nb_env
        self.buffers = [deque(maxlen=multi_step) for _ in range(nb_env)]

        super().__init__(
            max_length=max_length,
            names=["state", "action", "next_state", "reward", "done", "truncated"],
            sizes=[observation_space.shape, (), observation_space.shape, (), (), ()],
            dtypes=[
                torch.float32,
                torch.long,
                torch.float32,
                torch.float32,
                torch.bool,
                torch.bool,
            ],
            sampler=sampler,
            device=device,
        )

        # pré-calc γ^k pour l’agrégation vectorielle
        self._gammas = gamma ** torch.arange(
            multi_step, device=device, dtype=torch.float32
        )

    def _aggregate(self, buf: deque):
        """
        Convertit le contenu d'un deque en transition n-step.
        -> retourne un dict prêt pour super().store(**transition)
        """
        # Empile récompenses pour un produit scalaire vectoriel
        r = torch.stack([t["reward"] for t in buf])  # (L,)
        R = torch.dot(self._gammas[: len(r)], r)  # scalaire  γ^k * r_k

        return dict(
            state=buf[0]["state"][None, ...],
            action=buf[0]["action"][None, ...],
            next_state=buf[-1]["next_state"][None, ...],
            reward=R[None, ...],
            done=buf[-1]["done"][None, ...],
            truncated=buf[-1]["truncated"][None, ...],
        )

    # ---- API publique ----------------------------------------------
    @torch.no_grad()
    def store(self,  agent: "AbstractAgent", *, state, action, next_state, reward, done, truncated):
        """
        `state`, `action`, … : tenseurs dont la 0-ème dim = nb_env.
        """
        state =torch.as_tensor(state, dtype=self.dtypes["state"])
        action = torch.as_tensor(action, dtype=self.dtypes["action"])
        next_state = torch.as_tensor(next_state, dtype=self.dtypes["next_state"])
        reward = torch.as_tensor(reward, dtype=self.dtypes["reward"])
        done = torch.as_tensor(done, dtype=self.dtypes["done"])
        truncated = torch.as_tensor(truncated, dtype=self.dtypes["truncated"])
        
        # Boucle fine sur les envs ; la plupart du temps nb_env <= 16, négligeable.
        for env_id in range(self.nb_env):
            buf = self.buffers[env_id]
            buf.append(
                {
                    "state":state[env_id],
                    "action":action[env_id],
                    "next_state":next_state[env_id],
                    "reward":reward[env_id],
                    "done":done[env_id],
                    "truncated":truncated[env_id]
                } # We use tensor[env_id:env_id+1] to select the one elem corresponding 
            )

            # fenêtre pleine : pousse une transition n-step
            if len(buf) == self.multi_step:
                super().store(agent=agent, **self._aggregate(buf))
                buf.popleft()  # fenêtre glissante

            # fin d'épisode : flush des restes
            if done[env_id] or truncated[env_id]:
                while buf:
                    super().store(agent=agent, **self._aggregate(buf))
                    buf.popleft()


