from rl_agents.replay_memory.sampler import AbstractSampler, RandomSampler
from rl_agents.service import AgentService


from abc import ABC, abstractmethod
from collections import deque
import torch
import numpy as np
from gymnasium.spaces import Space, Box

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent


class AbstractReplayMemory(AgentService, ABC):
    @abstractmethod
    def store(self, agent: "AbstractAgent", **kwargs): ...

    @abstractmethod
    def sample(self, batch_size: int): ...

    def train_callback(self, agent: "AbstractAgent", infos: torch.Tensor): ...


class BaseReplayMemory(
    torch.nn.Module, AbstractReplayMemory  # Make methode .to(device) available
):
    def __init__(
        self,
        length: int,
        names: list[str],
        sizes: list[tuple],
        dtypes: list[torch.dtype],
        sampler: AbstractSampler,
        device: torch.DeviceObjType = None,
    ):
        torch.nn.Module.__init__(self)
        AgentService.__init__(self)
        self.length = int(length)
        self.tensor_memories = {}
        self.names = names
        self.sizes = [(self.length,) + size for size in sizes]
        self.dtypes = dtypes
        self.device = device
        self.sampler = sampler.connect(self)
        self.n = len(names)

        assert self.n == len(sizes) and self.n == len(
            dtypes
        ), f"names ({len(names)}), sizes ({len(sizes)}), dtypes ({len(dtypes)}) must have the same length"

        for i in range(self.n):
            self.tensor_memories[names[i]] = torch.zeros(
                size=self.sizes[i], dtype=dtypes[i], device=device
            )

        self.i: int = 0

    def size(self):
        return min(self.i, self.length)

    @torch.no_grad()
    def store(self, agent: "AbstractAgent", **kwargs):
        """Save a transition"""
        assert (
            len(kwargs) == self.n
        ), f"Detected some missing elements as kwargs. Names must match : {self.names}. Currently : {kwargs.keys()}"

        i = int(self.i % self.length)

        for name, val in kwargs.items():
            try:
                # print(name, i, val, type(val), self.tensor_memories[name][i], self.tensor_memories[name].shape)
                try:
                    index_name = self.names.index(name)
                except ValueError as e:
                    raise ValueError(f"{name} does not exist as a field in {self.__class__.__name__}. Current fields : {', '.join(self.names)}")
                
                self.tensor_memories[name][i] = torch.as_tensor(val, dtype = self.dtypes[index_name])
            except KeyError:
                raise KeyError(f"{name} does not exist as a field in {self.__class__.__name__}. Current fields : {', '.join(self.names)}")
        
        self.i += 1
        self.sampler.store(agent=agent, **kwargs)

    @torch.no_grad()
    def sample(self, batch_size: int):
        if self.size() < batch_size:
            return

        batch, weights = self.sampler.sample(batch_size=batch_size, size=self.size())
        elements = {}
        for name in self.names:
            elements[name] = self.tensor_memories[name][batch]
        elements["weight"] = weights
        return elements

    @torch.no_grad()
    def train_callback(self, agent: "AbstractAgent", infos: dict):
        self.sampler.train_callback(agent=agent, infos=infos)


class ReplayMemory(BaseReplayMemory):
    def __init__(
        self,
        length: int,
        observation_space: Space,
        sampler: AbstractSampler = RandomSampler(),
        device: torch.DeviceObjType = None,
    ):
        assert isinstance(
            observation_space, Box
        ), "ReplayMemory only supports gymnasium.spaces.Box as observation_space"
        self.observation_space = observation_space

        super().__init__(
            length=length,
            names=["state", "action", "next_state", "reward", "done"],
            sizes=[
                self.observation_space.shape,
                (),
                self.observation_space.shape,
                (),
                (),
            ],
            dtypes=[
                torch.float32,
                torch.long,
                torch.float32,
                torch.float32,
                torch.bool,
            ],
            sampler=sampler,
            device=device,
        )


class MultiStepReplayMemory(BaseReplayMemory):
    """
    n-step replay pour plusieurs environnements parallèles.
    Les arguments d'entrée de .store doivent avoir shape (num_envs, …).
    """

    def __init__(
        self,
        length: int,
        observation_space: Box,
        num_envs: int,
        multi_step: int = 3,
        gamma: float = 0.99,
        sampler: AbstractSampler = RandomSampler(),
        device=None,
    ):

        assert isinstance(observation_space, Box)
        self.multi_step, self.gamma, self.num_envs = multi_step, gamma, num_envs
        self.buffers = [deque(maxlen=multi_step) for _ in range(num_envs)]

        super().__init__(
            length=length,
            names=["state", "action", "next_state", "reward", "done"],
            sizes=[observation_space.shape, (), observation_space.shape, (), ()],
            dtypes=[
                torch.float32,
                torch.long,
                torch.float32,
                torch.float32,
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
            state=buf[0]["state"],
            action=buf[0]["action"],
            next_state=buf[-1]["next_state"],
            reward=R,
            done=buf[-1]["done"],
        )

    # ---- API publique ----------------------------------------------
    @torch.no_grad()
    def store(self,  agent: "AbstractAgent", *, state, action, next_state, reward, done):
        """
        `state`, `action`, … : tenseurs dont la 0-ème dim = num_envs.
        """
        # Boucle fine sur les envs ; la plupart du temps num_envs <= 16, négligeable.
        for env_id in range(self.num_envs):
            buf = self.buffers[env_id]
            buf.append(
                {
                    "state": torch.as_tensor(state[env_id]),
                    "action": torch.as_tensor(action[env_id]),
                    "next_state": torch.as_tensor(next_state[env_id]),
                    "reward": torch.as_tensor(reward[env_id]),
                    "done": torch.as_tensor(bool(done[env_id])),
                }
            )

            # fenêtre pleine : pousse une transition n-step
            if len(buf) == self.multi_step:
                super().store(agent=agent, **self._aggregate(buf))
                buf.popleft()  # fenêtre glissante

            # fin d'épisode : flush des restes
            if done[env_id]:
                while buf:
                    super().store(agent=agent, **self._aggregate(buf))
                    buf.popleft()
