from rl_agents.replay_memory.sampler import AbstractSampler, RandomSampler
from rl_agents.service import AgentService

from abc import ABC, abstractmethod
from collections import deque
import torch
import numpy as np
from gymnasium.spaces import Space, Box
from functools import partial
import warnings
from dataclasses import dataclass, make_dataclass, asdict, fields

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent

import gymnasium as gym
gym.Env()

torch.utils.data.TensorDataset
class AbstractReplayMemory(torch.utils.data.Dataset, AgentService, ABC):
    @abstractmethod
    def store(self, agent: "AbstractAgent", **kwargs): ...

    @abstractmethod
    def reset(self): ...

    _max_length = None
    @property
    def max_length(self):
        if self._max_length is None: NotImplementedError(f"Please implement attribute max_length in {self.__class__.__name__}")
        return self._max_length

    @max_length.setter
    def max_length(self, val): self._max_length = val


@dataclass(kw_only=True, slots=True)
class Experience:
    def __getattr__(self, item):
        # Called only if attribute not found normally
        raise AttributeError(
            f"'The exprience {type(self).__name__}' has no attribute '{item}'. Please add it the replay_memory"
            f"Available attributes: {list(vars(self).keys())}"
        )

@dataclass(kw_only=True, slots=True)
class ExperienceSample:
    indices : torch.Tensor
    def __getattr__(self, item):
        # Called only if attribute not found normally
        raise AttributeError(
            f"'The replay_memory related to {type(self).__name__}' has no attribute '{item}'. Your replay_memory is probably not meant to be use in this context."
            f"Available attributes: {[field.name for field in fields(self)]}"
        )

class BaseReplayMemory(AbstractReplayMemory):
    def __init__(
        self,
        max_length: int,
        fields: list[tuple[str, tuple, torch.dtype]],
        device: torch.DeviceObjType = None,
    ):
        torch.nn.Module.__init__(self)
        AgentService.__init__(self)
        self.max_length = int(max_length)
        self.device = device

        # Unpack fields
        self.fields  = fields
        self._mandatory_fields = set()
        self.names, self.sizes, self.dtypes, self.default_values = [], {}, {}, {}
        for field in fields: self._set_up_field(*field) # Field : (name, shape, dtype, (optional) default_value)
        self._generate_dataclasses()
        self.i = 0


    def _generate_dataclasses(self):
        # Create the dataclasses dynamically
        experience_fields = [(name, torch.Tensor) for name in self.names]

        self.experience_dataclass_generator = make_dataclass(
            f"Experience{self.__class__.__name__}",
            fields=experience_fields,
            bases=(Experience,),
            kw_only=True,
            slots=True,
        )
        self.sample_dataclass_generator = make_dataclass(
            f"ExperienceSample{self.__class__.__name__}",
            fields=experience_fields,
            bases=(ExperienceSample,),
            kw_only=True,
            slots=True,
        )

    def _set_up_field(self, name : str, shape : tuple[int], dtype : None, default_value = None):
        assert name not in self.names, f"Found duplicate fields {name} in {self.names}"
        self.names.append(name)
        self.sizes[name] = (self.max_length,) + shape
        self.dtypes[name] = dtype
        if default_value is not None: self.default_values[name] = default_value
        else: self._mandatory_fields.add(name)
        self._reset_tensor(name=name)

    def add_field(self, name : str, shape : tuple[int], dtype : None, default_value = None):
        self._set_up_field(name=name, shape=shape, dtype=dtype, default_value=default_value)
        self._generate_dataclasses()


    def reset(self, name = None):
        for name in self.names: 
            self._reset_tensor(name = name)
        self.i: int = 0

    def _reset_tensor(self, name : str):
        fill_value = 0
        if name in self.default_values:fill_value=self.default_values[name]

        self.register_buffer(
            name=f"memory_{name}",
            tensor=torch.full(size = self.sizes[name], fill_value=fill_value, dtype = self.dtypes[name]),
            persistent=True,
        )
    def __len__(self):
        return min(self.i, self.max_length)

    @torch.no_grad()
    def store(self, agent: "AbstractAgent", experience = None, **kwargs):
        """Save a experience"""
        
        if experience is not None: kwargs.update(asdict(experience))

        assert self._mandatory_fields.issubset(kwargs.keys()), (
            f"Missing fields. Expected at least {sorted(self._mandatory_fields)}, got {sorted(kwargs.keys())}"
        )
        
        i = int(self.i % self.max_length)

        nb_env = next(iter(kwargs.values())).size(0)
        for name, val in kwargs.items():
            try:
                val_tensor = torch.as_tensor(val, dtype = self.dtypes[name])
            except KeyError:
                warnings.warn(f"{name} does not exist as a field in {self.__class__.__name__}. Current fields : {', '.join(self.names)}")
            else:
                if i+nb_env <= self.max_length: getattr(self, f"memory_{name}")[i:i+nb_env] = val_tensor
                else:
                    n_end = self.max_length - i
                    n_start = nb_env - n_end
                    getattr(self, f"memory_{name}")[i:i+n_end] = val_tensor[:n_end]
                    getattr(self, f"memory_{name}")[0:n_start] = val_tensor[n_end:n_end + n_start]

        
        self.i += nb_env
        # self.sampler.store(agent=agent, replay_memory = self, **kwargs)

    # @torch.no_grad()
    # def sample(self, agent : "AbstractAgent", batch_size: int) -> ExperienceSample:
    #     if self.__len__() < batch_size:
    #         return

    #     batch_sample = self.sampler.sample(agent=agent, batch_size=batch_size, size=self.__len__())
    #     return self[batch_sample]


    def __getitem__(self, loc) -> ExperienceSample:
        index : int | np.ndarray
        name :str
        if isinstance(loc, tuple):
            name, index= loc
            return getattr(self, f"memory_{name}")[index]
        elif isinstance(loc, str):
            name = loc
            return getattr(self, f"memory_{name}")[:self.__len__()]
        else:
            indices = torch.as_tensor(loc)
            return self.sample_dataclass_generator(
                indices=indices,
                **asdict(self.__get_experience_from_indices__(indices=indices))
            )
    def __getitems__(self, indices):
        return self.__getitem__(loc = indices)
    
    def __get_experience_from_values__(self, **kwargs): return self.experience_dataclass_generator(**{name : torch.as_tensor(value, dtype = self.dtypes[name]) for name, value in kwargs.items()})
    def __get_experience_from_indices__(self, indices): return self.experience_dataclass_generator(**{name : getattr(self, f"memory_{name}")[indices] for name in self.names})

    def __setitem__(self, loc, val):
        indices : int | np.ndarray
        name :str
        if isinstance(loc, tuple):
            name, indices= loc
            getattr(self, f"memory_{name}")[indices] = val
        elif isinstance(loc, str):
            name = loc
            getattr(self, f"memory_{name}")[:self.__len__()] = val


class ReplayMemory(BaseReplayMemory):
    def __init__(
        self,
        max_length: int,
        observation_space: Space,
        device: torch.DeviceObjType = None,
    ):
        assert isinstance(
            observation_space, Box
        ), "ReplayMemory only supports gymnasium.spaces.Box as observation_space"
        self.observation_space = observation_space

        super().__init__(
            max_length=max_length,
            fields=[
                ("state",       self.observation_space.shape,   torch.float32),
                ("action",      (),                             torch.long),
                ("next_state",  self.observation_space.shape,   torch.float32),
                ("reward",      (),                             torch.float32),
                ("done",        (),                             torch.bool),
                ("truncated",   (),                             torch.bool),
            ],
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
        device: torch.DeviceObjType = None,
    ):
        warnings.warn(
            f"When using {self.__class__.__name__}, please provide the multi_step parameter "
            "for the services that support it (e.g.: DQNFunction, DistributionalDQNFunction ...)"
        )
        assert isinstance(observation_space, Box)
        self.multi_step, self.gamma, self.nb_env = multi_step, gamma, nb_env
        self.buffers = [deque(maxlen=multi_step) for _ in range(nb_env)]

        super().__init__(
            max_length=max_length,
            fields=[
                ("state",       observation_space.shape,    torch.float32),
                ("action",      (),                         torch.long),
                ("next_state",  observation_space.shape,    torch.float32),
                ("reward",      (),                         torch.float32),
                ("done",        (),                         torch.bool),
                ("truncated",   (),                         torch.bool),
            ],
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


