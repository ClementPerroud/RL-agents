from rl_agents.replay_memory.memory import EditableMemory, MemoryField
from abc import ABC, abstractmethod
from collections import deque, namedtuple
import torch
import numpy as np
from gymnasium.spaces import Space, Box
from functools import partial
import warnings
from dataclasses import dataclass, make_dataclass, asdict, fields


@dataclass(kw_only=True, slots=True, frozen=True)
class Experience:
    def __getattr__(self, item):
        # Called only if attribute not found normally
        raise AttributeError(
            f"'The exprience {type(self).__name__}' has no attribute '{item}'. Please add it the replay_memory"
            f"Available attributes: {[field.name for field in fields(self)]}"
        )

@dataclass(kw_only=True, slots=True, frozen=True)
class ExperienceSample:
    indices : torch.Tensor
    def __getattr__(self, item):
        # Called only if attribute not found normally
        raise AttributeError(
            f"'The replay_memory related to {type(self).__name__}' has no attribute '{item}'. Your replay_memory is probably not meant to be use in this context."
            f"Available attributes: {[field.name for field in fields(self)]}"
        )

class BaseReplayMemory(torch.nn.Module, EditableMemory[Experience]):
    def __init__(
        self,
        max_length: int,
        fields: list[MemoryField],
        device: torch.DeviceObjType = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_length = int(max_length)
        self.device = device


        self.fields  = []
        for field in fields:
            if isinstance(field, MemoryField): pass
            if isinstance(field, tuple): field = MemoryField(*field)
            else: raise ValueError(f"Invalid type of field {type(field)}. Must be a tuple or MemoryField.")
            self.fields.append(field)
        # Unpack fields
        self._mandatory_fields = set()
        self.names, self.sizes, self.dtypes, self.default_values = [], {}, {}, {}
        for field in self.fields: self._set_up_field(*field) # Field : (name, shape, dtype, (optional) default_value)
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
            frozen=True
        )
        self.sample_dataclass_generator = make_dataclass(
            f"ExperienceSample{self.__class__.__name__}",
            fields=experience_fields,
            bases=(ExperienceSample,),
            kw_only=True,
            slots=True,
            frozen=True
        )

    def _set_up_field(self, name : str, shape : tuple[int], dtype : None, default_value = None):
        assert name not in self.names, f"Found duplicate fields {name} in {self.names}"
        self.names.append(name)
        self.sizes[name] = (self.max_length,) + shape
        self.dtypes[name] = dtype
        if default_value is not None: self.default_values[name] = default_value
        else: self._mandatory_fields.add(name)
        self._set_tensor(name=name)

    def add_field(self, name : str, shape : tuple[int], dtype : None, default_value = None):
        self._set_up_field(name=name, shape=shape, dtype=dtype, default_value=default_value)
        self._generate_dataclasses()
    
    def remove_field(self, name : str):
        self.names.remove(name)
        self.sizes.pop(name)
        self.dtypes.pop(name)
        self.default_values.pop(name, 0)
        if name in self._mandatory_fields: self._mandatory_fields.remove(name)
        self.__delattr__(name="memory_{name}")
        self._generate_dataclasses()
        

    def reset(self, name = None):
        for name in self.names: 
            buf : torch.Tensor = getattr(self, f"memory_{name}")
            fill_value = self.default_values.get(name, 0)
            buf.fill_(fill_value)
        self.i: int = 0

    def _set_tensor(self, name : str):
        fill_value = self.default_values.get(name, 0)
        self.register_buffer(
            name=f"memory_{name}",
            tensor=torch.full(size = self.sizes[name], fill_value=fill_value, dtype = self.dtypes[name]),
            persistent=True,
        )
    
    def __len__(self): return min(self.i, self.max_length)

    @torch.no_grad()
    def store(self, experience = None, **kwargs):
        """Save a experience"""
        
        if experience is not None: kwargs.update(asdict(experience))

        assert self._mandatory_fields.issubset(kwargs.keys()), (
            f"Missing fields. Expected at least {sorted(self._mandatory_fields)}, got {sorted(kwargs.keys())}"
        )
        
        i = int(self.i % self.max_length)

        nb_env = next(iter(kwargs.values())).size(0)

        # Compute indices
        if i+nb_env <= self.max_length: indices = torch.arange(i,i+nb_env)
        else:
            n_end = self.max_length - i
            n_start = nb_env - n_end 
            indices = torch.cat([torch.arange(i,i+nb_env), torch.arange(0,n_start)])
        indices = indices.long()

        for name, val in kwargs.items():
            try:
                val_tensor = torch.as_tensor(val, dtype = self.dtypes[name])
            except KeyError:
                warnings.warn(f"{name} does not exist as a field in {self.__class__.__name__}. Current fields : {', '.join(self.names)}")
            else:
                if i+nb_env <= self.max_length: 
                    getattr(self, f"memory_{name}")[indices] = val_tensor

        self.i += nb_env
        return indices

    @torch.no_grad()
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
    
    def __getitems__(self, indices): return self.__getitem__(loc = indices)
    
    @torch.no_grad()
    def __get_experience_from_values__(self, **kwargs): return self.experience_dataclass_generator(**{name : torch.as_tensor(value, dtype = self.dtypes[name]) for name, value in kwargs.items()})
    def __get_experience_from_indices__(self, indices): return self.experience_dataclass_generator(**{name : getattr(self, f"memory_{name}")[indices] for name in self.names})

    @torch.no_grad()
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
                MemoryField("state",       self.observation_space.shape,   torch.float32),
                MemoryField("action",      (),                             torch.long),
                MemoryField("next_state",  self.observation_space.shape,   torch.float32),
                MemoryField("reward",      (),                             torch.float32),
                MemoryField("done",        (),                             torch.bool),
                MemoryField("truncated",   (),                             torch.bool),
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
            "for the services that support it (e.g.: DQN, C51DQN ...)"
        )
        assert isinstance(observation_space, Box)
        self.multi_step, self.gamma, self.nb_env = multi_step, gamma, nb_env
        self.buffers = [deque(maxlen=multi_step) for _ in range(nb_env)]

        super().__init__(
            max_length=max_length,
            fields=[
                MemoryField("state",       observation_space.shape,    torch.float32),
                MemoryField("action",      (),                         torch.long),
                MemoryField("next_state",  observation_space.shape,    torch.float32),
                MemoryField("reward",      (),                         torch.float32),
                MemoryField("done",        (),                         torch.bool),
                MemoryField("truncated",   (),                         torch.bool),
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
        r = torch.stack([e.reward for e in buf])  # (L,)
        R = torch.dot(self._gammas[: len(r)], r)  # scalaire  γ^k * r_k

        return dict(
            state=buf[0].state[None, ...],
            action=buf[0].action[None, ...],
            next_state=buf[-1].next_state[None, ...],
            reward=R[None, ...],
            done=buf[-1].done[None, ...],
            truncated=buf[-1].truncated[None, ...],
        )


    # ---- API publique ----------------------------------------------
    @torch.no_grad()
    def store(self, **kwargs):
        """
        `state`, `action`, … : tenseurs dont la 0-ème dim = nb_env.
        """
        indices = []
        # Boucle fine sur les envs ; la plupart du temps nb_env <= 16, négligeable.
        for env_id in range(self.nb_env):
            kwargs_env = {key : val[env_id] for key, val in kwargs.items()}
            experience_env = self.__get_experience_from_values__(**kwargs_env)

            buf = self.buffers[env_id]
            buf.append(experience_env) # We use tensor[env_id:env_id+1] to select the one elem corresponding 

            # fenêtre pleine : pousse une transition n-step
            if len(buf) == self.multi_step:
                indices.append(
                    super().store(**self._aggregate(buf))
                )
                buf.popleft()  # fenêtre glissante

            # fin d'épisode : flush des restes
            if experience_env.done or experience_env.truncated:
                while buf:
                    indices.append(
                        super().store(**self._aggregate(buf))
                    )
                    buf.popleft()

        if len(indices) == 0: return None 
        return torch.cat(indices, dim = 0)


