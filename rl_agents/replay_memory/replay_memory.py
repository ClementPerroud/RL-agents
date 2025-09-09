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


        self.fields : list[MemoryField] = []
        self.fields_by_name : dict[str, MemoryField] = {}
        self._mandatory_fields_name = set()
        for field in fields:
            if isinstance(field, MemoryField): pass
            if isinstance(field, tuple): field = MemoryField(*field)
            else: raise ValueError(f"Invalid type of field {type(field)}. Must be a tuple or MemoryField.")
            self._set_up_field(field) 

        self._generate_dataclasses()
        self.i = 0

    @property
    def names(self) -> list[str]:
        return list(self.fields_by_name.keys())

    def _generate_dataclasses(self):
        # Create the dataclasses dynamically
        experience_fields = [(field.name, field.codec.tensor_class) for field in self.fields]

        self.experience_dataclass_generator = make_dataclass(
            f"Experience{self.__class__.__name__}", fields=experience_fields,
            bases=(Experience,), kw_only=True, slots=True, frozen=True
        )
        self.sample_dataclass_generator = make_dataclass(
            f"ExperienceSample{self.__class__.__name__}", fields=experience_fields,
            bases=(ExperienceSample,), kw_only=True, slots=True, frozen=True
        )


    def add_field(self, field : MemoryField):
        self._set_up_field(field)
        self._generate_dataclasses()

    def _set_up_field(self, field : MemoryField):
        assert field.name not in self.fields_by_name, f"Found duplicate fields {field.name} in {self.names}"
        self.fields.append(field)
        self.fields_by_name[field.name] = field
        if field.default is None: 
            self._mandatory_fields_name.add(field.name)
    
        buf = field.codec.allocate(size= (self.max_length, ) + field.shape, field=field, device=self.device)
        self.register_buffer(name=f"memory_{field.name}", tensor=buf, persistent=True)

    def remove_field(self, name : str):
        field = self.fields_by_name[name]
        self.fields.remove(field)
        self.fields_by_name.pop(field.name)
        if name in self._mandatory_fields_name: self._mandatory_fields_name.remove(name)
        self.__delattr__(name="memory_{name}")
        self._generate_dataclasses()
        

    def reset(self, name = None):
        for name in self.fields_by_name: 
            buf : torch.Tensor = getattr(self, f"memory_{name}")
            field = self.fields_by_name[name]
            field.codec.reset_fill(buf=buf, field=field)
        self.i: int = 0

    def __len__(self): return min(self.i, self.max_length)

    @torch.no_grad()
    def store(self, experience = None, **kwargs):
        """Save a experience"""
        if experience is not None: kwargs.update(asdict(experience))

        # ensure mandatory fields present
        missing = self._mandatory_fields_name.difference(kwargs.keys())
        assert not missing, f"Missing fields: {sorted(missing)}"
        
        i = int(self.i % self.max_length)

        nb_env = torch.as_tensor(next(iter(kwargs.values()))).size(0)

        # Compute indices
        if i+nb_env <= self.max_length: indices = torch.arange(i,i+nb_env)
        else:
            n_end = self.max_length
            n_start = nb_env + i - n_end 
            indices = torch.cat([torch.arange(i, n_end, device=self.device), torch.arange(0, n_start, device=self.device)])
        indices = indices.long()

        for name, val in kwargs.items():
            field = self.fields_by_name.get(name)
            if field is None:
                warnings.warn(f"{name} does not exist as a field in {self.__class__.__name__}. Current fields : {', '.join(self.names)}")
                continue
            val = field.codec.encode(val, field=field, device=self.device)
            getattr(self, f"memory_{name}")[indices] = val

        self.i += nb_env
        return indices

    @torch.no_grad()
    def __getitem__(self, loc) -> ExperienceSample:
        index : int | np.ndarray
        name :str
        if isinstance(loc, tuple):
            name, index= loc
            val = getattr(self, f"memory_{name}")[index]
            field = self.fields_by_name[name]
            return self.fields_by_name[name].codec.decode(val, field=field)
        elif isinstance(loc, str):
            name = loc
            val = getattr(self, f"memory_{name}")[:self.__len__()]
            field = self.fields_by_name[name]
            return self.fields_by_name[name].codec.decode(val, field=field)
        elif isinstance(loc, torch.Tensor):
            indices = loc
            return self.__get_experience_from_indices__(indices=indices)
        raise NotImplementedError(f"Cannot get items from {self} from loc of type {loc.__class__}")
    
    def __getitems__(self, indices): return self.__getitem__(loc = indices)
    
    @torch.no_grad()
    def __compute_experience_from_values__(self, **kwargs): 
        out = {}
        for name, value in kwargs.items():
            field = self.fields_by_name[name]
            out[name] = field.codec.encode(value, field=field, device=self.device)
        return self.experience_dataclass_generator(**out)

    @torch.no_grad()
    def __get_experience_from_indices__(self, indices):
        values = {name : getattr(self, f"memory_{name}")[indices] for name in self.fields_by_name.keys()}
        out = {}
        for name, value in values.items():
            field = self.fields_by_name[name]
            out[name] = field.codec.decode(value, field=field)
        return self.sample_dataclass_generator(
                indices=indices,
                **out
            )

    @torch.no_grad()
    def __setitem__(self, loc, val):
        indices : int | np.ndarray
        name :str
        if isinstance(loc, tuple):
            name, indices= loc
            field = self.fields_by_name[name]
            getattr(self, f"memory_{name}")[indices] = field.codec.encode(val, field=field, device=self.device)
        elif isinstance(loc, str):
            name = loc
            field = self.fields_by_name[name]
            getattr(self, f"memory_{name}")[:self.__len__()] = field.codec.encode(val, field=field, device=self.device)


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
            experience_env = self.__compute_experience_from_values__(**kwargs_env)

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


