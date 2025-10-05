from rl_agents.memory.codec import MemoryCodec, TensorCodec
from rl_agents.memory.experience import Experience
from typing import Protocol, NamedTuple, Type, runtime_checkable
import torch
import numpy as np
import warnings
from dataclasses import dataclass, make_dataclass, asdict, fields


class MemoryField (NamedTuple):
    name : str
    shape : tuple
    dtype : Type
    codec : MemoryCodec = TensorCodec()
    default : None = None

@runtime_checkable
class Memory[T](Protocol):
    max_length : int
    fields : list[MemoryField]
    device : torch.DeviceObjType
    def store(self, **kwargs) -> tuple[torch.Tensor, object]:...
    def reset(self): ...
    def __len__(self) -> int : ...
    def __getitem__(self, index) -> T: ...
    def __getitems__(self, index) -> T: ...
    def __setitem__(self, loc, val): ...

class EditableMemory[T](Memory[T]):
    def add_field(self, field : MemoryField):...
    def remove_field(self, name : str):...

class BaseExperienceMemory(torch.nn.Module, EditableMemory[Experience]):
    def __init__(
        self,
        max_length: int,
        fields: list[MemoryField],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_length = int(max_length)


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
    def device(self) -> torch.DeviceObjType:
        try:
            return next(self.buffers()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def names(self) -> list[str]:
        return list(self.fields_by_name.keys())

    def _generate_dataclasses(self):
        # Create the dataclasses dynamically
        experience_fields = [(field.name, field.codec.tensor_class) for field in self.fields]

        self.experience_class = make_dataclass(
            f"Experience{self.__class__.__name__}", fields=experience_fields,
            bases=(Experience,), kw_only=True, slots=True, frozen=True
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
    def __getitem__(self, loc) -> Experience:
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
        return self.experience_class(**out)

    @torch.no_grad()
    def __get_experience_from_indices__(self, indices):
        values = {name : getattr(self, f"memory_{name}")[indices] for name in self.fields_by_name.keys()}
        out = {}
        for name, value in values.items():
            field = self.fields_by_name[name]
            out[name] = field.codec.decode(value, field=field)
        return self.experience_class(
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
