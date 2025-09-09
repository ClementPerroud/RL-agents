from rl_agents.replay_memory.codec import MemoryCodec, TensorCodec

from typing import TYPE_CHECKING, Protocol, NamedTuple, Type, runtime_checkable
import torch



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
    def store(self, **kwargs) -> tuple[torch.Tensor, object]:...
    def reset(self): ...
    def __len__(self) -> int : ...
    def __getitem__(self, index) -> T: ...
    def __getitems__(self, index) -> T: ...
    def __setitem__(self, loc, val): ...

class EditableMemory[T](Memory[T]):
    def add_field(self, field : MemoryField):...
    def remove_field(self, name : str):...
