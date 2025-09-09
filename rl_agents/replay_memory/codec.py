from typing import TYPE_CHECKING
if TYPE_CHECKING: 
    from rl_agents.replay_memory.memory import MemoryField
    

from typing import Protocol, runtime_checkable, Any, Tuple, Type
import torch
from dataclasses import dataclass
import torch

@runtime_checkable
class HasCodec(Protocol):
    def get_codec(self) -> "MemoryCodec":
        ...

class AutomaticCodecFactory:
    def generate_codec_from_item(self, val : object):
        if isinstance(val,  HasCodec):
            return val.get_codec()
        elif isinstance(val, torch.Tensor):
            return TensorCodec()
        raise NotImplementedError("Not implemented")

# ---- Protocol ---------------------------------------------------------------
class MemoryCodec(Protocol):
    tensor_class : Type
    def allocate(self, size, field : "MemoryField", device) -> torch.Tensor: ...
    def reset_fill(self, buf, field : "MemoryField") -> None: ...
    def encode(self, value, field : "MemoryField", device) -> torch.Tensor: ...
    def decode(self, tensor, field : "MemoryField") -> Any: ...

# ---- Default tensor codec ---------------------------------------------------
class TensorCodec:
    tensor_class = torch.Tensor
    def allocate(self, size, field : "MemoryField", device):
        fill = 0 if field.default is None else field.default
        return torch.full(size, fill_value=fill, dtype=field.dtype, device=device)

    def reset_fill(self, buf, field : "MemoryField"):
        fill = 0 if field.default is None else field.default
        buf.fill_(fill)

    def encode(self, value, field : "MemoryField", device):
        return torch.as_tensor(value, dtype=field.dtype, device=device)

    def decode(self, tensor, field : "MemoryField"):
        return tensor