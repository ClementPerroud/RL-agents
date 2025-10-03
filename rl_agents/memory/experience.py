from typing import Protocol
import torch
from dataclasses import dataclass,  fields

class ExperienceLike(Protocol):
    indices : torch.Tensor
    state : torch.Tensor
    action : torch.Tensor
    reward : torch.Tensor
    next_state : torch.Tensor
    done : torch.Tensor
    truncated : torch.Tensor

class ExpWithLogProb(Protocol):
    log_prob : torch.Tensor

@dataclass(kw_only=True, slots=True, frozen=True)
class Experience:
    indices : torch.Tensor = None
    def __getattr__(self, item):
        # Called only if attribute not found normally
        raise AttributeError(
            f"'The replay_memory related to {type(self).__name__}' has no attribute '{item}'. Your replay_memory is probably not meant to be use in this context."
            f"Available attributes: {[field.name for field in fields(self)]}"
        )

