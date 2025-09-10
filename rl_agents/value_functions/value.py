from rl_agents.service import AgentService
from abc import ABC, abstractmethod
import torch
from typing import Protocol, runtime_checkable

@runtime_checkable
class V(Protocol):
    def __init__(self, **kwargs): super().__init__()
    def V(self, state : torch.Tensor) -> torch.Tensor:...

@runtime_checkable
class Q(Protocol):
    def __init__(self, **kwargs): super().__init__()
    def Q(self, state : torch.Tensor, action : torch.Tensor, **kwargs) -> torch.Tensor:...
    def Q_per_action(self, state : torch.Tensor) -> torch.Tensor:...

@runtime_checkable
class Trainable(Protocol):
    def __init__(self, **kwargs): super().__init__()
    def compute_td_errors(self, loss_input : torch.Tensor, loss_target : torch.Tensor) -> torch.Tensor: ...
    def compute_loss_input(self, loss_input : torch.Tensor, loss_target : torch.Tensor) -> torch.Tensor: ...
    def compute_loss_target(self, loss_input : torch.Tensor, loss_target : torch.Tensor)-> torch.Tensor: ...
