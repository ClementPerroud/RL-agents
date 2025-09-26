from rl_agents.service import AgentService
from abc import ABC, abstractmethod
import torch
from enum import Enum, auto
from typing import Protocol, runtime_checkable

@runtime_checkable
class V(Protocol):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def V(self, state : torch.Tensor) -> torch.Tensor:...

@runtime_checkable
class Q(Protocol):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def Q(self, state : torch.Tensor, action : torch.Tensor, **kwargs) -> torch.Tensor:...
    def Q_per_action(self, state : torch.Tensor) -> torch.Tensor:...

@runtime_checkable
class Trainable(Protocol):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def compute_td_errors(self, loss_input : torch.Tensor, loss_target : torch.Tensor) -> torch.Tensor: ...
    def compute_loss_input(self, loss_input : torch.Tensor, loss_target : torch.Tensor) -> torch.Tensor: ...
    def compute_loss_target(self, loss_input : torch.Tensor, loss_target : torch.Tensor)-> torch.Tensor: ...


class Op(Enum):
    # Policy
    ACTOR_PICK_ACTION = auto()
    
    # V functions
    DVN_LOSS_INPUT_Q = auto()
    DVN_LOSS_TARGET_Q = auto()

    # Q functions
    DQN_LOSS_INPUT_Q = auto()
    DQN_LOSS_TARGET_PICK_ACTION = auto()
    DQN_LOSS_TARGET_Q = auto()

    # DDPG
    DDPG_LOSS_ACTOR_PICK_ACTION = auto()
    DDPG_LOSS_CRITIC_Q = auto()
