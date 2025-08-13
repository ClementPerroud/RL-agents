from rl_agents.agent import AbstractAgent
from rl_agents.service import AgentService

from abc import ABC, abstractmethod
from typing import Callable
import torch

class AbstractQFunction(AgentService, ABC):
    @abstractmethod
    def Q(self, state: torch.Tensor, training : bool) -> torch.Tensor: ...

    @abstractmethod
    def Q_a(self, state: torch.Tensor, action: torch.Tensor, training : bool) -> torch.Tensor: ...

    @abstractmethod
    def train_q_function(self,
        state: torch.Tensor,  # [batch, state_shape ...] obtained at t
        action: torch.Tensor,  # [batch] obtained at t+multi_steps
        reward: torch.Tensor,  # [batch] obtained between t+1 and t+multi_step (then summed up using discounted sum)
        next_state: torch.Tensor,  # [batch, state_shapes ...] obtained at t+multi_steps
        done: torch.Tensor,  # [batch] obtained at t+multi_steps
        weight: torch.Tensor = None,
        callbacks_q_function_training: list[Callable] = [],
        **kwargs) -> None: ...

    @abstractmethod
    def compute_td_errors(
        self,
        y_true: torch.Tensor,  # [batch, state_shape ...] obtained at t
        y_pred: torch.Tensor,  # [batch] obtained at t+multi_steps
    )-> torch.Tensor:...
        
    @abstractmethod
    def compute_target_predictions(
        self,
        state: torch.Tensor,  # [batch, state_shape ...] obtained at t
        action: torch.Tensor,  # [batch] obtained at t+multi_steps
        reward: torch.Tensor,  # [batch] obtained between t+1 and t+multi_step (then summed up using discounted sum)
        next_state: torch.Tensor,  # [batch, state_shapes ...] obtained at t+multi_steps
        done: torch.Tensor,  # [batch] obtained at t+multi_steps
    )-> torch.Tensor:...