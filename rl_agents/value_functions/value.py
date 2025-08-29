from rl_agents.service import AgentService
from abc import ABC, abstractmethod
import torch

class V(AgentService, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def V(self, state : torch.Tensor) -> torch.Tensor:...


class Q(AgentService, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    def Q(self, state : torch.Tensor, action : torch.Tensor, **kwargs) -> torch.Tensor:...


    @abstractmethod
    def Q_per_action(self, state : torch.Tensor) -> torch.Tensor:...

class Trainable(ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    def compute_td_errors(self, loss_input : torch.Tensor, loss_target : torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def compute_loss_input(self, loss_input : torch.Tensor, loss_target : torch.Tensor) -> torch.Tensor: ...
    
    @abstractmethod
    def compute_loss_target(self, loss_input : torch.Tensor, loss_target : torch.Tensor)-> torch.Tensor: ...
