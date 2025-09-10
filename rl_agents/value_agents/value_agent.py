from rl_agents.agent import BaseAgent

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.policies.policy import AbstractPolicy
    from rl_agents.value_functions.value import Q

import torch
from abc import ABC, abstractmethod


class AbstractValueAgent(BaseAgent, ABC):
    def __init__(self, q_function : 'Q', nb_env : int, policy : 'AbstractPolicy'):
        super().__init__(nb_env = nb_env, policy = policy)
        self.q_function = q_function
    ...
    # @abstractmethod
    # def Q(self, state: torch.Tensor): ...

    # @abstractmethod
    # def Q_a(self, state: torch.Tensor, action: torch.Tensor): ...

    # @abstractmethod
    # def compute_td_errors(self): ...

    # @abstractmethod
    # def compute_loss_inputs(self): ...