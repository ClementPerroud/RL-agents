from rl_agents.value_functions.dqn_function import DVNFunction
from rl_agents.agent import AbstractAgent
from rl_agents.service import AgentService
import warnings

from abc import ABC, abstractmethod
import torch

class BaseAdvantageFunction(AgentService, ABC):
    @abstractmethod
    def reset(self):
        ...
    
    @abstractmethod
    def compute_advantage(self):
        ...

class GAEFunction(BaseAdvantageFunction):
    def __init__(self, value_function : DVNFunction, gamma : float, lambda_ : float, multi_steps=None):
        self.value_function = value_function
        self.lambda_ = lambda_
        self._state_tp1 = None
        self.advantage_tp1 = 0

    def reset(self, agent: AbstractAgent, device=None):
        self.advantage_tp1 = torch.zeros(agent.nb_env, dtype=torch.float32, device=device)

    def compute_advantage(self,
            state: torch.Tensor,  # [batch, state_shape ...] obtained at t
            action: torch.Tensor,  # [batch] obtained at t+multi_steps
            reward: torch.Tensor,  # [batch] obtained between t+1 and t+multi_step (then summed up using discounted sum)
            next_state: torch.Tensor,  # [batch, state_shapes ...] obtained at t+multi_steps
            done: torch.Tensor,  # [batch] obtained at t+multi_steps
            truncated: torch.Tensor,  # [batch] obtained at t+multi_steps
            **kwargs
        ):

        y_true, y_pred = self.value_function.compute_loss_inputs(state=state, action=action, reward=reward, next_state=next_state, done=done)    
        delta = self.value_function.out_to_value(y_true) - self.value_function.out_to_value(y_pred)

        end = done | truncated
        advantage_t = delta + self.value_function.gamma * self.lambda_ * (1 - end.float()) * self.advantage_tp1
        self.advantage_tp1 = advantage_t
        # self._state_tp1 = state
        return advantage_t
