from rl_agents.service import AgentService
from rl_agents.value_functions.value import Q

import torch
import gymnasium as gym


class ContinuousQWrapper(Q, AgentService):
    def __init__(self, core_net : torch.nn.Module, action_space : gym.spaces.Box):
        super().__init__()
        # Core net : 
        # inputs [B, state_shape ... ], [B, action_shape] -> [B, hidden_dim]
        self.core_net = core_net
        self.action_space = action_space
        self.head = torch.nn.LazyLinear(1)

    def Q(self, state : torch.Tensor, action : torch.Tensor, **kwargs):
        x : torch.Tensor = self.core_net(state, action) # x : [B, H]
        return self.head(x).squeeze(1) # [B]

    def Q_per_action(self, *args, **kwargs): raise ValueError("Continuous action cannot use Q_per_action.")

class DiscreteQWrapper(Q, AgentService):
    def __init__(self, core_net : torch.nn.Module, action_space : gym.spaces.Discrete):
        super().__init__()
        self.core_net = core_net
        self.action_space = action_space
        self.head = torch.nn.LazyLinear(self.action_space.n)

    def Q_per_action(self, state : torch.Tensor, **kwargs):
        # state : [B, S shape ...]
        x = self.core_net(state)
        return self.head(x) # [B, A]
    
    def Q(self, state : torch.Tensor, action : torch.Tensor, **kwargs):
        # state : [B, S shape ...], action (discrete): [B]
        value_per_action = self.Q_per_action(state=state, **kwargs) # [B, A]
        return torch.gather(input=value_per_action, dim = 1, index= action.long().unsqueeze(-1)).squeeze(-1) # [B]
