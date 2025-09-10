from rl_agents.value_functions.dvn_function import DVN
from rl_agents.policies.policy import Policy
from rl_agents.memory.memory import Experience
from rl_agents.service import AgentService
from rl_agents.value_functions.value import Q, V, Trainable
from rl_agents.value_functions.value_manager import VManager
from rl_agents.policies.value_policy import DiscreteBestQValuePolicy
from rl_agents.utils.mode import eval_mode

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.agent import BaseAgent

from typing import Callable
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


class DQN(DVN):
    def __init__(self,
            net : Q,
            gamma : float,
            loss_fn : torch.nn.modules.loss._Loss = torch.nn.SmoothL1Loss(),
            policy : Policy = None,
            manager : VManager = VManager(),
            **kwargs
        ):
        assert isinstance(net, Q), "net must implement Q. Please use the wrappers of create a custom Q."
        super().__init__(net=net, gamma=gamma, loss_fn=loss_fn, manager=manager, **kwargs)

        self.policy = policy
        self.gamma = gamma
        self.loss_fn = loss_fn
        loss_fn.reduction= "none"
        if policy == None or policy == "best": self.policy = DiscreteBestQValuePolicy(q = self)


    def Q(self, state: torch.Tensor, action: torch.Tensor, get_net_kwargs : dict = {}, q_kwargs :dict = {}) -> torch.Tensor:
        return self.manager.get_net(**get_net_kwargs).Q(state=state, action=action, **q_kwargs)

    def Q_per_action(self, state : torch.Tensor, get_net_kwargs : dict = {}, q_kwargs :dict = {}):
        return self.manager.get_net(**get_net_kwargs).Q_per_action(state=state, **q_kwargs)
    
    def V(self, state: torch.Tensor, pick_action_kwargs : dict = {}, q_kwarks : dict = {}) -> torch.Tensor:
        action = self.policy.pick_action(state = state, **pick_action_kwargs)
        return self.Q(state=state, action=action, **q_kwarks)

    def compute_loss_input(self, experience : Experience, **kwargs) -> torch.Tensor:
        return self.Q(experience.state, experience.action, **kwargs)
    
