from rl_agents.value_functions.dvn_function import DVN
from rl_agents.policies.policy import Policy
from rl_agents.memory.memory import Experience
from rl_agents.service import AgentService
from rl_agents.value_functions.value import Q, V, Trainable, Op
from rl_agents.policies.value_policy import DiscreteBestQValuePolicy
from rl_agents.utils.hidden_modules import HiddenModulesUtilsMixin
from rl_agents.utils.mode import eval_mode

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.agent import BaseAgent

from typing import Callable
import torch
import gymnasium as gym
from types import SimpleNamespace

class _QWrapper():
    

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


class DQN(HiddenModulesUtilsMixin, DVN):
    def __init__(self,
            net : Q,
            gamma : float,
            loss_fn : torch.nn.modules.loss._Loss = torch.nn.SmoothL1Loss(),
            policy : Policy = None,
            **kwargs
        ):
        super().__init__(net=net, gamma=gamma, loss_fn=loss_fn, **kwargs)
        assert isinstance(self.net, Q), "net must implement Q. Please use the wrappers of create a custom Q."
        self.net : Q
        
        if policy == None or policy == "best": policy = DiscreteBestQValuePolicy(q = self)
        self.hidden.policy = policy
        self.gamma = gamma
        self.loss_fn = loss_fn
        loss_fn.reduction= "none"


    def Q(self, state: torch.Tensor, action: torch.Tensor, op = None) -> torch.Tensor:
        return self.net.Q(state=state, action=action, op = op)

    def Q_per_action(self, state : torch.Tensor, op = None):
        return self.net.Q_per_action(state=state, op = op)
    
    def V(self, state: torch.Tensor, op = (None, None)) -> torch.Tensor:
        action = self.hidden.policy.pick_action(state = state, op = op[0])
        return self.Q(state=state, action=action, op=op[1])

    def compute_loss_input(self, experience : Experience) -> torch.Tensor:
        return self.Q(experience.state, experience.action, op = Op.DQN_LOSS_INPUT_Q)
    
    @torch.no_grad()
    def compute_loss_target(self, experience : Experience) -> torch.Tensor:
        return experience.reward + (1 - experience.done.float()) * self.gamma * self.V(experience.next_state, op = (Op.DQN_LOSS_TARGET_PICK_ACTION, Op.DQN_LOSS_TARGET_Q))

