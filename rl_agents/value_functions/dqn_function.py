from rl_agents.value_functions.dvn_function import DVN
from rl_agents.policies.policy import AbstractPolicy
from rl_agents.replay_memory.replay_memory import ReplayMemory, ExperienceSample, Experience
from rl_agents.service import AgentService
from rl_agents.value_functions.value import Q, V, Trainable
from rl_agents.value_functions.value_manager import VManager
from rl_agents.policies.value_policy import DiscreteBestQValuePolicy
from rl_agents.utils.mode import eval_mode

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent

from typing import Callable
import torch
import gymnasium as gym


class ContinuousQWrapper(Q):
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

class DiscreteQWrapper(Q):
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


class DQN(DVN, Q, V, Trainable):
    def __init__(self,
            net : Q,
            gamma : float,
            loss_fn : torch.nn.modules.loss._Loss = torch.nn.SmoothL1Loss(),
            policy : AbstractPolicy = None,
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


    def Q(self, state: torch.Tensor, action: torch.Tensor, target = False, **kwargs) -> torch.Tensor: return self.manager.get_net(target = target).Q(state=state, action=action, **kwargs)
    def Q_per_action(self, state : torch.Tensor, target = False,**kwargs): return self.manager.get_net(target =target).Q_per_action(state=state, **kwargs)
    
    def V(self, state: torch.Tensor, q_target = False, **kwargs) -> torch.Tensor:
        action = self.policy.pick_action(state = state, **kwargs)
        return self.Q(state=state, action=action, target = q_target)

    # Loss config
    def compute_loss_input(self, experience : Experience) -> torch.Tensor: 
        return self.Q(experience.state, experience.action)
    
    @torch.no_grad()
    def compute_loss_target(self, experience : Experience) -> torch.Tensor:
        return experience.reward + (1 - experience.done.float()) * self.gamma * self.V(experience.next_state, q_target = True)

