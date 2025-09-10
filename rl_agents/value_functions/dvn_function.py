from rl_agents.memory.memory import Experience
from rl_agents.service import AgentService
from rl_agents.value_functions.value_manager import VManager
from rl_agents.value_functions.value import V, Trainable
from rl_agents.utils.mode import eval_mode

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.agent import BaseAgent

import torch

class VWrapper(AgentService):
    def __init__(self, core_net : torch.nn.Module):
        super().__init__()
        self.core_net = core_net
        self.head = torch.nn.LazyLinear(1)

    def V(self, state : torch.Tensor) -> torch.Tensor:
        x = self.core_net(state)
        return self.head(x).squeeze(1)


class DVN(AgentService, V, Trainable):
    def __init__(self,
            net : AgentService,
            gamma : float,
            loss_fn : torch.nn.modules.loss._Loss = torch.nn.SmoothL1Loss(),
            manager : VManager = VManager(),
            **kwargs
        ):
        super().__init__(**kwargs)
        self.manager = manager
        self.manager.set_net(net)
        
        self.gamma = gamma
        self.loss_fn = loss_fn
        loss_fn.reduction= "none"

    def V(self, state: torch.Tensor, target = False, **kwargs) -> torch.Tensor:
        # state : [batch/nb_env, state_shape ...]
        return self.manager.get_net(target=target).V(state)
        # Return Q Values : [batch/nb_env]

    def compute_loss_input(self, experience : Experience, **kwargs) -> torch.Tensor:
        return self.V(experience.state, **kwargs)
    

    @torch.no_grad()
    def compute_loss_target(self, experience : Experience, **kwargs) -> torch.Tensor:
        default_kwargs = {
            "pick_action_kwargs":{ #passed to : V -> self.policy.pick_action
                "get_net_kwargs": self.manager.compute_loss_target_default #passed to : self.policy.q_per_action = Q_per_paction
            }
        }
        kwargs = {**kwargs, **default_kwargs}
        return experience.reward + (1 - experience.done.float()) * self.gamma * self.V(experience.next_state, **kwargs)

    def compute_td_errors(self, loss_input : torch.Tensor, loss_target : torch.Tensor):
        return (loss_target - loss_input).abs()
