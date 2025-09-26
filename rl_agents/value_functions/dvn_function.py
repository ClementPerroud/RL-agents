from rl_agents.memory.memory import Experience
from rl_agents.service import AgentService
from rl_agents.value_functions.value import V, Trainable, Op
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
            net : V,
            gamma : float,
            loss_fn : torch.nn.modules.loss._Loss = torch.nn.SmoothL1Loss(),
            **kwargs
        ):
        super().__init__(**kwargs)
        self.net = net
        self.gamma = gamma
        self.loss_fn = loss_fn
        loss_fn.reduction= "none"

    def V(self, state: torch.Tensor, **kwargs) -> torch.Tensor:
        # state : [batch/nb_env, state_shape ...]
        return self.net.V(state=state)
        # Return Q Values : [batch/nb_env]

    def compute_loss_input(self, experience : Experience, **kwargs) -> torch.Tensor:
        return self.V(state=experience.state, op=Op.DVN_LOSS_INPUT_Q, **kwargs)
    
    @torch.no_grad()
    def compute_loss_target(self, experience : Experience, **kwargs) -> torch.Tensor:
        return experience.reward + (1 - experience.done.float()) * self.gamma * self.V(experience.next_state, op = Op.DVN_LOSS_TARGET_Q, **kwargs)

    def compute_td_errors(self, loss_input : torch.Tensor, loss_target : torch.Tensor):
        return (loss_target - loss_input).abs()
