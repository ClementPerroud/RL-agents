from rl_agents.agent import AbstractAgent
from rl_agents.service import AgentService
from copy import deepcopy
import torch
from typing import Callable


class DoubleQNNProxy(AgentService):
    def __init__(
        self,
        q_net : 'AgentService',
        tau: int,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.q_net: AgentService = q_net
        self.q_net_target: AgentService = deepcopy(q_net)
        self.q_net_target.requires_grad_(False)

        self._copy_weights()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.q_net.forward(*args, **kwargs)
        return self.q_net_target.forward(*args, **kwargs)

    @torch.no_grad()
    def update(self, agent: AbstractAgent):
        if agent.step % self.tau == 0:
            self._copy_weights()

    def _copy_weights(self):
        self.q_net_target.load_state_dict(self.q_net.state_dict())

class SoftDoubleQNNProxy(DoubleQNNProxy):
    def __init__(
        self,
        q_net : 'AgentService',
        tau: int,
        **kwargs
    ):
        super().__init__(q_net=q_net, tau=tau)

    @torch.no_grad()
    def update(self, agent: AbstractAgent):
        target_net_state_dict = self.q_net_target.state_dict()
        net_state_dict = self.q_net.state_dict()
        for key in net_state_dict:
            target_net_state_dict[key] = net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.q_net_target.load_state_dict(target_net_state_dict)


