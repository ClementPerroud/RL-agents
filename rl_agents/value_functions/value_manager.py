from rl_agents.agent import AbstractAgent
from rl_agents.service import AgentService

from copy import deepcopy
import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.value_functions.dqn_function import V

class VManager(AgentService):
    def set_net(self, net :"V"): self.net = net
    def get_net(self, *args, **kwargs): return self.net

class DoubleVManager(VManager):
    def __init__(
        self,
        tau: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tau = tau
        self._bootstrapped = False

    def set_net(self, net):
        self.net: "V" = net
        self.net_target: "V" = net # Temporarly, net_target = net. Check out initialize_target_if_needed()

    def get_net(self, target = False, *args, **kwargs) -> "V":
        if target: return self.net_target
        return self.net

    def initialize_target_if_needed(self):
        # First update
        if not self._bootstrapped:
            if self.is_initialized(self.net):
                # Once net initialized, we can set the target network
                self.net_target = deepcopy(self.net)
                self._copy_weights()
                self.net_target.requires_grad_(False)
                self._bootstrapped = True
            
    @torch.no_grad()
    def update(self, agent: AbstractAgent):
        self.initialize_target_if_needed()

        if self._bootstrapped and agent.step % self.tau == 0:
            self._copy_weights()

    def _copy_weights(self):
        self.net_target.load_state_dict(self.net.state_dict())

    def is_initialized(self, module : torch.nn.Module) -> bool:
        # 1) Any Lazy submodule still waiting for shape materialization?
        for sm in module.modules():
            if isinstance(sm, LazyModuleMixin) and sm.has_uninitialized_params():
                return False

        # 2) Any params/buffers still "meta" or explicitly uninitialized?
        for t in list(module.parameters()) + list(module.buffers()):
            if isinstance(t, (UninitializedParameter, UninitializedBuffer)):
                return False
            if getattr(t, "is_meta", False) or getattr(getattr(t, "device", None), "type", None) == "meta":
                return False

        return True

class SoftDoubleVManager(DoubleVManager):
    def __init__(
        self,
        tau: int,
        *args,
        **kwargs
    ):  
        super().__init__(tau=tau, *args, **kwargs)
        self.tau_rate = 1.0/float(self.tau)

    @torch.no_grad()
    def update(self, agent: AbstractAgent):
        self.initialize_target_if_needed()
        if self._bootstrapped:
            net_target_state_dict = self.net_target.state_dict()
            net_state_dict = self.net.state_dict()
            for key in net_state_dict:
                if torch.is_floating_point(net_state_dict[key]):
                    net_target_state_dict[key] = net_state_dict[key]*self.tau_rate + net_target_state_dict[key]*(1-self.tau_rate)
                else:
                    net_target_state_dict[key] = net_state_dict[key]
            self.net_target.load_state_dict(net_target_state_dict)


