from rl_agents.agent import BaseAgent
from rl_agents.value_functions.value import Op
from rl_agents.service import AgentService
from rl_agents.utils.distribution.distribution import distribution_aware
from rl_agents.utils.wrapper import Wrapper

from copy import deepcopy
from dataclasses import dataclass
import torch
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter
from functools import wraps

from typing import TYPE_CHECKING, Protocol, runtime_checkable, Literal, Union, Any
if TYPE_CHECKING:
    from rl_agents.value_functions.dqn_function import V


class Updater(Protocol):
    def update_targets(self, agent : BaseAgent, target_manager : "TargetManagerWrapper"): ...

@dataclass(frozen=True)
class OpStrategy:
    # who evaluates for "pick_action" and "value"
    target: bool                                                    # use target net to select action?
    reduce_targets: Literal["first", "none", "min", "mean"] = "first"       # aggregator for multi-critic targets


class TargetStrategy(Protocol):
    nb_target : int
    def plan(self, op: Op) -> OpStrategy: ...

class DDQNStrategy(TargetStrategy):
    nb_target = 1
    def plan(self, op: Op) -> OpStrategy:
        match op:
            case Op.ACTOR_PICK_ACTION:              return OpStrategy(target=False)
            case Op.DQN_LOSS_TARGET_PICK_ACTION:    return OpStrategy(target=True)
            case Op.DQN_LOSS_TARGET_Q:              return OpStrategy(target=False)
            case Op.DQN_LOSS_INPUT_Q:               return OpStrategy(target=False)
        raise KeyError(f"op {op} not implemented in strategy {self.__class__.__name__}")

class DDPGStrategy(TargetStrategy):
    nb_target = 1
    def plan(self, op: Op) -> OpStrategy:
        match op:
            case Op.ACTOR_PICK_ACTION:              return OpStrategy(target=False)
            case Op.DQN_LOSS_TARGET_PICK_ACTION:    return OpStrategy(target=True)
            case Op.DQN_LOSS_TARGET_Q:              return OpStrategy(target=True)
            case Op.DQN_LOSS_INPUT_Q:               return OpStrategy(target=False)
            case Op.DDPG_LOSS_ACTOR_PICK_ACTION:    return OpStrategy(target=False)
            case Op.DDPG_LOSS_CRITIC_Q:             return OpStrategy(target=False)
        raise KeyError(f"op {op} not implemented in strategy {self.__class__.__name__}")

class TargetManagerWrapper(Wrapper, AgentService):
    # Default kwargs    
    def __init__(
        self,
        service : AgentService,
        target_strategy : TargetStrategy,
        updater : Updater,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.updater = updater
        self.service = service
        self.target_strategy = target_strategy
        self.target_services = [service for _ in range(self.target_strategy.nb_target)] # Temporarly, target_service = service. Check out initialize_target_if_needed()

        self._bootstrapped = False
        self._method_cache: dict[str, Any] = {}
        self.wrapped = self.service # Respect Wrapper proto


    def _select_nets(self, op_strategy : OpStrategy) -> list[torch.nn.Module]:
        if not op_strategy.target: return [self.service]
        # Target
        match op_strategy.reduce_targets:
            case "first" | "none":  return [self.target_services[0]]
            case "min" | "mean":    return self.target_services
            case _:                 raise KeyError(op_strategy.reduce_targets)
    
    def _reduce(self, op_strategy : OpStrategy, returns : torch.Tensor):
        match op_strategy.reduce_targets:
            case "first" | "none":  return returns[0]
            case "min":             return returns.min(0)
            case "mean":            return returns.mean(0)

    @distribution_aware
    def _wrap_method(self, method_name, op : Op, *args, **kwargs):
        op_strategy = self.target_strategy.plan(op)
        nets = self._select_nets(op_strategy=op_strategy)
        returns = [
            getattr(net, method_name)(*args, op = op, **kwargs) 
            for net in nets
        ]
        returns = torch.stack(returns)
        return self._reduce(op_strategy=op_strategy, returns = returns)

    def _wrap_attribute(self, attribute_name):
        service = self.service
        return getattr(service, attribute_name)

    def initialize_target_if_needed(self):
        # First update
        if not self._bootstrapped:
            if self.is_initialized(self.service):
                # Once net initialized, we can set the target network
                for i in range(len(self.target_services)):
                    self.target_services[i] = deepcopy(self.service).eval().requires_grad_(False)
                    self.target_services[i].load_state_dict(self.service.state_dict())
                self._bootstrapped = True
            
    @torch.no_grad()
    def update(self, agent: BaseAgent, **kwargs):
        self.initialize_target_if_needed()
        if self._bootstrapped:
            self.updater.update_targets(agent = agent, target_manager=self)

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
    
    # Q implementation
    def V(self, state: torch.Tensor, op : Op, **kwargs) -> torch.Tensor: return self._wrap_method("V", state=state, op=op, **kwargs)
    def Q(self, state: torch.Tensor, action : torch.Tensor, op : Op, **kwargs) -> torch.Tensor: return self._wrap_method("Q", state=state, action=action, op=op, **kwargs)
    def Q_per_action(self, state: torch.Tensor, op : Op, **kwargs) -> torch.Tensor: return self._wrap_method("Q_per_action", state=state, op=op, **kwargs)
    
    # Policy implementation
    def pick_action(self, state: torch.Tensor, op : Op, **kwargs) -> torch.Tensor: return self._wrap_method("pick_action", state=state, op=op, **kwargs)

    # Q51 implementation
    @property
    def nb_atoms(self): return self._wrap_attribute("nb_atoms")

    @property
    def v_min(self): return self._wrap_attribute("v_min")

    @property
    def v_max(self): return self._wrap_attribute("v_max")

    @property
    def atom_config(self): return self._wrap_attribute("atom_config")

class HardUpdater(AgentService):
    def __init__(self, rate : float):
        super().__init__()
        self.rate = float(rate)
    def update_targets(self, target_manager : TargetManagerWrapper):
        service = target_manager.service
        for target_service in target_manager.target_services:
            target_service.load_state_dict(service.state_dict())
    
class SoftUpdater(AgentService):
    def __init__(self, rate : float, update_every : int):
        super().__init__()
        self.rate = float(rate)
        self.update_every = update_every

    def update_targets(self, agent : BaseAgent, target_manager : TargetManagerWrapper):
        if agent.nb_step % self.update_every == 0:
            service = target_manager.service
            for target_service in target_manager.target_services:
                target_service_state_dict = target_service.state_dict()
                net_state_dict = service.state_dict()
                for key in net_state_dict:
                    if torch.is_floating_point(net_state_dict[key]):
                        target_service_state_dict[key] = net_state_dict[key]*self.rate + target_service_state_dict[key]*(1-self.rate)
                    else:
                        target_service_state_dict[key] = net_state_dict[key]
                target_service.load_state_dict(target_service_state_dict)