from rl_agents.agent import BaseAgent
from rl_agents.value_functions.value import Op
from rl_agents.service import AgentService
from rl_agents.utils.distribution.distribution import distribution_aware
from rl_agents.utils.wrapper import Wrapper
from rl_agents.utils.ensemble_service_wrapper import EnsembleServiceWrapper, OpStrategy

import torch
from copy import deepcopy
from typing import Protocol

class MultiWrapperStrategy(Protocol):
    nb_services : int
    def plan(self, op: Op) -> OpStrategy: ...

class TD3Strategy(MultiWrapperStrategy):
    def plan(self, op: Op) -> OpStrategy:
        match op:
            # Actor Loss
            case Op.DQN_LOSS_TARGET_Q:              return OpStrategy(reduce="min")
            case Op.DQN_LOSS_INPUT_Q:               return OpStrategy(reduce="none")
            # Critic Loss
            case Op.DDPG_LOSS_CRITIC_Q:             return OpStrategy("min")
            # case Op.DDPG_LOSS_ACTOR_PICK_ACTION:    return OpStrategy(reduce="min")
            # case Op.DDPG_LOSS_ACTOR_PICK_ACTION:    return OpStrategy()
        raise KeyError(f"op {op} not implemented in strategy {self.__class__.__name__}")


class MultiWrapper(EnsembleServiceWrapper):
    # Default kwargs    
    def __init__(
        self,
        service : AgentService,
        n : int,
        strategy : MultiWrapperStrategy,
        **kwargs
    ):
        super().__init__(
            strategy=strategy, 
            services=torch.nn.ModuleList([deepcopy(service) for i in range(n)]), 
            **kwargs)
        self.n = n

    def _select_nets(self, op_strategy : OpStrategy) -> list[torch.nn.Module]:
        match op_strategy.reduce:
            case "none" | "min" | "max":    return self.services
    