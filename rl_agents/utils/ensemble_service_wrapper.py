from rl_agents.value_functions.value import Op
from rl_agents.service import AgentService
from rl_agents.utils.distribution.distribution import distribution_aware
from rl_agents.utils.wrapper import Wrapper

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
import inspect
from typing import Literal, Any

@dataclass(frozen=True)
class OpStrategy:
    reduce: Literal["first", "none", "min", "mean"] = "first"

class EnsembleServiceWrapper(Wrapper, AgentService, ABC):
    # Default kwargs    
    def __init__(self, strategy, services, **kwargs):
        super().__init__(**kwargs)
        self.strategy = strategy 
        self.services = services
        self.wrapped = services[0]

    @abstractmethod
    def _select_nets(self, op_strategy : OpStrategy) -> list[torch.nn.Module]: ...
        
    @distribution_aware
    def _reduce(self, op_strategy : OpStrategy, returns : torch.Tensor):
        match op_strategy.reduce:
            case "none" | "all":    return returns
            case "first":           return returns[0]
            case "last":            return returns[-1]
            case "min":             return returns.amin(0)
            case "max":             return returns.amax(0)
            case "mean":            return returns.mean(0)
    
    @distribution_aware
    def _aggregate_returns(self, op_strategy : OpStrategy, returns : list[Any]):
        if len(returns) == 0: return returns
        if len(returns) == 1: return returns[0]
        
        _first_elem = returns[0]
        if isinstance(_first_elem, torch.Tensor):
            return self._reduce(
                op_strategy=op_strategy, 
                returns=torch.stack(returns)
                )
        if isinstance(_first_elem, tuple) or isinstance(_first_elem, list):
            return _first_elem.__class__([
                self._aggregate_returns(returns= [_return[i] for _return in returns])
                    for i in range(len(_first_elem))
            ])
        raise NotImplementedError(f"Type of return {returns.__class__.__name__} is not supported with EnsembleTargetWrapper")

    def _wrap_method(self, method_name, *args, op : Op = None, **kwargs):
        op_strategy = self.strategy.plan(op)
        nets = self._select_nets(op_strategy=op_strategy)
        returns = [
            getattr(net, method_name)(*args, op = op, **kwargs) 
            for net in nets
        ]
        return self._aggregate_returns(op_strategy=op_strategy, returns = returns)

    def _wrap_attribute(self, attribute_name):
        service = self.service
        return getattr(service, attribute_name)

    
    def __getattr__(self, name):
        # Redirect calls to the targeted services.
        try:
            # First, try to use classic torch.nn.Module __getattr__
            return super().__getattr__(name)
        except AttributeError as e:
            # If it fails, try to get the attribute from the wrapped object.
            any_service = self.services[0]
            wrapped_attr = getattr(any_service, name, None)

            # If the attribute is a class method 
            if inspect.ismethod(wrapped_attr):
                @wraps(wrapped_attr)
                def wrapper(*args, **kwargs):
                    op = kwargs.pop("op", None)
                    return self._wrap_method(name, *args, op=op, **kwargs)
                return wrapper
            elif wrapped_attr is not None:
                return wrapped_attr
            
            raise e
            

        