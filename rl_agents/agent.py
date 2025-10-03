from rl_agents.policies.policy import Policy
from rl_agents.service import AgentService
from rl_agents.memory.memory import Memory
from rl_agents.value_functions.value import Op

from typing import Protocol, runtime_checkable
from abc import ABC, abstractmethod
import torch
import numpy as np
import time
import datetime

@runtime_checkable
class Agent(Protocol):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    memory: Memory
    nb_step : int
    nb_episode : int

class BaseAgent(Agent, Policy, AgentService, ABC):
    def __init__(
        self,
        nb_env: int,
        policy: Policy,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nb_env = nb_env
        self.policy = policy

        self.nb_episode = 0
        self.nb_step = 0
        self.start_time = time.time()

    @abstractmethod
    def train_agent(self):
        assert self.training, "Please set the agent in training mode using .train()"

    def step(self, **kwargs):
        self.nb_step += 1
        self.__update__(**kwargs)
        if any([key in kwargs for key in ["done", "truncated", "terminated"]]):
            done : np.ndarray = torch.as_tensor(kwargs.get("done", False))
            truncated : np.ndarray = torch.as_tensor(kwargs.get("truncated", False))
            terminated : np.ndarray = torch.as_tensor(kwargs.get("terminated", False))

            need_reset = done | truncated | terminated
            if need_reset.any():
                env_ids = np.nonzero(need_reset)
                self.__reset__(env_ids = np.nonzero(need_reset))
                self.nb_episode += int(env_ids.numel())
            

    def pick_action(self, state : np.ndarray):
        state = torch.as_tensor(state, dtype=torch.float32)

        single_env_condition = self.nb_env == 1 and (state.shape[0] != 1 or state.ndim == 1)

        if single_env_condition: state = state.unsqueeze(0)

        pick_action_return = self.policy.pick_action(state=state, op=Op.ACTOR_PICK_ACTION)
        
        if single_env_condition: 
            if isinstance(pick_action_return, torch.Tensor): pick_action_return = pick_action_return.squeeze(0)
            elif isinstance(pick_action_return, tuple): pick_action_return = tuple([elem.squeeze(0) for elem in pick_action_return])
            else: raise ValueError("Pick action must be a Tensor or a tuple of Tensor")

        if isinstance(pick_action_return, torch.Tensor): pick_action_return = pick_action_return.detach().numpy()
        elif isinstance(pick_action_return, tuple): pick_action_return = tuple([elem.detach().numpy() for elem in pick_action_return])
        else: raise ValueError("Pick action must be a Tensor or a tuple of Tensor")

        return pick_action_return

    def __reset__(self, env_ids : np.ndarray):
        for element in self.modules():
            if isinstance(element, AgentService):
                element.reset(agent=self, env_ids=env_ids)

    def __update__(self, **kwargs):
        for element in self.modules():
            if isinstance(element, AgentService):
                element.update(agent=self, **kwargs)
    
    def duration(self) -> datetime.timedelta:
        return datetime.timedelta(seconds=time.time() - self.start_time)