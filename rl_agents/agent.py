from rl_agents.policies.policy import AbstractPolicy
from rl_agents.service import AgentService

from abc import ABC, abstractmethod
import torch
import numpy as np


class AbstractAgent(AbstractPolicy, AgentService, ABC):
    def __init__(
        self,
        nb_env: int,
        policy: AbstractPolicy
    ):
        AgentService.__init__(self)
        AbstractPolicy.__init__(self)
        self.nb_env = nb_env
        self.policy = policy

        self.episode = 0
        self.step = 0

    def update(self, agent : 'AbstractAgent'):
        self.step += 1

    @abstractmethod
    def train_agent(self):
        assert self.training, "Please set the agent in training mode using .train()"
    
    def pick_action(self, state : np.ndarray):
        state = torch.as_tensor(state, dtype=torch.float32)

        single_env_condition = self.nb_env == 1 and (state.shape[0] != 1 or state.ndim == 1)
        if single_env_condition: state = state.unsqueeze(0)

        pick_action_return = self.policy.pick_action(agent=self, state=state)
        
        if single_env_condition: 
            if isinstance(pick_action_return, torch.Tensor): pick_action_return = pick_action_return.squeeze(0)
            if isinstance(pick_action_return, tuple):
                pick_action_return = (elem.squeeze(0) for elem in pick_action_return)
            else: ValueError("Pick action must be a Tensor or a tuple of Tensor")

        if isinstance(pick_action_return, torch.Tensor): pick_action_return = pick_action_return.detach().numpy()
        if isinstance(pick_action_return, tuple): pick_action_return = (elem.detach().numpy() for elem in pick_action_return)
        else: ValueError("Pick action must be a Tensor or a tuple of Tensor")

        self.__update__(agent=self)
        return pick_action_return