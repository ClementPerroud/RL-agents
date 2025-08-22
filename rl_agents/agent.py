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
        self.nb_env = nb_env
        self.policy = policy.connect(self)

        self.episode = 0
        self.step = 0

    @property
    def services(self) -> set["AgentService"]:
        if not hasattr(self, "_services"):  # Triggered the first call
            # Initialize services
            self._services = set()
            self._find_services(self)
        return self._services

    def update(self):
        self.step += 1
        for element in self.services:
            element.update(agent=self)

    def _find_services(self, service: AgentService, _first: bool = True):
        if not _first:
            self.services.add(service)
        for sub_service in service.sub_services:
            self._find_services(service=sub_service, _first=False)

    @abstractmethod
    def train_agent(self):
        assert self.training, "Please set the agent in training mode using .train()"
    
    def pick_action(self, state : np.ndarray):
        state = torch.as_tensor(state, dtype=torch.float32)

        single_env_condition = self.nb_env == 1 and (state.shape[0] != 1 or state.ndim == 1)
        if single_env_condition: state = state.unsqueeze(0)

        action = self.policy.pick_action(agent=self, state=state)
        
        if single_env_condition: action = action.squeeze(0)

        self.update()
        return action.detach().numpy()