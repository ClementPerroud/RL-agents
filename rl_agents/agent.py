from rl_agents.action_strategy.action_strategy import AbstractActionStrategy
from rl_agents.service import AgentService

from abc import ABC, abstractmethod
import torch
import numpy as np


class AbstractAgent(AbstractActionStrategy, AgentService, ABC):
    def __init__(
        self,
        nb_env: int,
        action_strategy: AbstractActionStrategy,
        device : torch.DeviceObjType = None,
    ):
        self.nb_env = nb_env
        self.action_strategy = action_strategy.connect(self)

        self.episode = 0
        self.step = 0
        self.device = device

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
    def train_agent(self): ...

    @abstractmethod
    def _pick_deterministic_action(self, state: torch.Tensor) -> np.ndarray: ...

    def pick_action(self, state):
        action = self.action_strategy.pick_action(agent=self, state=state)

        self.update()
        return action
