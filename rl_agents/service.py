from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import torch
from enum import Enum

if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent


class AgentService(torch.nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def connect(self, parent_agent_service: "AgentService"):
        parent_agent_service.sub_services.append(self)
        return self

    @property
    def sub_services(self) -> list["AgentService"]:
        if not hasattr(self, "_sub_services"):
            self._sub_services = []
        return self._sub_services

    def update(self, agent: "AbstractAgent"): ...
