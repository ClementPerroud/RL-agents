from typing import TYPE_CHECKING

from rl_agents.action_model.model import ActionModel
from rl_agents.service import AgentService

from abc import ABC, abstractmethod
import torch


class Agent(ActionModel, ABC):
    def __init__(self,
            action_model : ActionModel,
            nb_env,
        ):
        self.action_model = action_model
        self.nb_env = nb_env
        
        self.episode = 0
        self.step = 0

        self._childs = []
        self.services : set[AgentService] = set()
        self._find_services(self)

    def update(self, infos : dict):
        for element in self.services:
            element.update(infos = infos)

    def _find_services(self, service : AgentService, _first = True):
        if not _first: self.services.add(service)
        for sub_service in service._childs:
            self._find_services(service= sub_service, _first = False)

