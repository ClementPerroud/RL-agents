from typing import TYPE_CHECKING
from abc import ABC, abstractmethod


class AgentService(ABC):
    def __init__(self):
        self.training = True
        self._childs = []

    def connect(self, parent_agent_service : 'AgentService'):
        self._parent = parent_agent_service
        self._parent._childs.append(self)
        return self
    
    def train(self): self.training = True
    def eval(self): self.training = False

    def update(self, infos : dict):
        ...
    
    # @property
    # def agent(self) -> 'Agent':
    #     try:
    #         return self._agent
    #     except AttributeError as e: # Triggered if self._agent does not exist
    #         # Setup the reference
    #         if not isinstance(self.parent, 'AgentService'): self._agent = self.parent
    #         else: self._agent = self.parent.agent
    #         return self._agent

    # @property
    # def parent(self) -> 'AgentService':
    #     try:
    #         return self._parent
    #     except AttributeError as e: # Triggered if self._pagent does not exist. It meens to instance have not been connected using .connect(...)
    #         raise Exception(f"This element cannot be used without being associated with its parent. If your are developping new services, please connect the element (class : {self.__class__.__name__}) to an agent at initialization using : element.connect(agent)")
    