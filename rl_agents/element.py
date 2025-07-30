from rl_agents.action_model.model import ActionModel

from abc import ABC, abstractmethod
import torch

class Agent(ABC):
    def __init__(self):
        self.elements = []
    
    def update(self, infos : dict):
        for element in self.elements:
            element.update(infos = infos)
    
    @property
    @abstractmethod
    def action_model(self) -> torch.nn.Module:
        ...

    @property
    @abstractmethod
    def nb_env(self) -> int:
        ...

    @property
    @abstractmethod
    def elements(self) -> list['AgentService']:
        ...

    @property
    @abstractmethod
    def step(self) -> int:
        ...
    
    @property
    @abstractmethod
    def nb_oberservations(self) -> int:
        ...

    
class AgentService(ABC):
    def __init__(self):
        self._agent : Agent = None
        if isinstance(self, Agent):
            self.elements = []
    

    def connect(self, agent : Agent):
        self._agent = agent
        self._agent.elements.append(self)
        return self
    
    @property
    def agent(self) -> 'Agent':
        try:
            return self._agent
        except BaseException as e:
            raise Exception("Please connect the element to an agent at initialization using : element.connect(agent)")

    def update(self, infos : dict):
        ...