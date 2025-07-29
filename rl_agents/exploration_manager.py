
from rl_agents.element import AgentService
from abc import ABC, abstractmethod
import numpy as np

class ActionManager(ABC, AgentService):
    @abstractmethod
    def get_action(self):
        ...

class EspilonGreedyActionManager(ActionManager):
    def __init__(self, q = (1 - 5E-5)):
        self.q = q
        self.epsilon = 1

    def epsilon_function(self):
        return max(0.001, self.q** self.agent.step)
    
    def get_action(self, states : np.ndarray):
        # states : (nb_env)
        rands = np.random.rand(self.agent.nb_env) # shape : (nb_env,)
        column_mask = rands < self.epsilon
        masked_states = states[rands]




    def update(self, infos):
        self.epsilon = self.epsilon_function()