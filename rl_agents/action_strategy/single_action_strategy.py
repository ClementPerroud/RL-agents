from rl_agents.action_strategy.action_strategy import AbstractActionStrategy
from rl_agents.agent import AbstractAgent
import numpy as np
import torch
# Use for testing purpose
class SingleActionProxy(
        AbstractActionStrategy,
    ):
    def __init__(self, action):
        super().__init__()
        self.action = np.array(action)


    def pick_action(self, agent : AbstractAgent, state : np.ndarray):
        if agent.nb_env > 1: # Return must np.array of shape [nb_env,]
            return np.ones(shape = (agent.nb_env,)) * self.action
        # Else : nb_env = 1
        return self.action
