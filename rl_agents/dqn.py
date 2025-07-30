from rl_agents.element import Agent
from rl_agents.action_model.model import ActionModel 

import torch
import numpy as np


class DQNgent(torch.nn.Module, Agent):
    def __init__(self, action_model : 'ActionModel'):
        self.action_model = action_model.connect(self)
        self.nb_actions = 1
        self.nb_env = 1
