from rl_agents.element import Agent
import torch
import numpy as np




class DQNgent(torch.nn.Module, Agent):
    def __init__(self, epsilon_function, model : torch.Module, action_manager : 'ActionManager'):
        self.model = model
        self.epsilon_function = epsilon_function
        self.action_manager = action_manager.connect(self)
        self.nb_actions = 1
        self.nb_env = 1
