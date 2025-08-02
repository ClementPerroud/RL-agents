from rl_agents.service import AgentService

import torch
from abc import ABC


class AbstractDeepQNeuralNetwork(torch.nn.Module, AgentService, ABC):
    ...

