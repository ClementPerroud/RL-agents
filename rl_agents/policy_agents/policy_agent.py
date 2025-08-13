from rl_agents.agent import AbstractAgent

import torch
from abc import ABC, abstractmethod


class AbstractPolicyAgent(AbstractAgent, ABC):
    ...