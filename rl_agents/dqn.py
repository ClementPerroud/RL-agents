from rl_agents.agent import Agent
from rl_agents.replay_memory import AbstractReplayMemory

import torch
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.action_model.model import ActionModel
    

class BaseDQNgent(
        torch.nn.Module,
        Agent,
        AbstractReplayMemory, # DQN agent need to have a replay memory.
        ):
    def __init__(self, gamma, tau, replay_memory : AbstractReplayMemory, action_model : 'ActionModel', nb_env : int):
        torch.nn.Module.__init__(self)
        Agent.__init__(self, action_model = action_model, nb_env = nb_env)
        self.gamma = gamma
        self.tau = tau
        self.replay_memory = replay_memory
    
    def pick_action(self, states):
        return self.action_model.pick_action(states)
    
    def store(self, **kwargs):
        assert self.training, "Cannot store any memory during eval."
        self.replay_memory.store(**kwargs)
    


    
    
# ---- Training
# agent.train()
# while True:
#   action = agent.pick_action()
#   env.act(action)
#   agent.store()
#   agent.train()

# ---- Training
# agent.eval()
# while True:
#   action = agent.pick_action()
#   env.act(action)
