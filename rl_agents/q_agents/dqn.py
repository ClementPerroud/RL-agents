from rl_agents.agent import AbstractAgent, Mode
from rl_agents.replay_memory.replay_memory import AbstractReplayMemory, MultiStepReplayMemory
from rl_agents.q_agents.deep_q_model import AbstractDeepQNeuralNetwork
from rl_agents.q_functions.q_function import AbstractQFunction
from rl_agents.policies.policy import AbstractPolicy
from rl_agents.q_agents.value_agent import AbstractValueAgent

import torch
import numpy as np
import logging

class DQNAgent(
    torch.nn.Module,
    AbstractValueAgent,
):
    def __init__(
        self,
        nb_env: int,
        policy: AbstractPolicy,
        q_function : AbstractQFunction,
        train_every: int,
        replay_memory: AbstractReplayMemory,
        batch_size: int,
    ):

        torch.nn.Module.__init__(self)
        AbstractValueAgent.__init__(self, q_function=q_function, nb_env=nb_env, policy=policy)
        self.q_function = q_function.connect(self)
        self.replay_memory = replay_memory.connect(self)
        
        self.batch_size = batch_size
        self.train_every = train_every


    def sample(self):
        return self.replay_memory.sample(batch_size=self.batch_size)

    def store(self, **kwargs):
        assert self.training, "Cannot store any memory during eval."
        for key, value in kwargs.items():
            kwargs[key] = torch.as_tensor(value)
            if self.nb_env == 1: kwargs[key] = kwargs[key][None, ...] # Uniformize the shape, so first dim is always nb_env 
        self.replay_memory.store(agent=self, **kwargs)

    def train_agent(self) -> float:
        assert self.mode.value == Mode.TRAINING.value, "Please set the agent in training mode using .train()"
        
        # Training evert x steps
        if self.step % self.train_every == 0:
            with torch.no_grad():
                samples = self.replay_memory.sample(batch_size=self.batch_size, training= True)
            if samples is None: return None

            q_loss = self.q_function.train_q_function(**samples)
            return q_loss
        return None


    @property
    def mode(self):
        if self.training: return Mode.TRAINING
        return Mode.EVAL
