from rl_agents.agent import AbstractAgent
from rl_agents.service import AgentService
from rl_agents.policy_agents.policy_agent import AbstractPolicyAgent
from rl_agents.policies.policy import AbstractPolicy
from rl_agents.memory.sampler import RandomSampler
from rl_agents.value_functions.value import V, Q, Trainable
from rl_agents.policies.stochastic_policy import StochasticPolicy
from rl_agents.memory.memory import Memory
from rl_agents.memory.replay_memory import ReplayMemory
from rl_agents.policy_agents.advantage_function import BaseAdvantageFunction
from rl_agents.utils.collates import do_nothing_collate

import torch
import numpy as np
from copy import deepcopy
import gymnasium as gym
import itertools    

class DDPGTrainer(AgentService):
    def __init__(self,
            train_every : int,
            batch_size : int,
            optimizer,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_every = train_every
        self.batch_size = batch_size
    
    def set_up_and_check(self, agent : "DDPG"):
        assert isinstance(agent.memory, Memory), "agent.memory must implements rl_agents.memory.Memory protocol"
        assert isinstance(agent.memory, ReplayMemory, "agent.memory must be or inherits from rl_agents.memory.ReplayMemory")

        # CRITIC CHECK : Check if the critic is a trainable Q function.
        assert isinstance(agent.critic, Q), "agent.critic must implements rl_agents.value_functions.value.Q"
        assert isinstance(agent.critic, Trainable), "agent.critic must implements rl_agents.value_functions.value.Trainable"

        # ACTOR CHECK : Check if 
        
    
    def train_agent(self, agent : "DDPG", policy : StochasticPolicy, experience):
        # advantage : [batch]
        if agent.step % self.train_every == 0 and agent.memory:
            pass
        return 0


class DDPG[A, C, M](AbstractPolicyAgent):
    """Advantage Actor-Critic Agent"""
    def __init__(self,
            nb_env : int,
            actor : A,
            critic : C,
            memory : M,
            trainer : DDPGTrainer,

            observation_space : gym.spaces.Space,
            action_space : gym.spaces.Space,
        ):
        super().__init__(nb_env = nb_env, policy = actor)
        self.actor = actor
        self.critic = critic
        self.memory = memory
        self.trainer = trainer

        self.observation_space = observation_space
        self.action_space = action_space
        self.trainer.set_up_and_check(self)
        

    def store(self, **kwargs):
        assert self.training, "Cannot store any memory during eval. Please set your agent to TRAINING mode."
        
        for key, value in kwargs.items():
            kwargs[key] = torch.as_tensor(value)
            if self.nb_env == 1: kwargs[key] = kwargs[key][None, ...] # Uniformize the shape, so first dim is always nb_env 
        # Adding log_prob
        self.rollout_memory.store(**kwargs)


    def train_agent(self):
        super().train_agent()
        loss = self.trainer.train_agent()

        return loss
