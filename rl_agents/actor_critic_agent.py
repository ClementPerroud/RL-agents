from rl_agents.agent import BaseAgent
from rl_agents.trainers.agent import AgentTrainer
from rl_agents.memory.memory import Memory

import torch
import gymnasium as gym


class ActorCriticAgent(BaseAgent):
    """Advantage Actor-Critic Agent"""
    def __init__(self,
            nb_env : int,
            trainer : AgentTrainer,
            observation_space : gym.spaces.Space,
            action_space : gym.spaces.Space,
            actor = None,
            critic = None,
            memory : Memory = None,

            **kwargs
        ):
        super().__init__(nb_env=nb_env, policy=actor, **kwargs)
        self.actor = actor
        self.critic = critic
        self.memory = memory
        self.trainer = trainer

        self.observation_space = observation_space
        self.action_space = action_space

        self.trainer.set_up_and_check(self)


    def train_agent(self):
        super().train_agent()
        return self.trainer.train_agent()
 

    def store(self, **kwargs):
        if not self.training: raise ValueError("Please set the agent to train mode using agent.train()")
        # Store experience
        for key, value in kwargs.items():
            kwargs[key] = torch.as_tensor(value)
            if kwargs[key].isnan().any(): raise ValueError(f"Detected nan in {key} : {kwargs[key]}")
            if self.nb_env == 1: kwargs[key] = kwargs[key][None, ...] # Uniformize the shape, so first dim is always nb_env 
        # Adding log_prob
        self.memory.store(**kwargs)
        # Step over sub modules
        self.step(**kwargs)
