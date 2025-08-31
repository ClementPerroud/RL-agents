from rl_agents.agent import AbstractAgent
from rl_agents.service import AgentService
from rl_agents.policy_agents.policy_agent import AbstractPolicyAgent
from rl_agents.policies.policy import AbstractPolicy
from rl_agents.replay_memory.sampler import RandomSampler
from rl_agents.value_functions.value import V
from rl_agents.policies.stochastic_policy import AbstractStochasticPolicy
from rl_agents.replay_memory.rollout_memory import RolloutMemory
from rl_agents.policy_agents.advantage_function import BaseAdvantageFunction
from rl_agents.utils.collates import do_nothing_collate

import torch
import numpy as np
from copy import deepcopy
import gymnasium as gym
import itertools    

class PPOLoss(AgentService):
    def __init__(self,
            epsilon : float, 
            entropy_loss_coeff : float,
            values_loss_coeff : float,
            clip_value_loss : bool,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_loss_coeff = entropy_loss_coeff
        self.epsilon = epsilon
        self.values_loss_coeff = values_loss_coeff
        self.clip_value_loss = clip_value_loss
    
    def forward(self, agent : "A2CAgent", policy : AbstractStochasticPolicy, experience):
        # advantage : [batch]

        action_distributions = policy.action_distributions(state=experience.state)
        entropy_loss = policy.entropy_loss(*action_distributions).mean()

        ratio = torch.exp(
            policy.evaluate_log_prob(
                action_distributions=action_distributions,
                action = experience.action,
            )
            - experience.log_prob
        )
        # ratio : [batch]

        # Policy Clip
        p1 = ratio * experience.advantage
        p2 = torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * experience.advantage
        policy_loss = torch.minimum(p1, p2).mean()

        # Value Clip
        value = agent.advantage_function.value_function.V(experience.state)
        value_loss_unclipped = 0.5 * (experience._return - value).pow(2)

        if self.clip_value_loss:
            value_clipped = experience.value + torch.clamp(value - experience.value, min = -self.epsilon, max= self.epsilon)
            value_loss_clipped = 0.5 * (experience._return - value_clipped).pow(2)
            value_loss= torch.max(value_loss_unclipped, value_loss_clipped).mean()
        else:
            value_loss = value_loss_unclipped.mean()

        loss =  - policy_loss + value_loss*self.values_loss_coeff - self.entropy_loss_coeff * entropy_loss
        return loss

class A2CAgent[A, C, M](AbstractPolicyAgent):
    """Advantage Actor-Critic Agent"""
    def __init__(self,
            nb_env : int,
            actor : A,
            critic : C,
            memory : M,

            observation_space : gym.spaces.Space,
            action_space : gym.spaces.Space,
        ):
        super().__init__(nb_env = nb_env, policy = actor)
        self.actor = actor
        self.critic = critic
        self.memory = memory

        self.observation_space = observation_space
        self.action_space = action_space
        



    
    def store(self, **kwargs):
        assert self.training, "Cannot store any memory during eval. Please set your agent to TRAINING mode."
        
        for key, value in kwargs.items():
            kwargs[key] = torch.as_tensor(value)
            if self.nb_env == 1: kwargs[key] = kwargs[key][None, ...] # Uniformize the shape, so first dim is always nb_env 
        # Adding log_prob
        self.rollout_memory.store(**kwargs)


    def train_agent(self):
        super().train_agent()

        mean_loss = None
        if self.step % self.rollout_period == 0:
            # 1 - Precomputations
            self.advantage_function.compute(agent = self)

            losses = []
            
            for i in range(self.epoch_per_rollout):
                perm = torch.randperm(len(self.rollout_memory))
                for start in range(0, len(self.rollout_memory), self.batch_size):
                    idx = perm[start:start + self.batch_size]
                    experience = self.rollout_memory[idx]
                    
                    self.optimizer.zero_grad()

                    loss = self.policy_loss(agent = self, policy = self.policy, experience = experience)

                    loss.backward()
                    if self.max_grad_norm is not None : torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    losses.append(loss.item())
            
            # 3 - After training
            mean_loss = np.mean(losses)
            self.rollout_memory.reset()            
        return mean_loss
