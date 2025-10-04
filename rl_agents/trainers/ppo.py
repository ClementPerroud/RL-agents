from rl_agents.policies.policy import StochasticPolicy
from rl_agents.memory.memory import Experience
from rl_agents.actor_critic_agent import ActorCriticAgent
from rl_agents.trainers.mixin.on_policy import OnPolicyTrainerMixin
from rl_agents.utils.distribution.distribution import Distribution, distribution_aware, distribution_mode, debug_mode
from rl_agents.utils.check import assert_is_instance

import torch
import numpy as np
from copy import deepcopy
import gymnasium as gym
import itertools    
from typing import Union

class PPOTrainer(OnPolicyTrainerMixin):
    def __init__(self,
            epsilon : float, 
            entropy_loss_coeff : float,
            value_loss : torch.nn.modules.loss._Loss,
            value_loss_coeff : float,
            clip_value_loss : bool,
            max_grad_norm : float,
            rollout_period : int,
            epoch_per_rollout : int,
            batch_size : int,
            *args, **kwargs):
        super().__init__(rollout_period=rollout_period, epoch_per_rollout=epoch_per_rollout,batch_size=batch_size)
        self.entropy_loss_coeff = entropy_loss_coeff
        self.epsilon = float(epsilon)
        self.value_loss = value_loss
        self.value_loss_coeff = float(value_loss_coeff)
        self.clip_value_loss = bool(clip_value_loss)
        self.max_grad_norm = max_grad_norm
        
    
    def set_up_and_check(self, agent : "ActorCriticAgent"):
        super().set_up_and_check(agent)
        
        self.policy : Union[StochasticPolicy, torch.nn.Module] = assert_is_instance(agent.actor, StochasticPolicy)

        self.optimizer = torch.optim.Adam(
            params=agent.parameters(),
            lr = 3E-4
        )
    
    @distribution_aware
    def train_step(self, experience : Experience):
        self.optimizer.zero_grad()

        action_distributions = self.policy.action_distributions(state=experience.state)
        entropy_loss = self.policy.entropy_loss(*action_distributions).mean()

        ratio = torch.exp(
            self.policy.evaluate_log_prob(
                action_distributions=action_distributions,
                action = experience.action,
            )
            - experience.log_prob
        )
        # ratio : [batch]

        # 1 - Policy Clip
        advantage = experience.advantage
        if isinstance(advantage, Distribution): advantage = advantage.expectation()

        p1 = ratio * advantage
        p2 = torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        policy_loss = torch.minimum(p1, p2).mean()

        # 2 - Value Clip
        experience_return = experience.advantage + experience.value
        value = self.advantage_function.value_function.V(experience.state)
        value_loss_unclipped = self.value_loss(value, experience_return)

        # value_loss_unclippedloss.backward()
        if self.clip_value_loss:
            
            value_clipped= experience.value + torch.clamp(value - experience.value, min = -self.epsilon, max= self.epsilon)

            value_loss_clipped = self.value_loss(value_clipped, experience_return) # C51Loss
            value_loss= torch.max(value_loss_unclipped, value_loss_clipped).mean()

        else:
            value_loss = value_loss_unclipped.mean()

        # loss = value_loss
        loss : torch.Tensor =  - policy_loss + value_loss*self.value_loss_coeff - self.entropy_loss_coeff * entropy_loss

        loss.backward()

        if self.max_grad_norm is not None : torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()
