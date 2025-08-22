from rl_agents.agent import AbstractAgent
from rl_agents.service import AgentService
from rl_agents.policy_agents.policy_agent import AbstractPolicyAgent
from rl_agents.policies.policy import AbstractPolicy
from rl_agents.value_functions.v_function import AbstractVFunction
from rl_agents.policies.deep_policy import AbstractDeepPolicy
from rl_agents.replay_memory.rollout_memory import RolloutMemory
from rl_agents.replay_memory.policy_training_memory import PPOTrainingMemory
from rl_agents.policy_agents.advantage_function import BaseAdvantageFunction

import torch
import numpy as np
from copy import deepcopy
import gymnasium as gym
import itertools    

class PPOLoss(AgentService):
    def __init__(self,
            epsilon : float, 
            entropy_loss_coeff : float,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_loss_coeff = entropy_loss_coeff
        self.epsilon = epsilon
    
    def forward(self, agent : AbstractAgent, policy : AbstractDeepPolicy, state, action, old_action_log_likelihood, advantage, **kwargs):
        # advantage : [batch]

        action_distributions = policy.action_distributions(agent=agent, state=state)
        ratio = torch.exp(
            policy.evaluate_action_log_likelihood(
                agent=agent,
                action_distributions=action_distributions,
                action = action,
            )
            - old_action_log_likelihood
        )
        # ratio : [batch]
        entropy_loss = policy.entropy_loss(*action_distributions)
        p1 = ratio * advantage
        p2 = torch.clip(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
        loss =  - ( torch.minimum(p1, p2).mean() + self.entropy_loss_coeff * entropy_loss)
        return loss

class PPOAgent(AbstractPolicyAgent):
    
    def __init__(self,
            nb_env : int,
            policy : AbstractDeepPolicy,
            value_function : AbstractVFunction,
            advantage_function : BaseAdvantageFunction,
            policy_loss : PPOLoss,
            rollout_period : int,
            epoch_per_rollout : int,
            batch_size,
            observation_space : gym.spaces.Space,
            action_space : gym.spaces.Space,
            values_loss_coeff : float,

        ):
        super().__init__(nb_env = nb_env, policy = policy)
        assert isinstance(self.policy, AbstractDeepPolicy), "policy must be a DeepPolicy."
        self.policy : AbstractDeepPolicy

        self.value_function = value_function
        self.advantage_function = advantage_function

        self.rollout_period = rollout_period
        self.epoch_per_rollout = epoch_per_rollout
        self.rollout_memory = RolloutMemory(max_length=rollout_period * 2, observation_space=observation_space, action_space=action_space)

        self.policy_loss = policy_loss

        self.batch_size = batch_size

        self._opt_parameters = itertools.chain(self.policy.parameters(), self.value_function.parameters())
        self.optimizer = torch.optim.Adam(
            params=self._opt_parameters,
            lr = 3E-4,
            eps=1E-5
        )

        self.training_memory = PPOTrainingMemory(max_length = rollout_period * 2, observation_space= observation_space, action_space=action_space)

        self.values_loss_coeff = values_loss_coeff

    
    def store(self, **kwargs):
        assert self.training, "Cannot store any memory during eval. Please set your agent to TRAINING mode."
        
        for key, value in kwargs.items():
            kwargs[key] = torch.as_tensor(value)
            if self.nb_env == 1: kwargs[key] = kwargs[key][None, ...] # Uniformize the shape, so first dim is always nb_env 
        # Adding old_action_log_likelihood
        kwargs["old_action_log_likelihood"] = self.policy.last_log_prob.detach()
        self.rollout_memory.store(agent=self, **kwargs)


    def train_agent(self):
        super().train_agent()

        mean_loss = None
        if self.step % self.rollout_period == 0:
            # Training
            # First create the training dataset with rollout memory
            self.training_memory.reset()
            self.advantage_function.reset(agent= self)
            
            with torch.no_grad():
                for i in range(len(self.rollout_memory)-self.nb_env, -1, -self.nb_env):

                    index = torch.arange(start=i, end=i+self.nb_env)

                    
                    state, action, next_state, reward, done, truncated, old_action_log_likelihood = self.rollout_memory[index].values() # TODO : Sample it from RolloutMemory
                    # state : [batch, state_shape], action : [batch, action_shape]

                    advantage = self.advantage_function.compute_advantage(
                        state=state, action = action, reward = reward, next_state=next_state, done=done, truncated = truncated
                    )
                    self.training_memory.store(
                        agent=self,
                        state=state, 
                        action=action, 
                        reward=reward, 
                        next_state=next_state,
                        done=done, 
                        truncated=truncated, 
                        old_action_log_likelihood=old_action_log_likelihood, 
                        advantage = advantage
                    )
            self.rollout_memory.reset()
            


            # 2 - Start the training

            # Normalize advantage
            advantages = self.training_memory["advantage"]
            self.training_memory["advantage"] = (advantages - torch.mean(advantages))/(torch.std(advantages) + 1E-8)

            data_loader = torch.utils.data.DataLoader(self.training_memory,  batch_size=self.batch_size, shuffle=True, drop_last=False)
            losses = []
            for i in range(self.epoch_per_rollout):
                for data_dict in data_loader:
                    state = data_dict["state"]
                    action = data_dict["action"]
                    reward = data_dict["reward"]
                    next_state = data_dict["next_state"]
                    done = data_dict["done"]
                    truncated = data_dict["truncated"]
                    old_action_log_likelihood = data_dict["old_action_log_likelihood"]
                    advantage = data_dict["advantage"]

                    self.optimizer.zero_grad()
                    policy_loss = self.policy_loss(
                        agent = self,
                        policy = self.policy,
                        state=state, 
                        action=action, 
                        reward=reward, 
                        next_state=next_state,
                        done=done, 
                        truncated=truncated, 
                        old_action_log_likelihood=old_action_log_likelihood, 
                        advantage = advantage
                    )
                    value_loss = self.values_loss_coeff * self.value_function.trainer.loss_fn(*self.value_function.compute_loss_inputs(state=state, action=action, reward=reward, next_state=next_state, done=done, truncated=truncated)).mean()
                    
                    loss = policy_loss + value_loss

                    loss.backward()
                    # torch.nn.utils.clip_grad_value_(parameters=self.policy.policy_net.parameters(), clip_value= 1.)
                    # torch.nn.utils.clip_grad_value_(parameters=self.value_function.net.parameters(), clip_value= 1.)
                    self.optimizer.step()
                    losses.append(loss.item())
            
            # 3 - After training
            mean_loss = np.mean(losses)
            
        return mean_loss


if __name__ == "__main__":
    agent = PPOAgent(1, None)
