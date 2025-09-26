from rl_agents.agent import BaseAgent
from rl_agents.trainers.agent import BaseAgentTrainer
from rl_agents.memory.sampler import Sampler, RandomSampler
from rl_agents.memory.memory import Memory, Experience
from rl_agents.memory.rollout_memory import RolloutMemory
from rl_agents.memory.replay_memory import ReplayMemory
from rl_agents.critics.advantage_function import BaseAdvantageFunction
from rl_agents.utils.hidden_modules import HiddenModulesUtilsMixin
from rl_agents.utils.assert_check import assert_is_instance

from abc import ABCMeta, abstractmethod
import torch
import numpy as np

class OnPolicyTrainerMixin(
        HiddenModulesUtilsMixin,
        BaseAgentTrainer,
        metaclass=ABCMeta
    ):
    def __init__(self,
            rollout_period : int,
            epoch_per_rollout : int,
            batch_size : int,
            **kwargs):
    
        super().__init__(**kwargs)
        self.rollout_period = rollout_period
        self.epoch_per_rollout = epoch_per_rollout
        self.batch_size = batch_size

    @property
    def rollout_memory(self) -> ReplayMemory: 
        """Hidden reference to avoid infinite recursion with torch recursive method (like .to(), .train() ...)"""
        return self.hidden.rollout_memory

    @property
    def advantage_function(self) -> BaseAdvantageFunction: 
        """Hidden reference to avoid infinite recursion with torch recursive method (like .to(), .train() ...)"""
        return self.hidden.advantage_function

    def set_up_and_check(self, agent : "BaseAgent"):
        super().set_up_and_check(agent)
        assert agent.memory is None, "PPOTrainer does not require any agent.memory"
        self.agent.memory = RolloutMemory(
            max_length=self.rollout_period * agent.nb_env,
            observation_space=agent.observation_space,
            action_space=agent.action_space
        )
        self.hidden.rollout_memory = self.agent.memory

        assert_is_instance(self.agent.critic, BaseAdvantageFunction)
        self.hidden.advantage_function = self.agent.critic
    
    
    @abstractmethod
    def train_step(self, experience : Experience): ...

    def train_agent(self):
        mean_loss = None
        if self.agent.nb_step % self.rollout_period == 0:
            # 1 - Precomputations
            self.advantage_function.compute(trainer = self)

            losses = []
            
            for i in range(self.epoch_per_rollout):
                perm = torch.randperm(len(self.rollout_memory))
                for start in range(0, len(self.rollout_memory), self.batch_size):
                    idx = perm[start:start + self.batch_size]
                    experience = self.rollout_memory[idx]

                    losses.append(self.train_step(experience = experience))                                        
            
            # 3 - After training
            mean_loss = np.mean(losses)
            self.rollout_memory.reset()            
        return mean_loss