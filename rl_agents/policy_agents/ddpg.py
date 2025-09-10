from rl_agents.agent import Agent,BaseAgent
from rl_agents.service import AgentService
from rl_agents.policy_agents.policy_agent import AbstractPolicyAgent
from rl_agents.policies.policy import AbstractPolicy
from rl_agents.memory.sampler import Sampler, WeightableSampler, UpdatableSampler, RandomSampler
from rl_agents.value_functions.value import V, Q, Trainable
from rl_agents.policies.stochastic_policy import StochasticPolicy
from rl_agents.memory.memory import Memory, ExperienceSample
from rl_agents.memory.replay_memory import ReplayMemory
from rl_agents.policy_agents.advantage_function import BaseAdvantageFunction
from rl_agents.utils.collates import do_nothing_collate
from rl_agents.utils.mode import eval_mode

import torch
import numpy as np
from copy import deepcopy
import gymnasium as gym
from typing import Protocol, runtime_checkable, Union
from abc import ABC, abstractmethod

@runtime_checkable
class AgentTrainer(Protocol):
    def set_up_and_check(self, agent : BaseAgent):...
    def train_agent(self): ...

class BaseAgentTrainer(ABC, AgentService):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def set_up_and_check(self, agent : BaseAgent):
        assert isinstance(agent, Agent), "agent must implements rl_agents.agent.Agent protocol"
        assert isinstance(agent.memory, Memory), "agent.memory must implements rl_agents.memory.Memory protocol"
        self.agent : Agent = agent


class OffPolicyTrainerMixin(ABC, BaseAgentTrainer):
    def __init__(self, sampler : RandomSampler, train_every : int, batch_size : int, **kwargs): 
        super().__init__(**kwargs)
        self.train_every = train_every
        self.batch_size = batch_size
        self.sampler = sampler
    
    def set_up_and_check(self, agent):
        super().set_up_and_check(agent)
        assert isinstance(self.agent.memory, Memory), "agent.memory must implements rl_agents.memory.Memory protocol"
        assert isinstance(self.agent.memory, ReplayMemory), "agent.memory must be or inherits from rl_agents.memory.ReplayMemory"
        self.replay_memory : ReplayMemory = self.agent.memory

        assert isinstance(self.sampler, Sampler), "agent.memory must be or inherits from rl_agents.memory.ReplayMemory"
    
    def train_agent(self) -> float:
        # advantage : [batch]
        if self.agent.step % self.train_every == 0 and len(self.agent.memory) > self.batch_size:
            return self.train_step()
        return np.nan
    
    @abstractmethod
    def train_step(self): ...

class VCriticTrainerMixin(BaseAgentTrainer):
    def set_up_and_check(self, agent):
        super().set_up_and_check(agent)
        # CRITIC CHECK : Check if the critic is a trainable Q function.
        assert isinstance(self.agent, ActorCriticAgent), "agent must inherits from ActorCriticAgent"
        assert isinstance(self.agent.critic, V), "agent.critic must implements rl_agents.value_functions.value.V"
        assert isinstance(self.agent.critic, Trainable), "agent.critic must implements rl_agents.value_functions.value.Trainable"
        self.value_function : Union[V, Trainable] = self.agent.critic

class QCriticTrainerMixin(BaseAgentTrainer):
    def set_up_and_check(self, agent):
        super().set_up_and_check(agent)
        # CRITIC CHECK : Check if the critic is a trainable Q function.
        assert isinstance(self.agent, ActorCriticAgent), "agent must inherits from ActorCriticAgent"
        assert isinstance(self.agent.critic, Q), "agent.critic must implements rl_agents.value_functions.value.Q"
        assert isinstance(self.agent.critic, Trainable), "agent.critic must implements rl_agents.value_functions.value.Trainable"
        self.q_function : Union[Q, Trainable] = self.agent.critic

    
class DQNTrainer(OffPolicyTrainerMixin, QCriticTrainerMixin, BaseAdvantageFunction):
    def __init__(self, loss_fn : torch.nn.modules.loss._Loss, optimizer : torch.optim.Optimizer, **kwargs):
        super().__init__(**kwargs)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_fn.reduction = "none"
    
    def compute_loss(self, experience : ExperienceSample) -> torch.Tensor:
        loss_input = self.q_function.compute_loss_input(experience=experience)
        with eval_mode(self), torch.no_grad():
            loss_target = self.q_function.compute_loss_target(experience=experience)

        loss : torch.Tensor = self.loss_fn(loss_input, loss_target)
        if isinstance(self.sampler, WeightableSampler): loss = self.sampler.apply_weights(loss=loss, indices=experience.indices)
        loss = loss.mean()

        if isinstance(self.sampler, UpdatableSampler):
            with torch.no_grad():
                self.sampler.update_experiences(
                    agent = self, indices = experience.indices, td_errors = self.q_function.compute_td_errors(loss_input=loss_input, loss_target=loss_target)
                )
        
        return loss

    def train_step(self):
        indices = self.sampler.sample(self.batch_size)
        experience = self.replay_memory[indices]

        self.optimizer.zero_grad()
        loss = self.compute_loss(experience=experience)
        loss.backward()

        self.optimizer.step()
        return loss.item()
        
class DDPGTrainer(AgentService):
    def __init__(self,
            train_every : int,
            batch_size : int,
            dqn_trainer : DQNTrainer,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_every = train_every
        self.batch_size = batch_size
        self.dqn_trainer = dqn_trainer
    
    def set_up_and_check(self, agent : "ActorCriticAgent"):
        assert isinstance(agent.memory, Memory), "agent.memory must implements rl_agents.memory.Memory protocol"
        assert isinstance(agent.memory, ReplayMemory, "agent.memory must be or inherits from rl_agents.memory.ReplayMemory")

        # CRITIC CHECK : Check if the critic is a trainable Q function.
        assert isinstance(agent.critic, Q), "agent.critic must implements rl_agents.value_functions.value.Q"
        assert isinstance(agent.critic, Trainable), "agent.critic must implements rl_agents.value_functions.value.Trainable"

    def train_step(self):






class ActorCriticAgent[A, C, M](AbstractPolicyAgent):
    """Advantage Actor-Critic Agent"""
    def __init__(self,
            nb_env : int,
            actor : A,
            critic : C,
            memory : M,
            trainer : AgentTrainer,

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
