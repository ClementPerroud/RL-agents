from rl_agents.agent import BaseAgent
from rl_agents.service import AgentService
from rl_agents.policies.policy import Policy, DiscretePolicy
from rl_agents.utils.assert_check import assert_is_instance
from rl_agents.utils.wrapper import Wrapper

import numpy as np
import torch
from abc import ABC, abstractmethod
from gymnasium.spaces import Space
import math

class BaseEspilonGreedyPolicyWrapper(Wrapper, Policy, AgentService, ABC):
    def __init__(self, policy : Policy, action_space: Space, epsilon = 1.0):
        super().__init__()
        self.policy = assert_is_instance(policy, Policy)
        self.action_space = action_space
        self.epsilon = epsilon

    @abstractmethod
    def epsilon_function(self, agent: BaseAgent): ...

    def pick_action(self, state: torch.Tensor, **kwargs):
        
        nb_env = state.size(0)
        if not self.training:
            return self.policy.pick_action(state = state, **kwargs)
        
        # state : (nb_env, ...)
        rands = torch.rand(nb_env, device= state.device)  # shape : (nb_env,)
        env_random_action = rands < self.epsilon
        env_model_action = ~env_random_action
        actions = torch.zeros(
            size=(nb_env,) + self.action_space.shape, device= state.device, dtype=torch.long
        ) # shape (nb_env, action_shape ...)

        if env_model_action.any():
            masked_state = state[
                env_model_action
            ]  # shape (nb_env_selected,  state_shape ...)
            model_actions = self.policy.pick_action(state = masked_state, **kwargs)
            actions[env_model_action] = model_actions.long()

        if env_random_action.any():
            nb_random = int(env_random_action.sum())
            random_actions = torch.Tensor(
                [self.action_space.sample() for i in range(nb_random)], device= state.device
            ).long()
            actions[env_random_action] = random_actions

        return actions

    def update(self, agent: BaseAgent, **kwargs):
        self.epsilon = self.epsilon_function(agent=agent)


class EspilonGreedyPolicyWrapper(BaseEspilonGreedyPolicyWrapper):
    def __init__(self, policy : Policy, epsilon_decay: int, action_space: Space, end_epsilon: float, start_epsilon : float = 1):
        super().__init__(policy= policy, action_space=action_space, epsilon= start_epsilon)
        self.epsilon_decay = epsilon_decay
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
    
    def epsilon_function(self, agent):
        return self.end_epsilon + (self.start_epsilon - self.end_epsilon) * math.exp(- agent.nb_step / self.epsilon_decay)
