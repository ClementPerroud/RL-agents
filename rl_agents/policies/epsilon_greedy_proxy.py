from rl_agents.agent import AbstractAgent, Mode
from rl_agents.policies.policy import AbstractPolicy
import numpy as np
import torch
from abc import ABC, abstractmethod
from gymnasium.spaces import Space
import math

class BaseEspilonGreedyPolicy(AbstractPolicy, ABC):
    def __init__(self, policy : AbstractPolicy, action_space: Space):
        self.policy = policy
        self.action_space = action_space
        self.epsilon = 1.0

    @abstractmethod
    def epsilon_function(self, agent: AbstractAgent): ...

    def pick_action(self, agent: AbstractAgent, state: torch.Tensor, training : bool):

        if not training:
            return self.policy.pick_action(state)
        
        # state : (nb_env, ...)
        rands = torch.rand(agent.nb_env)  # shape : (nb_env,)
        env_random_action = rands < self.epsilon
        env_model_action = ~env_random_action
        actions = torch.zeros(
            size=(agent.nb_env,) + self.action_space.shape
        ).long() # shape (nb_env, action_shape ...)
        if env_model_action.any():
            masked_state = state[
                env_model_action
            ]  # shape (nb_env_selected,  state_shape ...)
            model_actions = self.policy.pick_action(agent = agent, state = masked_state, training= True)
            actions[env_model_action] = model_actions.long()

        if env_random_action.any():
            nb_random = env_random_action.sum()
            random_actions = torch.Tensor(
                [self.action_space.sample() for i in range(nb_random)]
            ).long()
            actions[env_random_action] = random_actions

        if agent.nb_env == 1:
            actions = actions[0]
        return actions

    def update(self, agent: AbstractAgent):
        self.epsilon = self.epsilon_function(agent=agent)


class EspilonGreedyPolicy(BaseEspilonGreedyPolicy):
    def __init__(self, policy : AbstractPolicy, q: float, start_epsilon : float, end_epsilon: float, action_space: Space):
        super().__init__(policy= policy, action_space=action_space)
        self.q = q
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon

    def epsilon_function(self, agent):
        return max(self.end_epsilon, self.end_epsilon + (self.start_epsilon - self.end_epsilon) *self.q**agent.step)
