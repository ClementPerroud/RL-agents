from rl_agents.agent import AbstractAgent
from rl_agents.action_strategy.action_strategy import AbstractActionStrategy
import torch
import numpy as np
from abc import ABC, abstractmethod
from gymnasium.spaces import Space, Discrete, Box
import functools

class BaseEspilonGreedyActionStrategy(AbstractActionStrategy, ABC):
    def __init__(self, action_space : Space):
        self.action_space = action_space
        self.epsilon = 1
    
    @abstractmethod
    def epsilon_function(self, agent : AbstractAgent):
        ...
    
    def pick_action(self, agent :AbstractAgent, state : np.ndarray):
        if agent.nb_env == 1: state = state[None,...]

        # state : (nb_env, ...)
        rands = np.random.rand(agent.nb_env) # shape : (nb_env,)
        env_random_action = rands < self.epsilon
        env_model_action = ~env_random_action
        actions = np.zeros(shape = (agent.nb_env,) + self.action_space.shape) # shape (nb_env, action_shape ...)

        if env_model_action.any():
            masked_state = state[env_model_action] # shape (nb_env_selected,  state_shape ...)
            model_actions = agent._pick_deterministic_action(masked_state)
            actions[env_model_action] = model_actions
        
        if env_random_action.any():
            nb_random = env_random_action.sum()
            random_actions = np.array([self.action_space.sample() for i in range(nb_random)])
            actions[env_random_action] = random_actions
        
        if agent.nb_env == 1: 
            actions = actions[0]
        return actions

    def update(self, agent : AbstractAgent):
        self.epsilon = self.epsilon_function(agent=agent)


class EspilonGreedyActionStrategy(BaseEspilonGreedyActionStrategy):
    def __init__(self, q : float, action_space : Space):
        super().__init__(action_space = action_space)
        self.q = q
    
    def epsilon_function(self, agent):
        return self.q ** agent.step