from rl_agents.agent import AgentService
from rl_agents.action_model.model import ActionModel
import torch
import numpy as np
from abc import ABC, abstractmethod
from gymnasium.spaces import Space, Discrete, Box


class BaseEspilonGreedyProxy(ActionModel, ABC):
    def __init__(self, action_model : ActionModel, nb_env : int, action_space : Space, q = 1 - 5E-5):
        self.action_model = action_model.connect(self)
        self.epsilon = 1
        self.nb_env = nb_env
        self.action_space = action_space
    
    @abstractmethod
    def epsilon_function(self):
        ...
    
    def pick_action(self, states : torch.Tensor):

        # states : (nb_env, ...)
        rands = torch.rand(self.nb_env) # shape : (nb_env,)
        env_random_action = rands < self.epsilon
        env_model_action = ~env_random_action

        actions = torch.zeros(size = (self.nb_env,) + self.action_space.shape) # shape (nb_env, action_shape ...)

        if env_model_action.any():
            masked_states = states[env_model_action] # shape (nb_env_selected,  states_shape ...)
            model_actions = self.action_model.pick_action(masked_states)
            actions[env_model_action] = model_actions
        
        if env_model_action.any():
            nb_random = env_random_action.sum()
            random_actions = torch.Tensor([self.action_space.sample() for i in range(nb_random)])
            actions[env_random_action] = random_actions
        
        random_actions = self.action_space.sample()
        self.action_space.shape
        return actions

    def update(self, infos):
        self.epsilon = self.epsilon_function()


class EspilonGreedyProxy(BaseEspilonGreedyProxy):
    def __init__(self, q = 1 - 1E-5, *args, **kwargs):
        self.q = q
        self.steps = 0
        super().__init__(*args, **kwargs)
    
    def epsilon_function(self):
        self.steps += 1
        return self.q ** self.steps

if __name__ == "__main__":
    from rl_agents.dqn import DQNgent
    from rl_agents.action_model.model import OTPActionModel

    nb_env = 100
    action_space = Discrete(4)
    states_space = Box(low = 0, high = 1, shape = (nb_env, 4, 6))
    model  = OTPActionModel(nb_env=4, action= -1)

    proxy_model = EspilonGreedyProxy(action_model= model, nb_env=nb_env, action_space= action_space)
    proxy_model.epsilon = 0.9
    print(model.pick_action(states_space.sample()))
    print(proxy_model.pick_action(states_space.sample()))
    
    print("Hello world !")