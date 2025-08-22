if __name__ == "__main__":
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 

from rl_agents.service import AgentService
from rl_agents.trainers.trainer import Trainer

from rl_agents.policy_agents.ppo_agent import PPOAgent, PPOLoss
from rl_agents.policies.deep_policy import DiscreteDeepPolicy
from rl_agents.value_functions.dqn_function import DVNFunction
from rl_agents.policy_agents.advantage_function import GAEFunction

import torch
import numpy as np
import gymnasium as gym
from collections import deque


class MainNet(AgentService):
    def __init__(self, observation_space : gym.spaces.Space, hidden_dim :int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We suppose observation_space and action_space to be 1D
        self.module_list = torch.nn.ModuleList()
        for i in range(2):
            self.module_list.add_module(f"lin_{i}", torch.nn.Linear(in_features= observation_space.shape[0] if i ==0 else hidden_dim, out_features= hidden_dim))
            self.module_list.add_module(f"act_{i}", torch.nn.ReLU())
    
    def forward(self, state : torch.Tensor, **kwargs) -> torch.Tensor:
        # state : [batch, nb_obs]
        x = state
        for module in self.module_list:
            x = module(x)
        
        # output : [batch, hidden_dim]
        return x
class SequentialNet(torch.nn.Sequential, AgentService):
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

def main():
    nb_env = 1
    gamma = 0.99
    lambda_ = 0.95
    hidden_dim = 64
    epsilon = 0.1

    env = gym.make("LunarLander-v3")

    action_space = env.action_space # 0 : short , 1 : long
    observation_space = env.observation_space # open, high, low, close, volume

    main_net = MainNet(observation_space=observation_space, hidden_dim=hidden_dim)
    policy_net = SequentialNet(
        main_net,
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, action_space.n),
        torch.nn.Softmax(dim = -1)
    )
    value_net = SequentialNet(
        main_net,
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1)
    )
    policy = DiscreteDeepPolicy(
        policy_net= policy_net
    )
    value_function = DVNFunction(net = value_net, gamma= gamma, trainer= Trainer(loss_fn=torch.nn.MSELoss()))
    advantage_function = GAEFunction(value_function=value_function, gamma=gamma, lambda_=lambda_)
    policy_loss = PPOLoss(epsilon=epsilon, entropy_loss_coeff=0.01)
    agent = PPOAgent(
        nb_env=nb_env,
        policy=policy,
        value_function=value_function,
        advantage_function=advantage_function,
        policy_loss=policy_loss,
        rollout_period= 2048,
        epoch_per_rollout=4,
        batch_size=64,
        observation_space=observation_space,
        action_space=action_space,
        values_loss_coeff= 0.5
    )


    episodes = 10000
    for i in range(episodes):
        episode_rewards = 0
        episode_losses = []
        episode_steps = 0

        truncated = False
        done = False
        state, infos = env.reset()
        
        while not truncated and not done:
            action = agent.pick_action(state= state)
            next_state, reward, done, truncated, infos = env.step(action = int(action))
            episode_rewards += reward
            # print(state, action, reward, next_state, done, truncated)
            done = done or truncated

            agent.store(state = state, action = action, reward = reward, next_state = next_state, done = done, truncated = truncated)
            loss = agent.train_agent()

            if loss is not None: episode_losses.append(loss)
            
            episode_steps += 1
            state = next_state

        episode_loss = np.array(episode_losses).mean() if len(episode_losses) > 0 else 0
        print(f"Episode {i:3d} - Steps : {episode_steps:4d} | Total Rewards : {episode_rewards:7.2f} | Loss : {episode_loss:0.2e} | Agent Step : {agent.step}")
        # print(episode_losses)

        # if episode_rewards >= 500:
        #     agent.plot_atoms_distributions()


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()