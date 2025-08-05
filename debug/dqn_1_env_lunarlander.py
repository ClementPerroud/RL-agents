from rl_agents.q_agents.deep_q_model import AbstractDeepQNeuralNetwork
from rl_agents.q_agents.double_q_net import  DoubleQNNProxy
from rl_agents.action_strategy.epsilon_greedy_strategy import EspilonGreedyActionStrategy
from rl_agents.replay_memory.replay_memory import ReplayMemory
from rl_agents.replay_memory.sampler import PrioritizedReplaySampler, RandomSampler
from rl_agents.q_agents.dqn import DQNAgent

import torch
import numpy as np
import gymnasium as gym


class QNN(AbstractDeepQNeuralNetwork):
    def __init__(self, observation_space : gym.spaces.Space, action_space : gym.spaces.Discrete, hidden_dim :int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We suppose observation_space and action_space to be 1D
        self.module_list = torch.nn.ModuleList()
        for i in range(4):
            self.module_list.add_module(f"lin_{i}", torch.nn.Linear(in_features= observation_space.shape[0] if i ==0 else hidden_dim, out_features= hidden_dim))
            self.module_list.add_module(f"act_{i}", torch.nn.ReLU())
        self.head = torch.nn.Linear(in_features=hidden_dim, out_features=action_space.n)

    
    def forward(self, state : torch.Tensor, **kwargs) -> torch.Tensor:
        # state : [batch, nb_obs]
        x = state
        for module in self.module_list:
            x = module(x)
        return self.head(x)

def main():
    # gym.make("LunarLander-v3")
    env = gym.make("LunarLander-v3")

    action_space = env.action_space # 0 : short , 1 : long
    observation_space = env.observation_space # open, high, low, close, volume

    nb_env = 1
    memory_size = 1E8
    replay_memory = ReplayMemory(
        length = memory_size, 
        sampler= PrioritizedReplaySampler(length= memory_size, alpha= 0.65, beta_0=0.5, duration=250_000),
        observation_space= observation_space)

    q_net=QNN(observation_space=observation_space, action_space= action_space, hidden_dim= 64)
    q_net = DoubleQNNProxy(
        q_net = q_net,
        tau= 1000
    )
    action_strategy = EspilonGreedyActionStrategy(
        q = 1 - 1E-4,
        min_epsilon= 0.01,
        action_space= action_space
    )
    agent = DQNAgent(
        nb_env= nb_env,
        action_strategy= action_strategy,
        n_steps= 3,
        train_every= 3,
        gamma=0.99,
        replay_memory= replay_memory,
        q_net = q_net,
        optimizer= torch.optim.Adam(q_net.parameters(), lr = 1E-3)
    )

    episodes = 1000
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

            agent.store(state = state, action = action, reward = reward, next_state = next_state, done = done)
            loss = agent.train_agent()

            if loss is not None: episode_losses.append(loss)
            episode_steps += 1
            state = next_state

        epsilon = action_strategy.epsilon
        episode_loss = np.array(episode_losses).mean()
        print(f"Episode {i:3d} - Steps : {episode_steps:4d} | Total Rewards : {episode_rewards:5.2f} | Loss : {episode_loss:0.2e} | Epsilon : {epsilon : 0.2f} | Agent Step : {agent.step}")


if __name__ == "__main__":
    main()