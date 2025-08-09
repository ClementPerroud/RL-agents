if __name__ == "__main__":
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 

from rl_agents.q_agents.deep_q_model import AbstractDeepQNeuralNetwork
from rl_agents.q_agents.double_q_net import  DoubleQNNProxy, SoftDoubleQNNProxy
from rl_agents.action_strategy.epsilon_greedy_strategy import EspilonGreedyActionStrategy
from rl_agents.replay_memory.replay_memory import ReplayMemory, MultiStepReplayMemory
from rl_agents.replay_memory.sampler import PrioritizedReplaySampler, RandomSampler
from rl_agents.q_agents.dqn import DQNAgent
from rl_agents.q_agents.categorical_dqn import CategoricalDQNAgent, CategoricalLoss

import torch
import numpy as np
import gymnasium as gym


class CategoricalQNN(AbstractDeepQNeuralNetwork):
    def __init__(self, nb_atoms : int, observation_space : gym.spaces.Space, action_space : gym.spaces.Discrete, hidden_dim :int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We suppose observation_space and action_space to be 1D
        self.nb_atoms = nb_atoms
        self.nb_actions = action_space.n
        self.module_list = torch.nn.ModuleList()
        for i in range(2):
            self.module_list.add_module(f"lin_{i}", torch.nn.Linear(in_features= observation_space.shape[0] if i ==0 else hidden_dim, out_features= hidden_dim))
            self.module_list.add_module(f"act_{i}", torch.nn.ReLU())
        self.head = torch.nn.Linear(in_features=hidden_dim, out_features=self.nb_actions * self.nb_atoms)
    
    def forward(self, state : torch.Tensor, **kwargs) -> torch.Tensor:
        # state : [batch, nb_obs]
        x = state
        for module in self.module_list:
            x = module(x)
        return torch.reshape(self.head(x), shape = (-1, self.nb_actions, self.nb_atoms))

def main():
    # gym.make("LunarLander-v3")
    env = gym.make("LunarLander-v3")

    action_space = env.action_space # 0 : short , 1 : long
    observation_space = env.observation_space # open, high, low, close, volume

    nb_env = 1
    memory_size = 300_000
    nb_atoms = 51
    gamma = 0.99
    v_min, v_max = -250, 300
    replay_memory = MultiStepReplayMemory(
        length = memory_size,
        nb_env= nb_env,
        gamma = gamma,
        multi_step= 3,
        # sampler= RandomSampler(),
        sampler= PrioritizedReplaySampler(length=memory_size, duration= 150_000),
        observation_space= observation_space)

    q_net_generator = lambda : CategoricalQNN(nb_atoms= nb_atoms, observation_space=observation_space, action_space= action_space, hidden_dim= 128)
    q_net = SoftDoubleQNNProxy(
        q_net_generator = q_net_generator,
        tau= 20
    )
    action_strategy = EspilonGreedyActionStrategy(
        q = 1 - 1E-4,
        start_epsilon= 0.9,
        end_epsilon= 0.01,
        action_space= action_space
    )
    agent = CategoricalDQNAgent(
        nb_env= nb_env,
        nb_atoms= nb_atoms, v_min=v_min, v_max=v_max,
        action_strategy= action_strategy,
        train_every= 3,
        gamma=gamma,
        replay_memory= replay_memory,
        q_net = q_net,
        loss_fn=CategoricalLoss(),
        optimizer= torch.optim.AdamW(q_net.parameters(), lr = 1E-3, amsgrad=True),
        batch_size= 128,
        device = "cpu"
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
            done = done or truncated

            agent.store(state = state, action = action, reward = reward, next_state = next_state, done = done)
            loss = agent.train_agent()

            if loss is not None: episode_losses.append(loss)
            
            episode_steps += 1
            state = next_state

        epsilon = action_strategy.epsilon
        episode_loss = np.array(episode_losses).mean()
        print(f"Episode {i:3d} - Steps : {episode_steps:4d} | Total Rewards : {episode_rewards:7.2f} | Loss : {episode_loss:0.2e} | Epsilon : {epsilon : 0.2f} | Agent Step : {agent.step}")
        # print(episode_losses)

        if i >= 150:
            agent.plot_atoms_distributions()

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()