if __name__ == "__main__":
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 

from rl_agents.q_agents.deep_q_model import AbstractDeepQNeuralNetwork
from rl_agents.q_agents.double_q_net import  DoubleQNNProxy, SoftDoubleQNNProxy
from rl_agents.policies.value_policy import ValuePolicy
from rl_agents.policies.epsilon_greedy_proxy import EspilonGreedyPolicy
from rl_agents.replay_memory.replay_memory import ReplayMemory, MultiStepReplayMemory
from rl_agents.replay_memory.sampler import PrioritizedReplaySampler, RandomSampler
from rl_agents.q_agents.dqn import DQNAgent
from rl_agents.q_agents.noisy_net_strategy import NoisyNetProxy
from rl_agents.q_functions.distributional_dqn_function import DistributionalDQNFunction, CategoricalLoss

import torch
import numpy as np
import gymnasium as gym
from collections import deque


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
        
        # output : [batch, nb_actions, nb_atoms]
        return torch.reshape(self.head(x), shape = (-1, self.nb_actions, self.nb_atoms))

def main():
    nb_env = 3
    memory_size = 300_000
    nb_atoms = 51
    gamma = 0.99
    v_min, v_max = -250, 300
    multi_step = 3

    single_env = gym.make("LunarLander-v3")
    env = gym.make_vec("LunarLander-v3", num_envs= nb_env, vectorization_mode="sync")

    action_space = single_env.action_space # 0 : short , 1 : long
    observation_space = single_env.observation_space # open, high, low, close, volume

    replay_memory = MultiStepReplayMemory(
        length = memory_size,
        nb_env= nb_env,
        gamma = gamma,
        multi_step= multi_step,
        # sampler= RandomSampler(),
        sampler= PrioritizedReplaySampler(length=memory_size, duration= 150_000),
        observation_space= observation_space)


    q_net = CategoricalQNN(nb_atoms= nb_atoms, observation_space=observation_space, action_space= action_space, hidden_dim= 128)
    q_net = SoftDoubleQNNProxy(
        q_net = q_net,
        tau= 20
    )
    q_function = DistributionalDQNFunction(
        nb_atoms=nb_atoms,
        v_min=v_min, v_max=v_max,
        q_net=q_net,
        optimizer=torch.optim.AdamW(q_net.parameters(), lr= 1E-3, amsgrad= True),
        loss_fn = CategoricalLoss(),
        gamma= gamma ** multi_step
    )
    # q_net = NoisyNetProxy(q_net=q_net, std_init= 0.2)

    policy = EspilonGreedyPolicy(
        q = (1 - 1E-4)**(3),
        start_epsilon= 0.9,
        end_epsilon= 0.01,
        action_space= action_space,
        policy= ValuePolicy(q_function=q_function)
    )
    agent = DQNAgent(
        nb_env= nb_env,
        policy= policy,
        train_every= 1,
        replay_memory= replay_memory,
        batch_size= 128,
        q_function=q_function
    )

    episode_rewards = np.zeros(shape=(nb_env,))
    episode_losses = deque(maxlen= 200)
    episode_steps = np.zeros(shape=(nb_env,), dtype=int)
    episodes = 0

    state, infos = env.reset()
    
    while episodes < 20000:
        action = agent.pick_action(state= state)
        next_state, reward, done, truncated, infos = env.step(actions = action.astype(int))
        # print(state, action, reward, next_state, done, truncated)
        done = done | truncated

        agent.store(state = state, action = action, reward = reward, next_state = next_state, done = done)
        loss = agent.train_agent()

        if loss is not None: episode_losses.append(loss)
        
        state = next_state
        episode_rewards += reward
        episode_steps += 1

        reseted_env = []
        for i in range(nb_env):
            if done[i]:
                episodes += 1
                episode_loss = np.array(episode_losses).mean()
                print(f"Episode {episodes:3d} - Steps : {episode_steps[i]:4d} | Total Rewards : {episode_rewards[i]:7.2f} | Loss : {episode_loss:0.2e} | Epsilon : {policy.epsilon : 0.2f} | Agent Step : {agent.step}")
                episode_steps[i] = 0
                episode_rewards[i] = 0
    # print(episode_losses)


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()