if __name__ == "__main__":
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 

from rl_agents.service import AgentService
from rl_agents.value_functions.value_manager import  DoubleQWrapper, SoftDoubleQWrapper
from rl_agents.policies.value_policy import ValuePolicy
from rl_agents.policies.epsilon_greedy import EspilonGreedyPolicy
from rl_agents.replay_memory.replay_memory import ReplayMemory, MultiStepReplayMemory
from rl_agents.replay_memory.sampler import PrioritizedReplaySampler, RandomSampler
from rl_agents.value_agents.dqn import DQNAgent
from rl_agents.value_agents.noisy_net_strategy import NoisyNetProxy
from rl_agents.value_functions.c51_dqn_function import C51DQN, C51Loss


import torch
import numpy as np
import gymnasium as gym
from collections import deque


class C51QNN(AgentService):
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
        max_length = memory_size,
        nb_env= nb_env,
        gamma = gamma,
        multi_step= multi_step,
        observation_space= observation_space,
    )
    sampler= RandomSampler(replay_memory=replay_memory)


    q_net = C51QNN(nb_atoms= nb_atoms, observation_space=observation_space, action_space= action_space, hidden_dim= 128)
    q_net = SoftDoubleQWrapper(
        q_net = q_net,
        tau= 20
    )
    q_function = C51DQN(
        nb_atoms=nb_atoms,
        v_min=v_min, v_max=v_max,
        net=q_net,
        loss_fn = C51Loss(),
        gamma= gamma,
        multi_steps= multi_step,
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
        q_function=q_function,
        optimizer=torch.optim.AdamW(q_net.parameters(), lr= 1E-3, amsgrad= True),
        replay_memory= replay_memory,
        sampler=sampler,
        batch_size= 128,
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

        agent.store(state = state, action = action, reward = reward, next_state = next_state, done = done, truncated=truncated)
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