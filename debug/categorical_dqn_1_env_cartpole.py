if __name__ == "__main__":
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 

from rl_agents.service import AgentService
from rl_agents.value_functions.value_manager import  SoftDoubleVManager
from rl_agents.policies.epsilon_greedy import EspilonGreedyPolicy
from rl_agents.replay_memory.replay_memory import ReplayMemory, MultiStepReplayMemory
from rl_agents.replay_memory.sampler import PrioritizedReplaySampler, RandomSampler
from rl_agents.value_agents.noisy_net_strategy import NoisyNetTransformer
from rl_agents.value_agents.dqn import DQNAgent
from rl_agents.value_functions.c51_dqn_function import C51DQN, C51Loss, DiscreteC51QWrapper
from rl_agents.policies.value_policy import DiscreteBestQValuePolicy

import torch
import numpy as np
import gymnasium as gym


def main():
    env = gym.make("LunarLander-v3")

    action_space = env.action_space # 0 : short , 1 : long
    observation_space = env.observation_space # open, high, low, close, volume

    NB_ENV = 1
    MEMORY_SIZE = 300_000
    NB_ATOMS = 51
    V_MIN, V_MAX = -250, 300
    GAMMA = 0.99
    HIDDEN_DIM = 128
    MULTI_STEP = 3

    replay_memory = MultiStepReplayMemory(
        max_length = MEMORY_SIZE,
        nb_env=NB_ENV,
        gamma=GAMMA,
        multi_step=MULTI_STEP,
        # sampler= PrioritizedReplaySampler(length=memory_size, duration= 20_000),
        observation_space= observation_space)

    core_net = torch.nn.Sequential(
        torch.nn.LazyLinear(HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.LazyLinear(HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.LazyLinear(HIDDEN_DIM), torch.nn.ReLU()
    )
    q_net = DiscreteC51QWrapper(core_net=core_net, action_space=action_space, nb_atoms=NB_ATOMS, v_min=V_MIN, v_max=V_MAX)
    q_net = NoisyNetTransformer(std_init=0.1)(q_net)
    q_manager = SoftDoubleVManager(
        tau= 50
    )
    q_function = C51DQN(
        nb_atoms= NB_ATOMS, v_min=V_MIN, v_max=V_MAX,
        net=q_net,
        manager=q_manager,
        loss_fn= C51Loss(),
        gamma = GAMMA
        )

    policy= DiscreteBestQValuePolicy(q=q_function)

    agent = DQNAgent(
        nb_env= NB_ENV,
        policy= policy,
        q_function= q_function,
        train_every= 3,
        replay_memory=replay_memory,
        sampler= RandomSampler(replay_memory=replay_memory),
        optimizer= torch.optim.AdamW(params=q_net.parameters(), lr = 1E-3),
        batch_size=64
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

            agent.store(state = state, action = action, reward = reward, next_state = next_state, done = done, truncated=truncated)
            loss = agent.train_agent()

            if loss is not None: episode_losses.append(loss)
            
            episode_steps += 1
            state = next_state

        episode_loss = np.array(episode_losses).mean()
        print(f"Episode {i:3d} - Steps : {episode_steps:4d} | Total Rewards : {episode_rewards:7.2f} | Loss : {episode_loss:0.2e} | Agent Step : {agent.step}")
        # print(episode_losses)

        # if episode_rewards >= 500:
        #     agent.plot_atoms_distributions()

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()