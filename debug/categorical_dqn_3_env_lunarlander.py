if __name__ == "__main__":
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 

from rl_agents.service import AgentService
from rl_agents.value_functions.target_manager import  SoftDoubleVManager, DoubleVManager
from rl_agents.policies.epsilon_greedy import EspilonGreedyPolicy
from rl_agents.policies.value_policy import DiscreteBestQValuePolicy
from rl_agents.memory.replay_memory import ReplayMemory, MultiStepReplayMemory
from rl_agents.memory.sampler import PrioritizedReplaySampler, RandomSampler
from rl_agents.value_functions.c51_dqn_function import C51DQN, DiscreteC51Wrapper
from rl_agents.value_agents.dqn import DQNAgent

import torch
import numpy as np
import gymnasium as gym
from collections import deque


def main():

    NB_ENV = 3
    MEMORY_SIZE = 50_000
    GAMMA = 0.99
    HIDDEN_DIM = 128
    BATCH_SIZE = 128
    TRAIN_EVERY = 1
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 5000
    LR = 3E-4
    TAU = 1./0.005

    V_MIN, V_MAX = -300, 300
    NB_ATOMS = 51

    env = gym.make_vec("LunarLander-v3", num_envs= NB_ENV, vectorization_mode="sync")

    action_space = env.single_action_space
    observation_space = env.single_observation_space 

    replay_memory = ReplayMemory(
        max_length = MEMORY_SIZE,
        observation_space= observation_space
    )
    sampler= RandomSampler(
        replay_memory=replay_memory
    )
    # sampler= PrioritizedReplaySampler(replay_memory=replay_memory, batch_size = 64, duration= 100_000),

    core_net  = torch.nn.Sequential(
        torch.nn.Linear(observation_space.shape[0], HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU()
    )
    q_net = DiscreteC51Wrapper(core_net=core_net, action_space=action_space, v_min=V_MIN, v_max=V_MAX, nb_atoms=NB_ATOMS)
    q_manager = SoftDoubleVManager(
        tau= TAU
    )
    q_function = C51DQN(
        net=q_net,
        manager=q_manager,
        gamma=GAMMA,
        v_min=V_MIN, v_max=V_MAX, nb_atoms=NB_ATOMS
    )

    policy = EspilonGreedyPolicy(
        epsilon_decay= EPS_DECAY,
        start_epsilon= EPS_START,
        end_epsilon= EPS_END,
        action_space= action_space,
        policy= DiscreteBestQValuePolicy(q = q_function),
    )
    
    optimizer = torch.optim.AdamW(params=q_function.parameters(), lr = LR, amsgrad=True)

    agent = DQNAgent(
        nb_env= NB_ENV,
        policy= policy,
        train_every= TRAIN_EVERY,
        q_function= q_function,
        replay_memory=replay_memory,
        sampler=sampler,
        optimizer= optimizer,
        batch_size=BATCH_SIZE,
    )


    episode_rewards = np.zeros(shape=(NB_ENV,))
    episode_losses = deque(maxlen= 200)
    episode_steps = np.zeros(shape=(NB_ENV,), dtype=int)

    agent.train()
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

            for i in range(NB_ENV):
                if done[i]:
                    episodes += 1
                    episode_loss = np.array(episode_losses).mean()
                    print(f"Episode {episodes:3d} - Steps : {episode_steps[i]:4d} | Total Rewards : {episode_rewards[i]:7.2f} | Loss : {episode_loss:0.2e} | Epsilon : {policy.epsilon : 0.2f} | Agent Step : {agent.nb_step}")
                    episode_steps[i] = 0
                    episode_rewards[i] = 0

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()