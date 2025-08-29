if __name__ == "__main__":
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 

from rl_agents.service import AgentService
from rl_agents.value_functions.value_manager import  SoftDoubleVManager, DoubleVManager
from rl_agents.policies.epsilon_greedy_proxy import EspilonGreedyPolicy
from rl_agents.policies.value_policy import DiscreteBestQValuePolicy
from rl_agents.replay_memory.replay_memory import ReplayMemory, MultiStepReplayMemory
from rl_agents.replay_memory.sampler import PrioritizedReplaySampler
from rl_agents.value_functions.distributional_dqn_function import C51DQN, DiscreteC51QWrapper
from rl_agents.value_agents.dqn import DQNAgent

import torch
import numpy as np
import gymnasium as gym


def main():
    # gym.make("LunarLander-v3")
    env = gym.make("LunarLander-v3")

    action_space = env.action_space # 0 : short , 1 : long
    observation_space = env.observation_space # open, high, low, close, volume

    NB_ENV = 1
    MEMORY_SIZE = 50_000
    GAMMA = 0.99
    HIDDEN_DIM = 128
    BATCH_SIZE = 128
    TRAIN_EVERY = 1
    LR = 3E-4
    TAU = 1./0.005
    MULTI_STEP = 3

    V_MIN, V_MAX = -300, 300
    NB_ATOMS = 51

    replay_memory = MultiStepReplayMemory(
        max_length = MEMORY_SIZE,
        observation_space= observation_space,
        nb_env=NB_ENV,
        gamma=GAMMA,
        multi_step=MULTI_STEP
    )
    # sampler= PrioritizedReplaySampler(replay_memory=replay_memory, batch_size = 64, duration= 100_000),

    core_net  = torch.nn.Sequential(
        torch.nn.Linear(observation_space.shape[0], HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU()
    )
    q_net = DiscreteC51QWrapper(core_net=core_net, action_space=action_space, v_min=V_MIN, v_max=V_MAX, nb_atoms=NB_ATOMS)
    q_manager = SoftDoubleVManager(
        tau= TAU
    )
    q_function = C51DQN(
        net=q_net,
        manager=q_manager,
        gamma=GAMMA,
        v_min=V_MIN, v_max=V_MAX, nb_atoms=NB_ATOMS
    )
    sampler= PrioritizedReplaySampler(
        replay_memory=replay_memory, service=q_function, duration= 50_000
    )

    policy= DiscreteBestQValuePolicy(q = q_function)

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

    agent.train()
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

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()