if __name__ == "__main__":
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 

from rl_agents.service import AgentService
from rl_agents.value_functions.target_manager import  DoubleVManager, SoftUpdater
from rl_agents.policies.epsilon_greedy import EspilonGreedyPolicy
from rl_agents.policies.value_policy import DiscreteBestQValuePolicy
from rl_agents.memory.replay_memory import ReplayMemory, MultiStepReplayMemory
from rl_agents.memory.sampler import PrioritizedReplaySampler, RandomSampler
from rl_agents.value_functions.c51_dqn_function import C51DQN, DiscreteC51Wrapper
from rl_agents.value_agents.dqn import DQNAgent

import torch
import numpy as np
import gymnasium as gym


def main():
    # gym.make("LunarLander-v3")
    env = gym.make("CartPole-v1")

    action_space = env.action_space # 0 : short , 1 : long
    observation_space = env.observation_space # open, high, low, close, volume

    NB_ENV = 1
    MEMORY_SIZE = 50_000
    GAMMA = 0.99
    HIDDEN_DIM = 128
    BATCH_SIZE = 128
    TRAIN_EVERY = 1
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 5000
    LR = 3E-4
    TAU = 0.005

    V_MIN, V_MAX = -300, 300
    NB_ATOMS = 51

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
    q_manager = DoubleVManager(
        updater= SoftUpdater(rate = TAU, update_every= TRAIN_EVERY),
        context_strategy= DoubleVManager.ddqn_context_strategy
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

        epsilon = policy.epsilon
        episode_loss = np.array(episode_losses).mean()
        print(f"Episode {i:3d} - Steps : {episode_steps:4d} | Total Rewards : {episode_rewards:7.2f} | Loss : {episode_loss:0.5e} | Epsilon : {epsilon : 0.2f} | Agent Step : {agent.nb_step}")
        # print(episode_losses)

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()