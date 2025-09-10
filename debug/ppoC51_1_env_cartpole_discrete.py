if __name__ == "__main__":
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 

from rl_agents.service import AgentService
from rl_agents.value_functions.value_manager import  SoftDoubleVManager, DoubleVManager
from rl_agents.policies.epsilon_greedy import EspilonGreedyPolicy
from rl_agents.policies.value_policy import DiscreteBestQValuePolicy
from rl_agents.memory.replay_memory import ReplayMemory, MultiStepReplayMemory
from rl_agents.memory.sampler import PrioritizedReplaySampler, RandomSampler
from rl_agents.value_functions.dqn_function import DQN, DiscreteQWrapper
from rl_agents.value_functions.c51_dqn_function import C51DQN, DiscreteC51Wrapper, C51Loss
from rl_agents.value_agents.dqn import DQNAgent

from rl_agents.policy_agents.ppo_agent import A2CAgent, PPOLoss
from rl_agents.policies.stochastic_policy import DiscreteStochasticPolicy
from rl_agents.value_functions.dvn_function import DVN, VWrapper
from rl_agents.policy_agents.advantage_function import GAEFunction

import torch
import numpy as np
import gymnasium as gym


def main():
    env = gym.make("LunarLander-v3")
    # env = gym.make("CartPole-v1")

    action_space = env.action_space # 0 : short , 1 : long
    observation_space = env.observation_space # open, high, low, close, volume

    NB_ENV = 1
    EPSILON = 0.2
    ROLLOUT_PERIOD = 2048 #TODO : Changes
    EPOCH = 6 #TODO : Changes
    ENTROPY_COEFF = 0.005
    VALUE_COEFF = 0.5
    GAMMA = 0.99
    LAMBDA = 0.95
    HIDDEN_DIM = 128
    BATCH_SIZE = 128

    V_MIN, V_MAX = -300, 300
    NB_ATOMS = 51
    NORMALIZE_ADV = True
    CLIP_VALUE = False
    MAX_GRAD_NORM = 1


    v_core_net = torch.nn.Sequential(
        torch.nn.Linear(observation_space.shape[0], HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
    )
    p_core_net = torch.nn.Sequential(
        torch.nn.Linear(observation_space.shape[0], HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
    )

    v_net = DiscreteC51Wrapper(
        core_net=v_core_net,
        action_space=action_space,
        nb_atoms = NB_ATOMS, v_min=V_MIN, v_max=V_MAX,
        )

    v_function = C51DQN(
        net=v_net,
        gamma=GAMMA,
        nb_atoms = NB_ATOMS, v_min=V_MIN, v_max=V_MAX,
    )

    policy = DiscreteStochasticPolicy(
        core_net=p_core_net, action_space=action_space
    )
    advantage_function = GAEFunction(
        value_function=v_function, lamb=LAMBDA, normalize_advantage=NORMALIZE_ADV
    )
    agent = A2CAgent(
        nb_env=NB_ENV,
        policy = policy,
        advantage_function=advantage_function,
        policy_loss=PPOLoss(
            epsilon = EPSILON,
            entropy_loss_coeff=ENTROPY_COEFF,
            value_loss= C51Loss(),
            value_loss_coeff=VALUE_COEFF,
            clip_value_loss=CLIP_VALUE,
        ),
        rollout_period=ROLLOUT_PERIOD,
        epoch_per_rollout=EPOCH,
        batch_size=BATCH_SIZE,
        max_grad_norm=MAX_GRAD_NORM,
        observation_space=observation_space,
        action_space=action_space
    )

    agent.train()
    episodes = 10000
    for i in range(episodes):
        episode_rewards = 0
        episode_losses = []
        episode_steps = 0

        truncated = False
        done = False
        state, infos = env.reset()
        
        while not truncated and not done:
            action, log_prob = agent.pick_action(state= state)
            next_state, reward, done, truncated, infos = env.step(action = int(action))
            episode_rewards += reward
            # print(state, action, reward, next_state, done, truncated)
            
            agent.store(state = state, action = action, reward = reward, next_state = next_state, done = done, truncated=truncated, log_prob=log_prob)
            loss = agent.train_agent()

            if loss is not None: episode_losses.append(loss)
            
            episode_steps += 1
            state = next_state

        episode_loss = np.array(episode_losses).mean() if len(episode_losses)>0 else np.nan
        print(f"Episode {i:3d} - Steps : {episode_steps:4d} | Total Rewards : {episode_rewards:7.2f} | Loss : {episode_loss:0.2e}| Agent Step : {agent.step}")
        # print(episode_losses)

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()