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
from rl_agents.value_functions.dqn_function import DQN, DiscreteQWrapper
from rl_agents.value_agents.dqn import DQNAgent

from rl_agents.policy_agents.ppo_agent import A2CAgent, PPOLoss
from rl_agents.policies.stochastic_policy import DiscreteStochasticPolicy
from rl_agents.value_functions.dvn_function import DVN, VWrapper
from rl_agents.critics.advantage_function import GAEFunction

import torch
import numpy as np
import gymnasium as gym
from collections import deque


def main():

    NB_ENV = 8
    EPSILON = 0.2
    ROLLOUT_PERIOD = 256
    EPOCH = 5
    ENTROPY_COEFF = 0.005
    VALUE_COEFF = 0.5
    GAMMA = 0.99
    LAMBDA = 0.95
    HIDDEN_DIM = 128
    BATCH_SIZE = 128

    CLIP_VALUE_LOSS = True
    NORMALIZE_ADV = True
    MAX_GRAD_NORM = None

    env = gym.make_vec("LunarLander-v3", num_envs= NB_ENV, vectorization_mode="sync")
    action_space = env.single_action_space # 0 : short , 1 : long
    observation_space = env.single_observation_space # open, high, low, close, volume

    v_core_net = torch.nn.Sequential(
        torch.nn.Linear(observation_space.shape[0], HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
    )
    p_core_net = torch.nn.Sequential(
        torch.nn.Linear(observation_space.shape[0], HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
    )

    v_net = VWrapper(core_net=v_core_net)

    v_function = DVN(
        net=v_net,
        gamma=GAMMA,
    )

    policy = DiscreteStochasticPolicy(core_net= p_core_net, action_space=action_space)

    advantage_function = GAEFunction(
        value_function=v_function, lamb=LAMBDA, normalize_advantage=NORMALIZE_ADV
    )
    agent = A2CAgent(
        nb_env=NB_ENV,
        policy = policy,
        advantage_function=advantage_function,
        policy_loss=PPOLoss(
            epsilon=EPSILON,
            entropy_loss_coeff=ENTROPY_COEFF,
            value_loss= torch.nn.MSELoss(reduction="none"),
            value_loss_coeff=VALUE_COEFF,
            clip_value_loss=CLIP_VALUE_LOSS
            ),
        rollout_period=ROLLOUT_PERIOD,
        epoch_per_rollout=EPOCH,
        batch_size=BATCH_SIZE,
        observation_space=observation_space,
        action_space=action_space,
        max_grad_norm=MAX_GRAD_NORM
    )

    episode_rewards = np.zeros(shape=(NB_ENV,))
    episode_losses = deque(maxlen= 200)
    episode_steps = np.zeros(shape=(NB_ENV,), dtype=int)

    agent.train()
    episodes = 0
    state, infos = env.reset()
    while episodes < 20000:
            action, log_prob = agent.pick_action(state= state)
            next_state, reward, done, truncated, infos = env.step(actions = action.astype(int))
            # print(state, action, reward, next_state, done, truncated)
            done = done | truncated

            agent.store(state = state, action = action, reward = reward, next_state = next_state, done = done, truncated=truncated, log_prob = log_prob)
            loss = agent.train_agent()

            if loss is not None: episode_losses.append(loss)
            
            state = next_state
            episode_rewards += reward
            episode_steps += 1

            for i in range(NB_ENV):
                if done[i]:
                    episodes += 1
                    episode_loss = np.array(episode_losses).mean()
                    print(f"Episode {episodes:3d} - Steps : {episode_steps[i]:4d} | Total Rewards : {episode_rewards[i]:7.2f} | Loss : {episode_loss:0.2e} | Agent Step : {agent.nb_step}")
                    episode_steps[i] = 0
                    episode_rewards[i] = 0


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()