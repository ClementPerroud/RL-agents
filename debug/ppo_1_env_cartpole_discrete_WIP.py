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
from rl_agents.replay_memory.sampler import PrioritizedReplaySampler, RandomSampler
from rl_agents.value_functions.dqn_function import DQN, DiscreteQWrapper
from rl_agents.value_agents.dqn import DQNAgent

from rl_agents.policy_agents.ppo_agent import A2CAgent, PPOLoss
from rl_agents.policies.deep_policy import DiscreteDeepPolicy
from rl_agents.value_functions.dqn_function import DVN
from rl_agents.policy_agents.advantage_function import GAEFunction

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
    EPSILON = 0.2
    ROLLOUT_PERIOD = 256
    EPOCH = 5
    ENTROPY_COEFF = 0.001
    VALUE_COEFF = 0.1
    GAMMA = 0.98
    LAMBDA = 1
    HIDDEN_DIM = 128
    BATCH_SIZE = 128

    TAU = 500


    core_net  = torch.nn.Sequential(
        torch.nn.Linear(observation_space.shape[0], HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU()
    )

    q_net = DiscreteQWrapper(
        core_net=torch.nn.Sequential(
            core_net,
            torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU()
        ),
        action_space=action_space
    )
    q_manager = SoftDoubleVManager(
        tau= TAU
    )
    q_function = DQN(
        net=q_net,
        manager=q_manager,
        gamma=GAMMA,
    )

    policy = DiscreteDeepPolicy(
        policy_net= torch.nn.Sequential(
            core_net,
            torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_DIM, action_space.n),
            torch.nn.Softmax(dim = -1)
        )
    )
    advantage_function = GAEFunction(
        value_function=q_function, gamma=GAMMA, lamb=LAMBDA
    )
    agent = A2CAgent(
        nb_env=NB_ENV,
        policy = policy,
        advantage_function=advantage_function,
        policy_loss=PPOLoss(epsilon = EPSILON, entropy_loss_coeff=ENTROPY_COEFF),
        rollout_period=ROLLOUT_PERIOD,
        epoch_per_rollout=EPOCH,
        batch_size=BATCH_SIZE,
        values_loss_coeff=VALUE_COEFF,
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

        episode_loss = np.array(episode_losses).mean()
        print(f"Episode {i:3d} - Steps : {episode_steps:4d} | Total Rewards : {episode_rewards:7.2f} | Loss : {episode_loss:0.2e}| Agent Step : {agent.step}")
        # print(episode_losses)

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()