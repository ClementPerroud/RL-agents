if __name__ == "__main__":
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 

from rl_agents.service import AgentService
from rl_agents.value_functions.target_manager import SoftUpdater, TargetManagerWrapper, DDPGStrategy, DDQNStrategy
from rl_agents.policies.epsilon_greedy import EspilonGreedyPolicyWrapper
from rl_agents.policies.value_policy import DiscreteBestQValuePolicy
from rl_agents.memory.replay_memory import ReplayMemory, MultiStepReplayMemory
from rl_agents.memory.sampler import PrioritizedReplaySampler, RandomSampler, Sampler
from rl_agents.value_agents.noisy_net_strategy import NoisyNetTransformer
from rl_agents.value_functions.dqn_function import ContinuousQWrapper
from rl_agents.value_functions.c51_dqn_function import C51Loss, ContinuousC51Wrapper, DiscreteC51Wrapper, plot_q_distribution
from rl_agents.policies.continuous_noise import GaussianNoiseWrapper, OUNoiseWrapper
from rl_agents.policies.deterministic_policy import ContinuousDeterministicPolicy
# from rl_agents.policy_agents.ddpg import DQNTrainer, ActorCriticAgent, DQNLoss
from rl_agents.actor_critic_agent import ActorCriticAgent
from rl_agents.trainers.dqn import DQNTrainer

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
    V_MIN, V_MAX = -400, 300
    GAMMA = 0.99
    HIDDEN_DIM = 128
    MULTI_STEP = 3
    TRAIN_EVERY = 3

    replay_memory = MultiStepReplayMemory(
        max_length = MEMORY_SIZE,
        observation_space= observation_space,
        action_space=action_space,
        nb_env=NB_ENV,
        gamma=GAMMA,
        multi_step=MULTI_STEP
    )

    q_core_net = torch.nn.Sequential(
        torch.nn.LazyLinear(HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.LazyLinear(HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.LazyLinear(HIDDEN_DIM), torch.nn.ReLU()
    )
    main_q_fn = DiscreteC51Wrapper(
        core_net=q_core_net, action_space=action_space,
        nb_atoms=NB_ATOMS, v_min=V_MIN, v_max=V_MAX)

    
    # q_fn = main_q_fn
    q_fn = TargetManagerWrapper(
        service=main_q_fn,
        target_strategy=DDQNStrategy(),
        updater=SoftUpdater(rate= 5E-3, update_every=TRAIN_EVERY)
    )

    policy = EspilonGreedyPolicyWrapper(
        policy=DiscreteBestQValuePolicy(q_fn = q_fn),
        action_space=action_space,
        epsilon_decay=5000,
        start_epsilon=0.9,
        end_epsilon=0.03
    )

    agent = ActorCriticAgent(
        nb_env = NB_ENV,
        actor = policy,
        critic = q_fn,
        memory = replay_memory,
        trainer = DQNTrainer(
            loss_fn = C51Loss(),
            optimizer = torch.optim.Adam(params=q_fn.parameters(), lr = 2.5E-4),
            sampler = PrioritizedReplaySampler(replay_memory=replay_memory, alpha=0.65, beta_0=0.5, duration=30_000),
            train_every = TRAIN_EVERY,
            batch_size = 256,
            gamma = GAMMA,
            q_policy = DiscreteBestQValuePolicy(q_fn = q_fn)
        ),
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
            action = agent.pick_action(state= state)

            next_state, reward, done, truncated, infos = env.step(action = action)
            episode_rewards += reward
            # print(state, action, reward, next_state, done, truncated)
            done = done or truncated

            agent.store(state = state, action = action, reward = reward, next_state = next_state, done = done, truncated=truncated)
            loss = agent.train_agent()

            if loss not in [np.nan, None]: episode_losses.append(loss)
            
            episode_steps += 1
            state = next_state

        episode_loss = np.array(episode_losses).mean(axis=0) if len(episode_losses) > 0 else np.nan
        print(f"Episode {i:3d} - Steps : {episode_steps:4d} | Total Rewards : {episode_rewards:7.2f} | Loss : {episode_loss} | Agent Step : {agent.nb_step}")
        
        # if len(replay_memory) > 1_000 and agent.nb_episode % 30 == 0:
        #     plot_q_distribution(
        #         c51dqn=q_fn,
        #         policy = DiscreteBestQValuePolicy(q = main_q_fn),
        #         replay_memory=replay_memory,
        #         n=1_000
        #     )


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()