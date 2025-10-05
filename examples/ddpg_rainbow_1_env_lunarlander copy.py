if __name__ == "__main__":
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 

from rl_agents.service import AgentService
from rl_agents.modules.target_wrapper import SoftUpdater, EnsembleTargetWrapper, DDPGStrategy
from rl_agents.policies.epsilon_greedy import EpsilonGreedyPolicyWrapper
from rl_agents.memory.replay_memory import ReplayMemory, MultiStepReplayMemory
from rl_agents.memory.sampler import PrioritizedReplaySampler, RandomSampler, Sampler
from rl_agents.modules.noisy_net_strategy import NoisyNetTransformer
from rl_agents.value_functions.q import ContinuousQWrapper
from rl_agents.value_functions.c51q import C51Loss, ContinuousC51QWrapper, plot_q_distribution
from rl_agents.policies.continuous_noise import GaussianNoiseWrapper, OUNoiseWrapper
from rl_agents.policies.deterministic_policy import ContinuousDeterministicPolicy
from rl_agents.trainers.ddpg import DDPGTrainer
from rl_agents.actor_critic_agent import ActorCriticAgent


import torch
import numpy as np
import gymnasium as gym

class QCoreNet(torch.nn.Module):
    def __init__(self, hidden_dim : int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = torch.nn.Sequential(
            torch.nn.LazyLinear(hidden_dim), torch.nn.ReLU(),
            torch.nn.LazyLinear(hidden_dim), torch.nn.ReLU(),
            torch.nn.LazyLinear(hidden_dim), torch.nn.ReLU()
        )
    def forward(self, state, action):
        # state : [B, S]
        # action : [B, A]
        x = torch.cat((state, action), dim = -1)
        return self.net(x)

def main():
    env = gym.make("LunarLander-v3", continuous = True)

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

    q_core_net = QCoreNet(hidden_dim=HIDDEN_DIM)
    q_net = ContinuousC51QWrapper(
        core_net=q_core_net, action_space=action_space,
        nb_atoms=NB_ATOMS, v_min=V_MIN, v_max=V_MAX)

    updater = SoftUpdater(rate= 5E-3, update_every=TRAIN_EVERY)
    q_net = EnsembleTargetWrapper(
        service=q_net,
        target_strategy=DDPGStrategy(),
        updater=updater
    )

    policy_core_net = torch.nn.Sequential(
        torch.nn.LazyLinear(HIDDEN_DIM-1), torch.nn.ReLU(),
        torch.nn.LazyLinear(HIDDEN_DIM-1), torch.nn.ReLU(),
        torch.nn.LazyLinear(HIDDEN_DIM-1), torch.nn.ReLU()
    )
    policy = ContinuousDeterministicPolicy(action_space=action_space, core_net=policy_core_net)

    policy = OUNoiseWrapper(policy=policy, nb_env=NB_ENV, action_space=action_space)
    policy = EnsembleTargetWrapper(
        service=policy,
        target_strategy=DDPGStrategy(),
        updater=updater
    )

    agent = ActorCriticAgent(
        nb_env= NB_ENV,
        actor= policy,
        critic= q_net,
        memory= replay_memory,
        trainer= DDPGTrainer(
            train_every=TRAIN_EVERY,
            batch_size=256,
            q_loss_fn=C51Loss(),
            q_optimizer=torch.optim.Adam(params=q_net.parameters(), lr = 1E-3),
            policy_optimizer=torch.optim.Adam(params=policy.parameters(), lr = 1E-4),
            # sampler = RandomSampler(replay_memory=replay_memory)
            sampler = PrioritizedReplaySampler(replay_memory=replay_memory, alpha=0.65, beta_0=0.5, duration=30_000),
            gamma=GAMMA,
            q_policy=policy
        ),
        observation_space=observation_space,
        action_space=action_space
    )
    agent.to("cpu")
    agent.train()

    episodes = 500
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

            if loss is not None: episode_losses.append([l.item() for l in loss])
            
            episode_steps += 1
            state = next_state

        episode_loss = np.array(episode_losses).mean(axis=0) if len(episode_losses) > 0 else np.nan
        print(f"Episode {agent.nb_episode:3d} - Steps : {episode_steps:4d} | Total Rewards : {episode_rewards:7.2f} | Loss : {episode_loss} | Agent Step : {agent.nb_step} | Total duration : {agent.duration()}")
        
        # if len(replay_memory) > 1_000 and agent.nb_episode % 30 == 0:
        #     plot_q_distribution(c51dqn=q_function, replay_memory=replay_memory, n=1_000)


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()