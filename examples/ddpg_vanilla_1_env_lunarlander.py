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
# from rl_agents.value_functions.c51_dqn_function import C51DQN, C51Loss, DiscreteC51Wrapper
from rl_agents.policies.continuous_noise import GaussianNoiseWrapper 
from rl_agents.policies.deterministic_policy import ContinuousDeterministicPolicy
from rl_agents.actor_critic_agent import ActorCriticAgent
from rl_agents.trainers.ddpg import DDPGTrainer

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
    # NB_ATOMS = 51
    # V_MIN, V_MAX = -250, 300
    GAMMA = 0.99
    HIDDEN_DIM = 128
    # MULTI_STEP = 3
    TRAIN_EVERY = 3

    replay_memory = ReplayMemory(
        max_length = MEMORY_SIZE,
        observation_space= observation_space,
        action_space=action_space
    )

    q_core_net = QCoreNet(hidden_dim=HIDDEN_DIM)
    q_net = ContinuousQWrapper(core_net=q_core_net, action_space=action_space)

    updater = SoftUpdater(rate= 5E-3, update_every=TRAIN_EVERY)
    q_net = EnsembleTargetWrapper(
        service=q_net,
        target_strategy=DDPGStrategy(),
        updater=updater
    )

    policy_core_net = torch.nn.Sequential(
        torch.nn.LazyLinear(HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.LazyLinear(HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.LazyLinear(HIDDEN_DIM), torch.nn.ReLU()
    )
    policy = ContinuousDeterministicPolicy(action_space=action_space, core_net=policy_core_net)
    policy = GaussianNoiseWrapper(policy=policy, std=0.1)
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
            q_loss_fn=torch.nn.MSELoss(reduction="none"),
            q_optimizer=torch.optim.Adam(params=q_net.parameters(), lr = 1E-3),
            policy_optimizer=torch.optim.Adam(params=policy.parameters(), lr = 1E-4),
            sampler = RandomSampler(replay_memory=replay_memory),
            gamma= GAMMA,
            q_policy=policy
        ),
        observation_space=observation_space,
        action_space=action_space
    )
    agent.to(device="cuda")
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
            next_state, reward, done, truncated, infos = env.step(action = action)
            episode_rewards += reward
            # print(state, action, reward, next_state, done, truncated)
            done = done or truncated

            agent.store(state = state, action = action, reward = reward, next_state = next_state, done = done, truncated=truncated)
            loss = agent.train_agent()

            if loss is not None: episode_losses.append([l.item() for l in loss])
            
            episode_steps += 1
            state = next_state

        episode_loss = np.array(episode_losses).mean(axis=0) if len(episode_losses) > 0 else None
        print(f"Episode {agent.nb_episode:3d} - Steps : {episode_steps:4d} | Total Rewards : {episode_rewards:9.2f} | Loss : {episode_loss} | Total Steps : {agent.nb_step :5d} | Total duration : {agent.duration()}")


if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()