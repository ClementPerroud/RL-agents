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
from rl_agents.replay_memory.replay_memory import ReplayMemory, MultiStepReplayMemory
from rl_agents.replay_memory.sampler import PrioritizedReplaySampler, RandomSampler
from rl_agents.value_functions.dqn_function import DQN, ContinuousQWrapper
from rl_agents.value_agents.dqn import DQNAgent

from rl_agents.policy_agents.ppo_agent import A2CAgent, PPOLoss
from rl_agents.policies.stochastic_policy import ContinuousDeepPolicy
from rl_agents.value_functions.dvn_function import DVN, VWrapper
from rl_agents.policy_agents.advantage_function import GAEFunction

import torch
import numpy as np
import gymnasium as gym

class PolicyNet(torch.nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_net = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU()
        )
        self.mean_linear = torch.nn.Linear(hidden_dim, action_space.shape[0])
        self.std_linear = torch.nn.Linear(hidden_dim, action_space.shape[0])
        
    def forward(self, state):
        x = self.main_net(state)
        return self.mean_linear(x), self.std_linear(x)

def main():
    # gym.make("LunarLander-v3")
    env = gym.make("LunarLander-v3", continuous=True)

    action_space = env.action_space # 0 : short , 1 : long
    observation_space = env.observation_space # open, high, low, close, volume

    NB_ENV = 1
    EPSILON = 0.2
    ROLLOUT_PERIOD = 2048
    EPOCH = 5
    ENTROPY_COEFF = 0.01
    VALUE_COEFF = 0.5
    GAMMA = 0.99
    LAMBDA = 0.95
    HIDDEN_DIM = 128
    BATCH_SIZE = 128

    CLIP_VALUE_LOSS = True
    NORMALIZE_ADV = True
    MAX_GRAD_NORM = None


    v_core_net = torch.nn.Sequential(
        torch.nn.Linear(observation_space.shape[0], HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
    )
    policy_net = PolicyNet(observation_space=observation_space, action_space=action_space, hidden_dim=HIDDEN_DIM)

    v_net = VWrapper(core_net=v_core_net)

    v_function = DVN(
        net=v_net,
        gamma=GAMMA,
    )

    policy = ContinuousDeepPolicy(
        policy_net= policy_net,
        action_space= action_space,
    )

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
            values_loss_coeff=VALUE_COEFF,
            clip_value_loss=CLIP_VALUE_LOSS
            ),
        rollout_period=ROLLOUT_PERIOD,
        epoch_per_rollout=EPOCH,
        batch_size=BATCH_SIZE,
        observation_space=observation_space,
        action_space=action_space,
        max_grad_norm=MAX_GRAD_NORM
    )

    episodes = 10000
    for i in range(0, episodes, 10):
        sma_reward = 0
        for _ in range(10):
            agent.train()
            episode_rewards = 0
            episode_losses = []
            episode_steps = 0

            truncated = False
            done = False
            state, infos = env.reset()
            
            while not truncated and not done:
                action, log_prob = agent.pick_action(state= state)
                next_state, reward, done, truncated, infos = env.step(action = action)
                episode_rewards += reward
                
                agent.store(state = state, action = action, reward = reward, next_state = next_state, done = done, truncated=truncated, log_prob=log_prob)
                loss = agent.train_agent()

                if loss is not None: episode_losses.append(loss)
                
                episode_steps += 1
                state = next_state

            sma_reward = episode_rewards * 0.2 + sma_reward * 0.8
            episode_loss = np.array(episode_losses).mean() if len(episode_losses)>0 else np.nan
            print(f"Episode {i:3d} - Steps : {episode_steps:4d} | Total Rewards : {episode_rewards:7.2f} | Loss : {episode_loss:0.2e}| Agent Step : {agent.step}")

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()