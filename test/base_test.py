from rl_agents.q_model.deep_q_model import AbstractDeepQNeuralNetwork
from rl_agents.action_strategy.epsilon_greedy_strategy import EspilonGreedyActionStrategy
from rl_agents.replay_memory import ReplayMemory
from rl_agents.dqn import DQNAgent

import torch
import numpy as np
import gymnasium as gym

class QNN(AbstractDeepQNeuralNetwork):
    def __init__(self, observation_space : gym.spaces.Space, action_space : gym.spaces.Discrete, hidden_dim :int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We suppose observation_space and action_space to be 1D
        self.lin1 = torch.nn.Linear(in_features= observation_space.shape[0], out_features= hidden_dim)
        self.activation1 = torch.nn.ReLU()
        self.head = torch.nn.Linear(in_features=hidden_dim, out_features=action_space.n)

    
    def forward(self, state : torch.Tensor, **kwargs) -> torch.Tensor:
        # state : [batch, nb_obs]
        x = self.activation1(self.lin1(state))
        return self.head(x)


# gym.make("LunarLander-v3")
env = gym.make("CartPole-v1")

action_space = env.action_space # 0 : short , 1 : long
observation_space = env.observation_space # open, high, low, close, volume

nb_env = 1
replay_memory = ReplayMemory(length = 1E5, observation_space= observation_space)


qnn = QNN(observation_space=observation_space, action_space= action_space, hidden_dim= 64)
action_strategy = EspilonGreedyActionStrategy(
    q = 1 - 1E-4,
    action_space= action_space
)
agent = DQNAgent(
    nb_env= nb_env,
    action_strategy= action_strategy,
    n_steps= 1,
    train_every= 10,
    gamma=0.99,
    tau = 1000,
    replay_memory= replay_memory,
    q_net = qnn
)

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
        next_state, reward, done, truncated, infos = env.step(action = int(action))
        episode_rewards += reward
        # print(state, action, reward, next_state, done, truncated)

        agent.store(state = state, action = action, reward = reward, next_state = next_state, done = done)
        loss = agent.train_agent()
        if loss is not None: episode_losses.append(loss)
        episode_steps += 1
        state = next_state

    epsilon = action_strategy.epsilon
    episode_loss = np.array(episode_losses).mean()
    print(f"Episode {i} - Steps : {episode_steps:d} | Total Rewards : {episode_rewards:0.2f} | Loss : {episode_loss:0.2e} | Epsilon : {epsilon : 0.2f} | Agent Step : {agent.step}")

