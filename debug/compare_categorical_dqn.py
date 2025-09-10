if __name__ == "__main__":
    import os
    import sys
    import inspect
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 

from rl_agents.service import AgentService
from rl_agents.value_functions.value_manager import  SoftDoubleVManager, DoubleVManager
from rl_agents.policies.epsilon_greedy import EspilonGreedyPolicy
from rl_agents.policies.value_policy import DiscreteBestQValuePolicy
from rl_agents.memory.replay_memory import ReplayMemory, MultiStepReplayMemory
from rl_agents.memory.sampler import PrioritizedReplaySampler, RandomSampler, LastSampler
from rl_agents.value_functions.c51_dqn_function import C51DQN, DiscreteC51Wrapper
from rl_agents.value_agents.dqn import DQNAgent

import torch
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
import numpy as np
import gymnasium as gym


class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(-1, 1, shape=(10,))
        self.action_space = gym.spaces.Discrete(4)

    def get_obs(self):
        np.random.seed(self.seed)
        obs = np.random.random(size=(10,))
        return obs
    
    def reset(self, *, seed = None, options = None):
        self.seed = 0
        return self.get_obs(), None

    def step(self, action):
        self.seed += 1
        return self.get_obs(), 1, False, False, None
        
def main():
    # gym.make("LunarLander-v3")
    env = DummyEnv()

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
    TAU = 1./0.005

    V_MIN, V_MAX = -300, 300
    NB_ATOMS = 51

    replay_memory = ReplayMemory(
        max_length = MEMORY_SIZE,
        observation_space= observation_space
    )
    sampler= LastSampler(
        replay_memory=replay_memory
    )
    # sampler= PrioritizedReplaySampler(replay_memory=replay_memory, batch_size = 64, duration= 100_000),
    lin_check = torch.nn.Linear(observation_space.shape[0], HIDDEN_DIM)
    core_net  = torch.nn.Sequential(
        lin_check, torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU()
    )

    q_net = DiscreteC51Wrapper(core_net=core_net, action_space=action_space, v_min=V_MIN, v_max=V_MAX, nb_atoms=NB_ATOMS)
    
    q_function = C51DQN(
        net=q_net,
        gamma=GAMMA,
        v_min=V_MIN, v_max=V_MAX, nb_atoms=NB_ATOMS
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
            if agent.step == 1:
                for module in q_net.modules():
                    if isinstance(module, torch.nn.Linear):
                        module.weight.data.fill_(1)
                        module.bias.data.fill_(1)
            next_state, reward, done, truncated, infos = env.step(action = int(action))
            episode_rewards += reward
            # print(state, action, reward, next_state, done, truncated)
            done = done or truncated
            
            agent.store(state = state, action = action, reward = reward, next_state = next_state, done = done, truncated=truncated)
            loss = agent.train_agent()


            if loss is not None:
                episode_losses.append(loss)

                print(state, action, loss, lin_check.weight.grad.flatten()[0:5])
            
            episode_steps += 1
            state = next_state

        epsilon = policy.epsilon
        episode_loss = np.array(episode_losses).mean()
        print(f"Episode {i:3d} - Steps : {episode_steps:4d} | Total Rewards : {episode_rewards:7.2f} | Loss : {episode_loss:0.2e} | Epsilon : {epsilon : 0.2f} | Agent Step : {agent.step}")
        # print(episode_losses)

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()