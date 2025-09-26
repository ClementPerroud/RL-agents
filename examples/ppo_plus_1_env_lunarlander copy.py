if __name__ == "__main__":
    import os
    import sys
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir) 

from rl_agents.policies.stochastic_policy import DiscreteStochasticPolicy
from rl_agents.value_functions.dvn_function import DVN, VWrapper
from rl_agents.critics.advantage_function import GAEFunction

from rl_agents.actor_critic_agent import ActorCriticAgent
from rl_agents.trainers.ppo import PPOTrainer

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
    env = gym.make("LunarLander-v3")

    action_space = env.action_space # 0 : short , 1 : long
    observation_space = env.observation_space # open, high, low, close, volume

    NB_ENV = 1
    EPSILON = 0.2
    ROLLOUT_PERIOD = 2048
    EPOCH = 5
    ENTROPY_COEFF = 0.005
    VALUE_COEFF = 0.5
    GAMMA = 0.99
    LAMBDA = 0.90
    HIDDEN_DIM = 128
    BATCH_SIZE = 128

    CLIP_VALUE_LOSS = True
    NORMALIZE_ADV = True
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

    v_net = VWrapper(core_net=v_core_net)

    v_function = DVN(
        net=v_net,
        gamma=GAMMA,
    )

    policy = DiscreteStochasticPolicy(core_net= p_core_net, action_space=action_space)

    advantage_function = GAEFunction(
        value_function=v_function, lamb=LAMBDA, normalize_advantage=NORMALIZE_ADV
    )

    agent = ActorCriticAgent(
        nb_env= NB_ENV,
        actor= policy,
        critic= advantage_function,
        trainer= PPOTrainer(
            epsilon=EPSILON,
            entropy_loss_coeff=ENTROPY_COEFF,
            value_loss=torch.nn.MSELoss(reduce = "none"),
            value_loss_coeff=VALUE_COEFF,
            clip_value_loss=CLIP_VALUE_LOSS,
            max_grad_norm=MAX_GRAD_NORM,
            rollout_period=ROLLOUT_PERIOD,
            epoch_per_rollout=EPOCH,
            batch_size=BATCH_SIZE
        ),
        observation_space=observation_space,
        action_space=action_space
    )

    episodes = 10_000
    for i in range(episodes):
        episode_rewards = 0
        episode_losses = []
        episode_steps = 0

        truncated = False
        done = False
        state, infos = env.reset()
        
        while not truncated and not done:
            action, log_prob  = agent.pick_action(state= state)

            next_state, reward, done, truncated, infos = env.step(action = action)
            episode_rewards += reward
            # print(state, action, reward, next_state, done, truncated)
            done = done or truncated

            agent.store(state = state, action = action, reward = reward, next_state = next_state, done = done, truncated=truncated, log_prob=log_prob)
            loss = agent.train_agent()

            if loss is not None: episode_losses.append(loss)
            
            episode_steps += 1
            state = next_state

        episode_loss = np.array(episode_losses).mean()
        print(f"Episode {i:3d} - Steps : {episode_steps:4d} | Total Rewards : {episode_rewards:7.2f} | Loss : {episode_loss:0.2e} | Agent Step : {agent.nb_step}")
        # print(episode_losses)

        # if episode_rewards >= 500:
        #     agent.plot_atoms_distributions()

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    main()