import numpy as np
import torch
from gymnasium.spaces import Box

from rl_agents.policies.policy import AbstractPolicy
from rl_agents.value_functions.dqn_function import AbstractDeepQNeuralNetwork, DQN
from rl_agents.value_agents.dqn import DQNAgent
from rl_agents.replay_memory.replay_memory import MultiStepReplayMemory
from rl_agents.trainers.trainer import Trainer
from rl_agents.replay_memory.sampler import RandomSampler


class DummyPolicy(AbstractPolicy):
    def pick_action(self, agent, state):
        return torch.zeros(agent.nb_env, dtype=torch.long)


class DummyQNetwork(AbstractDeepQNeuralNetwork):
    def __init__(self, nb_actions):
        super().__init__()
        self.nb_actions = nb_actions
        # add a dummy parameter so that optimizer has parameters to update
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, state, target: bool = False):
        state = torch.as_tensor(state, dtype=torch.float32)
        batch = state.shape[0]
        return torch.zeros(batch, self.nb_actions)


def test_dqn_agent_n_step_replay_memory():
    nb_env = 1
    gamma = 0.9
    n_steps = 10
    obs_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    memory = MultiStepReplayMemory(
        max_length=2000,
        observation_space=obs_space,
        nb_env=nb_env,
        multi_step=n_steps,
        gamma=gamma,
        sampler=RandomSampler()
    )
    
    trainer = Trainer(
        replay_memory=memory,
        batch_size=1,
        optimizer=torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    )
    
    q_function = DQN(
        net=DummyQNetwork(nb_actions=2),
        gamma=gamma,
        multi_steps=n_steps,
        trainer=trainer
    )

    agent = DQNAgent(
        nb_env=nb_env,
        policy=DummyPolicy(),
        q_function=q_function,
        train_every=1
    )

    states, actions, rewards, next_states, dones, truncateds = [], [], [], [], [], []
    for _ in range(1000):
        state = np.random.rand(nb_env, obs_space.shape[0]).astype(np.float32).squeeze(0)
        action = np.random.randint(0, 2, size=(nb_env,), dtype=np.int64).squeeze(0)
        reward = np.random.rand(nb_env).astype(np.float32).squeeze(0)
        next_state = np.random.rand(nb_env, obs_space.shape[0]).astype(np.float32).squeeze(0)
        done = np.zeros(nb_env, dtype=bool).squeeze(0)
        truncated = np.zeros(nb_env, dtype=bool).squeeze(0)

        agent.store(state=state, action=action, reward=reward, next_state=next_state, done=done, truncated=truncated)

        states.append(state.copy())
        actions.append(action.copy())
        rewards.append(reward.copy())
        next_states.append(next_state.copy())
        dones.append(done.copy())
        truncateds.append(truncated.copy())

    expected_size = 1000 - n_steps + 1
    assert len(agent.q_function.trainer.replay_memory) == expected_size

    mem_state = agent.q_function.trainer.replay_memory["state"].cpu().numpy()
    mem_action = agent.q_function.trainer.replay_memory["action"].cpu().numpy()
    mem_reward = agent.q_function.trainer.replay_memory["reward"].cpu().numpy()
    mem_next_state = agent.q_function.trainer.replay_memory["next_state"].cpu().numpy()
    mem_done = agent.q_function.trainer.replay_memory["done"].cpu().numpy()
    mem_truncated = agent.q_function.trainer.replay_memory["truncated"].cpu().numpy()

    gammas = gamma ** np.arange(n_steps, dtype=np.float32)

    for i in range(expected_size):
        np.testing.assert_array_equal(mem_state[i], states[i])
        np.testing.assert_array_equal(mem_action[i], actions[i])
        expected_reward = np.dot(gammas, [rewards[i + k] for k in range(n_steps)])
        np.testing.assert_allclose(mem_reward[i], expected_reward, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(mem_next_state[i], next_states[i + n_steps - 1])
        assert mem_done[i] == dones[i + n_steps - 1]
