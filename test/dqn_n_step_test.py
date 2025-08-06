import numpy as np
import torch
from gymnasium.spaces import Box

from rl_agents.action_strategy.action_strategy import AbstractActionStrategy
from rl_agents.q_agents.deep_q_model import AbstractDeepQNeuralNetwork
from rl_agents.q_agents.dqn import DQNAgent
from rl_agents.replay_memory.replay_memory import MultiStepReplayMemory


class DummyActionStrategy(AbstractActionStrategy):
    def pick_action(self, agent, state):
        return np.zeros(agent.nb_env, dtype=np.int64)


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
        length=2000,
        observation_space=obs_space,
        num_envs=nb_env,
        multi_step=n_steps,
        gamma=gamma,
    )
    agent = DQNAgent(
        nb_env=nb_env,
        action_strategy=DummyActionStrategy(),
        gamma=gamma ** n_steps,
        train_every=1,
        replay_memory=memory,
        q_net=DummyQNetwork(nb_actions=2),
        batch_size=1,
    )

    states, actions, rewards, next_states, dones = [], [], [], [], []
    for _ in range(1000):
        state = np.random.rand(nb_env, obs_space.shape[0]).astype(np.float32)
        action = np.random.randint(0, 2, size=(nb_env,), dtype=np.int64)
        reward = np.random.rand(nb_env).astype(np.float32)
        next_state = np.random.rand(nb_env, obs_space.shape[0]).astype(np.float32)
        done = np.zeros(nb_env, dtype=bool)

        agent.store(state=state, action=action, reward=reward, next_state=next_state, done=done)

        states.append(state.copy())
        actions.append(action.copy())
        rewards.append(reward.copy())
        next_states.append(next_state.copy())
        dones.append(done.copy())

    expected_size = 1000 - n_steps + 1
    assert memory.size() == expected_size

    mem_state = memory.tensor_memories["state"][:expected_size].cpu().numpy()
    mem_action = memory.tensor_memories["action"][:expected_size].cpu().numpy()
    mem_reward = memory.tensor_memories["reward"][:expected_size].cpu().numpy()
    mem_next_state = memory.tensor_memories["next_state"][:expected_size].cpu().numpy()
    mem_done = memory.tensor_memories["done"][:expected_size].cpu().numpy()

    gammas = gamma ** np.arange(n_steps, dtype=np.float32)

    for i in range(expected_size):
        np.testing.assert_array_equal(mem_state[i], states[i][0])
        np.testing.assert_array_equal(mem_action[i], actions[i][0])
        expected_reward = np.dot(gammas, [rewards[i + k][0] for k in range(n_steps)])
        np.testing.assert_allclose(mem_reward[i], expected_reward, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(mem_next_state[i], next_states[i + n_steps - 1][0])
        assert mem_done[i] == dones[i + n_steps - 1][0]
