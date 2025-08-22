from rl_agents.policies.single_policy import DummyPolicy
from rl_agents.policies.epsilon_greedy_proxy import EspilonGreedyPolicy

import numpy as np
import torch
from gymnasium.spaces import Discrete


class AgentTest:
    def __init__(self, nb_env):
        self.nb_env = nb_env
        self.training = True
        self.step = 0
        self.action_model = DummyPolicy(action=-1)

def test_signeactionmodel_pick_action():
    state = np.random.rand(5, 3)
    agent = AgentTest(nb_env=5)
    model = DummyPolicy(action=-1)
    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    out = model.pick_action(agent, state_tensor, training=True)
    assert out.shape == (state.shape[0],)
    assert torch.all(out == -1)


def test_epsilon_greedy_all_model():
    action_space = Discrete(4)
    state = np.random.rand(4, 2)
    agent = AgentTest(nb_env=4)
    base_policy = DummyPolicy(action=-1)
    strategy = EspilonGreedyPolicy(
        policy=base_policy, q=0.99, start_epsilon=1, end_epsilon=0.01, action_space=action_space
    )
    strategy.epsilon = 0.0
    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    out = strategy.pick_action(agent, state_tensor, training=True)
    assert torch.all(out != 2)


def test_epsilon_greedy_all_random():
    action_space = Discrete(3)
    nb_env = 10
    agent = AgentTest(nb_env=nb_env)
    state = np.random.rand(nb_env, 2)
    base_policy = DummyPolicy(action=-1)
    proxy = EspilonGreedyPolicy(
        policy=base_policy, q=0.99, start_epsilon=1, end_epsilon=0.01, action_space=action_space
    )
    proxy.epsilon = 1.0
    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    out = proxy.pick_action(agent, state_tensor, training=True)
    assert out.shape[0] == nb_env
    assert not torch.all(out == 1)
    assert out.min() >= 0 and out.max() < action_space.n


def test_epsilon_greedy_statistical_test_1():
    action_space = Discrete(3)
    nb_env = 100000
    agent = AgentTest(nb_env=nb_env)
    state = np.random.rand(nb_env, 2)
    base_policy = DummyPolicy(action=-1)
    proxy = EspilonGreedyPolicy(
        policy=base_policy, q=0.99, start_epsilon=1, end_epsilon=0.01, action_space=action_space
    )
    proxy.epsilon = 0.8
    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    out = proxy.pick_action(agent, state_tensor, training=True)
    assert out.shape[0] == nb_env
    pct_random = (out != -1).sum() / torch.prod(torch.tensor(out.shape))
    assert abs(proxy.epsilon - pct_random.item()) < 0.01


def test_epsilon_greedy_statistical_test_2():
    action_space = Discrete(3)
    nb_env = 100000
    agent = AgentTest(nb_env=nb_env)
    state = np.random.rand(nb_env, 2)
    base_policy = DummyPolicy(action=-1)
    proxy = EspilonGreedyPolicy(
        policy=base_policy, q=0.99, start_epsilon=1, end_epsilon=0.01, action_space=action_space
    )
    proxy.epsilon = 0.3
    state_tensor = torch.as_tensor(state, dtype=torch.float32)
    out = proxy.pick_action(agent, state_tensor, training=True)
    assert out.shape[0] == nb_env
    pct_random = (out != -1).sum() / torch.prod(torch.tensor(out.shape))
    assert abs(proxy.epsilon - pct_random.item()) < 0.01
