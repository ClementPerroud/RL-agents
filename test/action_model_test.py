from rl_agents.policy.single_policy import SingleActionProxy
from rl_agents.policy.epsilon_greedy_strategy import (
    EspilonGreedyPolicy,
)

import numpy as np
from gymnasium.spaces import Discrete


class AgentTest:
    def __init__(self, nb_env):
        self.nb_env = nb_env
        self.training = True
        self.action_model = SingleActionProxy(action=-1)

def test_signeactionmodel_pick_action():
    state = np.random.rand(5, 3)
    agent = AgentTest(nb_env=5)
    model = SingleActionProxy(action=-1)
    out = model.pick_action(agent, state)
    assert out.shape == (state.shape[0],)
    assert np.all(out == -1)


def test_epsilon_greedy_all_model():
    action_space = Discrete(4)
    state = np.random.rand(4, 2)
    agent = AgentTest(nb_env=4)
    strategy = EspilonGreedyPolicy(
        q=0.99, min_epsilon=0.01, action_space=action_space
    )
    strategy.epsilon = 0.0
    out = strategy.pick_action(agent, state)
    assert np.all(out != 2)


def test_epsilon_greedy_all_random():
    action_space = Discrete(3)
    nb_env = 10
    agent = AgentTest(nb_env=nb_env)
    state = np.random.rand(nb_env, 2)
    proxy = EspilonGreedyPolicy(
        q=0.99, min_epsilon=0.01, action_space=action_space
    )
    proxy.epsilon = 1.0
    out = proxy.pick_action(agent, state)
    assert out.shape[0] == nb_env
    assert not np.all(out == 1)
    assert out.min() >= 0 and out.max() < action_space.n


def test_epsilon_greedy_statistical_test_1():
    action_space = Discrete(3)
    nb_env = 100000
    agent = AgentTest(nb_env=nb_env)
    state = np.random.rand(nb_env, 2)

    proxy = EspilonGreedyPolicy(
        q=0.99, min_epsilon=0.01, action_space=action_space
    )
    proxy.epsilon = 0.8
    out = proxy.pick_action(agent, state)
    assert out.shape[0] == nb_env
    pct_random = (out != -1).sum() / np.prod(out.shape)
    assert abs(proxy.epsilon - pct_random.item()) < 0.01


def test_epsilon_greedy_statistical_test_2():
    action_space = Discrete(3)
    nb_env = 100000
    agent = AgentTest(nb_env=nb_env)
    state = np.random.rand(nb_env, 2)
    proxy = EspilonGreedyPolicy(
        q=0.99, min_epsilon=0.01, action_space=action_space
    )
    proxy.epsilon = 0.3
    out = proxy.pick_action(agent, state)
    assert out.shape[0] == nb_env
    pct_random = (out != -1).sum() / np.prod(out.shape)
    assert abs(proxy.epsilon - pct_random.item()) < 0.01
