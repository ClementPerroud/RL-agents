from rl_agents.action_model.model import OTPActionModel
from rl_agents.action_model.epsilon_greedy_proxy import EspilonGreedyProxy

import torch
import numpy as np
from  gymnasium.spaces import Discrete

def test_otpactionmodel_pick_action():
    states = torch.rand(5, 3)
    model = OTPActionModel(nb_env=5, action=-1)
    out = model.pick_action(states)
    assert out.shape == (states.shape[0],)
    assert torch.all(out == -1)

def test_epsilon_greedy_all_model():
    action_space = Discrete(4)
    states = torch.rand(4, 2)
    model = OTPActionModel(nb_env=4, action=2)
    proxy = EspilonGreedyProxy(action_model=model, nb_env=4, action_space=action_space)
    proxy.epsilon = 0.0
    out = proxy.pick_action(states)
    assert torch.all(out == 2)

def test_epsilon_greedy_all_random():
    action_space = Discrete(3)
    nb_env = 10
    states = torch.rand(nb_env, 2)
    model = OTPActionModel(nb_env=nb_env, action=1)
    proxy = EspilonGreedyProxy(action_model=model, nb_env=nb_env, action_space=action_space)
    proxy.epsilon = 1.0
    out = proxy.pick_action(states)
    assert out.shape[0] == nb_env
    assert not torch.all(out == 1)
    assert out.min() >= 0 and out.max() < action_space.n


def test_epsilon_greedy_statistical_test_1():
    action_space = Discrete(3)
    nb_env = 100000
    states = torch.rand(nb_env, 2)
    model = OTPActionModel(nb_env=nb_env, action=-1)
    proxy = EspilonGreedyProxy(action_model=model, nb_env=nb_env, action_space=action_space)
    proxy.epsilon = 0.8
    out = proxy.pick_action(states)
    assert out.shape[0] == nb_env
    pct_random = (out != -1).sum() / out.numel()
    assert abs(proxy.epsilon - pct_random.item()) < 0.01

def test_epsilon_greedy_statistical_test_2():
    action_space = Discrete(3)
    nb_env = 100000
    states = torch.rand(nb_env, 2)
    model = OTPActionModel(nb_env=nb_env, action=-1)
    proxy = EspilonGreedyProxy(action_model=model, nb_env=nb_env, action_space=action_space)
    proxy.epsilon = 0.3
    out = proxy.pick_action(states)
    assert out.shape[0] == nb_env
    pct_random = (out != -1).sum() / out.numel()
    assert abs(proxy.epsilon - pct_random.item()) < 0.01