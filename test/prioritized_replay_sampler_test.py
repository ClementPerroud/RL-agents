import numpy as np
import torch

from rl_agents.memory.sampler import PrioritizedReplaySampler

class DummyAgent:
    def __init__(self):
        self.nb_step = 0
        self.training = True


def test_sampler_initialization():
    sampler = PrioritizedReplaySampler(max_length=10, beta_0=0.7, duration=5)
    assert sampler.beta_0 == 0.7
    assert sampler.duration == 5


def test_sampler_train_callback_uses_small_epsilon():
    sampler = PrioritizedReplaySampler(max_length=1)
    sampler.priorities.add(1)
    sampler.train_callback(
        batch=torch.as_tensor([0], dtype=torch.float32),
        td_errors = torch.as_tensor([1], dtype=torch.float32)
    )
    expected = (1 + 1e-6) ** sampler.alpha
    assert np.isclose(sampler.priorities[0], expected)


def test_sampler_random_fallback():
    agent = DummyAgent()
    sampler = PrioritizedReplaySampler(max_length=10, duration=1)
    for _ in range(10):
        sampler.priorities.add(1.0)
    sampler.nb_step = 1
    batch, weights = sampler.sample(agent=agent, batch_size=2, size=10)
    assert torch.isclose(weights, torch.ones_like(weights)).all()
    assert len(batch) == 2

