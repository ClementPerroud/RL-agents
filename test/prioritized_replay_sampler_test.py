import numpy as np
import torch

from rl_agents.replay_memory.sampler import PrioritizedReplaySampler


def test_sampler_initialization():
    sampler = PrioritizedReplaySampler(length=10, beta_0=0.7, duration=5)
    assert sampler.beta_0 == 0.7
    assert sampler.duration == 5


def test_sampler_train_callback_uses_small_epsilon():
    sampler = PrioritizedReplaySampler(length=1)
    sampler.last_batch = [0]
    infos = {"y_true": torch.zeros(1), "y_pred": torch.zeros(1)}
    sampler.train_callback(agent=None, infos=infos)
    expected = (1e-6) ** sampler.alpha
    assert np.isclose(sampler.priorities[0], expected)


def test_sampler_random_fallback():
    sampler = PrioritizedReplaySampler(length=10, duration=1)
    for _ in range(10):
        sampler.priorities.add(1.0)
    sampler.step = 1
    batch, weights = sampler.sample(batch_size=2, size=10)
    assert weights == 1
    assert len(batch) == 2

