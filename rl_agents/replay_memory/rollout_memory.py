from rl_agents.replay_memory.replay_memory import BaseReplayMemory
from rl_agents.replay_memory.sampler import AbstractSampler, RandomSampler

import torch
from gymnasium.spaces import Space, Box

class RolloutMemory(BaseReplayMemory):
    def __init__(
        self,
        max_length: int,
        observation_space: Space,
        action_space: Space,
        sampler: AbstractSampler = RandomSampler(),
        device: torch.DeviceObjType = None,
    ):
        assert isinstance(
            observation_space, Box
        ), "RolloutMemory only supports gymnasium.spaces.Box as observation_space"
        self.observation_space = observation_space

        super().__init__(
            max_length=max_length,
            fields=[
                ("state",        observation_space.shape, torch.float32),
                ("action",       action_space.shape,      torch.float32),
                ("next_state",   observation_space.shape, torch.float32),
                ("reward",       (),                      torch.float32),
                ("done",         (),                      torch.bool),
                ("truncated",    (),                      torch.bool),
                ("log_prob", (),                      torch.float32),
            ],
            sampler=sampler,
            device=device,
        )
