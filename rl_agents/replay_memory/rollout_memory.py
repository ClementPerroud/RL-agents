from rl_agents.replay_memory.replay_memory import BaseReplayMemory
from rl_agents.replay_memory.sampler import AbstractSampler, RandomSampler

import torch
from gymnasium.spaces import Space, Box

class RolloutMemory(BaseReplayMemory):
    def __init__(
        self,
        length: int,
        observation_space: Space,
        sampler: AbstractSampler = RandomSampler(),
        device: torch.DeviceObjType = None,
    ):
        assert isinstance(
            observation_space, Box
        ), "ReplayMemory only supports gymnasium.spaces.Box as observation_space"
        self.observation_space = observation_space

        super().__init__(
            length=length,
            names=["state", "action", "next_state", "reward", "done"],
            sizes=[
                self.observation_space.shape,
                (),
                self.observation_space.shape,
                (),
                (),
            ],
            dtypes=[
                torch.float32,
                torch.long,
                torch.float32,
                torch.float32,
                torch.bool,
            ],
            sampler=sampler,
            device=device,
        )