from rl_agents.replay_memory.sampler import AbstractSampler, RandomSampler
from rl_agents.replay_memory.replay_memory import BaseReplayMemory

from gymnasium.spaces import Space, Box
import torch

class PPOTrainingMemory(BaseReplayMemory):
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
        ), "ReplayMemory only supports gymnasium.spaces.Box as observation_space"
        self.observation_space = observation_space

        super().__init__(
            max_length=max_length,
            names=["state", "action", "next_state", "reward", "done", "truncated", "old_action_log_likelihood", "advantage"],
            sizes=[
                self.observation_space.shape,
                action_space.shape,
                self.observation_space.shape,
                (),
                (),
                (),
                (),
                (),
            ],
            dtypes=[
                torch.float32,
                torch.float32,
                torch.float32,
                torch.float32,
                torch.bool,
                torch.bool,
                torch.float32,
                torch.float32,
            ],
            sampler=sampler,
            device=device,
        )