from rl_agents.memory.replay_memory import BaseExperienceMemory
from rl_agents.memory.codec import TensorCodec
from rl_agents.memory.replay_memory import MemoryField

import torch
from gymnasium.spaces import Space, Box

class RolloutMemory(BaseExperienceMemory):
    def __init__(
        self,
        max_length: int,
        observation_space: Space,
        action_space: Space,
        device: torch.DeviceObjType = None,
    ):
        assert isinstance(
            observation_space, Box
        ), "RolloutMemory only supports gymnasium.spaces.Box as observation_space"
        self.observation_space = observation_space

        super().__init__(
            max_length=max_length,
            fields=[
                MemoryField("state",        observation_space.shape, torch.float32),
                MemoryField("action",       action_space.shape,      torch.float32),
                MemoryField("next_state",   observation_space.shape, torch.float32),
                MemoryField("reward",       (),                      torch.float32),
                MemoryField("done",         (),                      torch.bool),
                MemoryField("truncated",    (),                      torch.bool),
                MemoryField("log_prob",     (),                      torch.float32),
            ],
            device=device,
        )
