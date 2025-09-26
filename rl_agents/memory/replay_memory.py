from rl_agents.memory.memory import EditableMemory, MemoryField, BaseExperienceMemory

from abc import ABC, abstractmethod
from collections import deque, namedtuple
import torch
import numpy as np
from gymnasium.spaces import Space, Box
from functools import partial
import warnings
from dataclasses import dataclass, make_dataclass, asdict, fields


class ReplayMemory(BaseExperienceMemory):
    def __init__(
        self,
        max_length: int,
        observation_space: Space,
        action_space : Space,
        device: torch.DeviceObjType = None,
    ):
        assert isinstance(
            observation_space, Box
        ), "ReplayMemory only supports gymnasium.spaces.Box as observation_space"
        self.observation_space = observation_space
        self.action_space = action_space

        super().__init__(
            max_length=max_length,
            fields=[
                MemoryField("state",       self.observation_space.shape,    torch.float32),
                MemoryField("action",      self.action_space.shape,         torch.float32),
                MemoryField("next_state",  self.observation_space.shape,    torch.float32),
                MemoryField("reward",      (),                              torch.float32),
                MemoryField("done",        (),                              torch.bool),
                MemoryField("truncated",   (),                              torch.bool),
            ],
            device=device,
        )


class MultiStepReplayMemory(BaseExperienceMemory):
    """
    n-step replay pour plusieurs environnements parallèles.
    Les arguments d'entrée de .store doivent avoir shape (nb_env, …).
    """

    def __init__(
        self,
        max_length: int,
        observation_space: Box,
        action_space : Space,
        nb_env: int,
        gamma: float,
        multi_step: int,
        device: torch.DeviceObjType = None,
    ):
        warnings.warn(
            f"When using {self.__class__.__name__}, please provide the multi_step parameter "
            "for the services that support it (e.g.: DQN, C51DQN ...)"
        )
        assert isinstance(observation_space, Box)
        self.multi_step, self.gamma, self.nb_env = multi_step, gamma, nb_env
        self.buffers = [deque(maxlen=multi_step) for _ in range(nb_env)]
        self.action_space = action_space

        super().__init__(
            max_length=max_length,
            fields=[
                MemoryField("state",       observation_space.shape,     torch.float32),
                MemoryField("action",      self.action_space.shape,     torch.float32),
                MemoryField("next_state",  observation_space.shape,     torch.float32),
                MemoryField("reward",      (),                          torch.float32),
                MemoryField("done",        (),                          torch.bool),
                MemoryField("truncated",   (),                          torch.bool),
            ],
            device=device,
        )

        # pré-calc γ^k pour l’agrégation vectorielle
        self._gammas = gamma ** torch.arange(
            multi_step, device=device, dtype=torch.float32
        )

    def _aggregate(self, buf: deque):
        """
        Convertit le contenu d'un deque en transition n-step.
        -> retourne un dict prêt pour super().store(**transition)
        """
        # Empile récompenses pour un produit scalaire vectoriel
        r = torch.stack([e.reward for e in buf])  # (L,)
        R = torch.dot(self._gammas[: len(r)], r)  # scalaire  γ^k * r_k

        return dict(
            state=buf[0].state[None, ...],
            action=buf[0].action[None, ...],
            next_state=buf[-1].next_state[None, ...],
            reward=R[None, ...],
            done=buf[-1].done[None, ...],
            truncated=buf[-1].truncated[None, ...],
        )


    # ---- API publique ----------------------------------------------
    @torch.no_grad()
    def store(self, **kwargs):
        """
        `state`, `action`, … : tenseurs dont la 0-ème dim = nb_env.
        """
        indices = []
        # Boucle fine sur les envs ; la plupart du temps nb_env <= 16, négligeable.
        for env_id in range(self.nb_env):
            kwargs_env = {key : val[env_id] for key, val in kwargs.items()}
            experience_env = self.__compute_experience_from_values__(**kwargs_env)

            buf = self.buffers[env_id]
            buf.append(experience_env) # We use tensor[env_id:env_id+1] to select the one elem corresponding 

            # fenêtre pleine : pousse une transition n-step
            if len(buf) == self.multi_step:
                indices.append(
                    super().store(**self._aggregate(buf))
                )
                buf.popleft()  # fenêtre glissante

            # fin d'épisode : flush des restes
            if experience_env.done or experience_env.truncated:
                while buf:
                    indices.append(
                        super().store(**self._aggregate(buf))
                    )
                    buf.popleft()

        if len(indices) == 0: return None 
        return torch.cat(indices, dim = 0)


