from rl_agents.value_functions.dqn_function import DVN
from rl_agents.policy_agents.policy_agent import AbstractPolicyAgent
from rl_agents.agent import AbstractAgent
from rl_agents.service import AgentService
from rl_agents.memory.replay_memory import ExperienceSample
from rl_agents.memory.memory import MemoryField
from rl_agents.memory.codec import AutomaticCodecFactory
from rl_agents.utils.collates import do_nothing_collate
from rl_agents.utils.distribution.distribution import Distribution, distribution_aware, distribution_mode
import warnings

from abc import ABC, abstractmethod
from dataclasses import asdict
import torch

class BaseAdvantageFunction(AgentService, ABC):    
    @abstractmethod
    def compute(self):
        ...


class ReverseDatasetProxy(torch.utils.data.Dataset):
    def __init__(self, dataset : torch.utils.data.Dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self): return self.dataset.__len__()
    def __getitem__(self, loc): return self.dataset[self.__len__() - torch.as_tensor(loc).long() - 1]
    def __getitems__(self, loc): return self.dataset[self.__len__() - torch.as_tensor(loc).long() - 1]

class GAEFunction(BaseAdvantageFunction):
    def __init__(self, value_function : DVN, lamb : float, normalize_advantage : bool):
        super().__init__()
        self.value_function = value_function
        self.lamb = lamb
        self.normalize_advantage = normalize_advantage

    @torch.no_grad()
    @distribution_aware
    def _gae(self, experience : ExperienceSample, advantage_tp1 : torch.Tensor = None) -> torch.Tensor:
        # state: torch.Tensor,  # [B, state_shape ...] obtained at t
        # action: torch.Tensor,  # [B] obtained at t+multi_steps
        # reward: torch.Tensor,  # [B] obtained between t+1 and t+multi_step (then summed up using discounted sum)
        # next_state: torch.Tensor,  # [B, state_shapes ...] obtained at t+multi_steps
        # done: torch.Tensor,  # [B] obtained at t+multi_steps
        # truncated: torch.Tensor,  # [B] obtained at t+multi_steps
        # advantage_tp1 : torch.Tensor, # [B]
        end = experience.done | experience.truncated

        value_t = self.value_function.V(experience.state)
        
        with distribution_mode(False):
            delta = (
                experience.reward + (1 - end.float()) * self.value_function.gamma * self.value_function.V(experience.next_state)
                - value_t
            )

        advantage_t = delta

        if advantage_tp1 is not None: advantage_t = advantage_t+ self.value_function.gamma * self.lamb * (1 - end.float()) * advantage_tp1

        return advantage_t, value_t
    

    @torch.no_grad()
    @distribution_aware
    def compute(self, agent : AbstractPolicyAgent):
        assert isinstance(agent, AbstractPolicyAgent), "Advantage Functions can only be used with PolicyAgents"
        reserved_dataset = ReverseDatasetProxy(dataset=agent.rollout_memory)



            # target_shape = self.value_function.compute_loss_target(experience=reserved_dataset[[0]]).shape

        if "advantage" not in agent.rollout_memory.names:
            # Compute shapes :
            one_experience : ExperienceSample = reserved_dataset[[0]]
            one_advantage, one_value = self._gae(experience=one_experience)
            codec_factory = AutomaticCodecFactory()
            agent.rollout_memory.add_field(MemoryField("advantage", shape=one_advantage.shape[1:], dtype=torch.float32, default=0, codec=codec_factory.generate_codec_from_item(one_advantage)))
            agent.rollout_memory.add_field(MemoryField("value", shape=one_value.shape[1:], dtype=torch.float32, default=0, codec=codec_factory.generate_codec_from_item(one_value)))

        assert len(reserved_dataset) % agent.nb_env == 0, "The number of experiences must be divisible by the number of running environments."
        batches = torch.arange(0, len(reserved_dataset)).reshape(shape = (-1, agent.nb_env))


        advantage = None
        for i in range(batches.size(0)):
            indices = batches[i,:]
            experience : ExperienceSample = reserved_dataset[indices]

            advantage, value = self._gae(experience=experience, advantage_tp1 = advantage)
            agent.rollout_memory["advantage", experience.indices] = advantage
            agent.rollout_memory["value", experience.indices] = value
        
        if self.normalize_advantage:
            advantages = agent.rollout_memory["advantage"]
            agent.rollout_memory["advantage"] = (advantages - torch.mean(advantages))/(torch.std(advantages) + 1E-8)