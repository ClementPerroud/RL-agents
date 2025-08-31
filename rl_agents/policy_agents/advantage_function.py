from rl_agents.value_functions.dqn_function import DVN
from rl_agents.policy_agents.policy_agent import AbstractPolicyAgent
from rl_agents.agent import AbstractAgent
from rl_agents.service import AgentService
from rl_agents.replay_memory.replay_memory import ExperienceSample
from rl_agents.utils.collates import do_nothing_collate
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
    def _gae(self, experience : ExperienceSample, advantage_tp1 : torch.Tensor) -> torch.Tensor:
        # state: torch.Tensor,  # [B, state_shape ...] obtained at t
        # action: torch.Tensor,  # [B] obtained at t+multi_steps
        # reward: torch.Tensor,  # [B] obtained between t+1 and t+multi_step (then summed up using discounted sum)
        # next_state: torch.Tensor,  # [B, state_shapes ...] obtained at t+multi_steps
        # done: torch.Tensor,  # [B] obtained at t+multi_steps
        # truncated: torch.Tensor,  # [B] obtained at t+multi_steps
        # advantage_tp1 : torch.Tensor, # [B]
        end = experience.done | experience.truncated

        value_t = self.value_function.V(experience.state)
        delta = (
            experience.reward + (1 - end.float()) * self.value_function.gamma * self.value_function.V(experience.next_state)
            - value_t
        )

        advantage_t = delta + self.value_function.gamma * self.lamb * (1 - end.float()) * advantage_tp1
        return_t = advantage_t + value_t
        # self._state_tp1 = state
        return advantage_t, return_t
    
    @torch.no_grad()
    def compute(self, agent : AbstractPolicyAgent):
        assert isinstance(agent, AbstractPolicyAgent), "Advantage Functions can only be used with PolicyAgents"
        reserved_dataset = ReverseDatasetProxy(dataset=agent.rollout_memory)

        if "advantage" not in agent.rollout_memory.names:
            agent.rollout_memory.add_field("advantage", (), torch.float32, default_value=0)
            agent.rollout_memory.add_field("_return", (), torch.float32, default_value=0)
            agent.rollout_memory.add_field("value", (), torch.float32, default_value=0)

            # target_shape = self.value_function.compute_loss_target(experience=reserved_dataset[[0]]).shape


        assert len(reserved_dataset) % agent.nb_env == 0, "The number of experiences must be divisible by the number of running environments."
        batches = torch.arange(0, len(reserved_dataset)).reshape(shape = (-1, agent.nb_env))
        advantage = torch.zeros(size=(agent.nb_env,))
        for i in range(batches.size(0)):
            indices = batches[i,:]
            experience : ExperienceSample = reserved_dataset[indices]

            advantage, _return = self._gae(experience=experience, advantage_tp1 = advantage)
            agent.rollout_memory["advantage", experience.indices] = advantage
            agent.rollout_memory["_return", experience.indices] = _return
            agent.rollout_memory["value", experience.indices] = self.value_function.V(experience.state)
        
        if self.normalize_advantage:
            advantages = agent.rollout_memory["advantage"]
            agent.rollout_memory["advantage"] = (advantages - torch.mean(advantages))/(torch.std(advantages) + 1E-8)