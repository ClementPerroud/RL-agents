from rl_agents.value_functions.dqn_function import DVNFunction
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
    def __getitem__(self, loc): return self.dataset[self.__len__() - loc - 1]
    def __getitems__(self, loc): return self.dataset[self.__len__() - torch.as_tensor(loc).long() - 1]

class GAEFunction(BaseAdvantageFunction):
    def __init__(self, value_function : DVNFunction, gamma : float, lamb : float, multi_steps=None):
        super().__init__()
        self.value_function = value_function
        self.lamb = lamb

    def _gae(self, experience : ExperienceSample, advantage_tp1 : torch.Tensor) -> torch.Tensor:
        # state: torch.Tensor,  # [B, state_shape ...] obtained at t
        # action: torch.Tensor,  # [B] obtained at t+multi_steps
        # reward: torch.Tensor,  # [B] obtained between t+1 and t+multi_step (then summed up using discounted sum)
        # next_state: torch.Tensor,  # [B, state_shapes ...] obtained at t+multi_steps
        # done: torch.Tensor,  # [B] obtained at t+multi_steps
        # truncated: torch.Tensor,  # [B] obtained at t+multi_steps
        # advantage_tp1 : torch.Tensor, # [B]

        y_pred, y_true = self.value_function.compute_loss_inputs(experience=experience)    
        delta = self.value_function.out_to_value(y_true) - self.value_function.out_to_value(y_pred)

        end = experience.done | experience.truncated

        advantage_t = delta + self.value_function.gamma * self.lamb * (1 - end.float()) * advantage_tp1
        
        # self._state_tp1 = state
        return advantage_t
    
    @torch.no_grad
    def compute(self, agent : AbstractPolicyAgent):
        assert isinstance(agent, AbstractPolicyAgent), "Advantage Functions can only be used with PolicyAgents"
        if "advantage" not in agent.rollout_memory.names: 
            agent.rollout_memory.add_field("advantage", (), torch.float32, default_value=0)

        dataloader = torch.utils.data.DataLoader(
            dataset=ReverseDatasetProxy(dataset=agent.rollout_memory), # Reserving the dataset so we go through the data backwards.
            batch_size = agent.nb_env,
            shuffle= False, in_order=True,
            collate_fn=do_nothing_collate
        )

        advantage = torch.zeros(size=(agent.nb_env,))
        for experience in dataloader:
            experience : ExperienceSample
            advantage = self._gae(experience=experience, advantage_tp1 = advantage)
            agent.rollout_memory["advantage", experience.indices] = advantage
        
        agent.rollout_memory