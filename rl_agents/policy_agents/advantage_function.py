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


class BackwardDatasetProxy(torch.utils.data.Dataset):
    """Proxy to iterate through dataset in reverse order for GAE computation"""
    
    def __init__(self, dataset: torch.utils.data.Dataset):
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, loc):
        return self.dataset[len(self) - loc - 1]
    
    def __getitems__(self, loc):
        return self.dataset[len(self) - torch.as_tensor(loc).long() - 1]

class GAEFunction(BaseAdvantageFunction):
    def __init__(self, value_function: DVNFunction, gamma: float, lamb: float, multi_steps=None):
        super().__init__()
        self.value_function = value_function
        self.gamma = gamma
        self.lamb = lamb

    @torch.no_grad()
    def compute(self, agent: AbstractPolicyAgent):
        mem = agent.rollout_memory

        if "advantage" not in mem.names: mem.add_field("advantage", (), torch.float32, default_value=0.0)
        if "returns" not in mem.names: mem.add_field("returns", (), torch.float32, default_value=0.0)

        backward_loader = torch.utils.data.DataLoader(
            dataset=BackwardDatasetProxy(dataset=mem), # Allow to iterate through a list backback
            batch_size=agent.nb_env,
            shuffle=False,
            collate_fn=do_nothing_collate,
        )

        next_adv = torch.zeros((agent.nb_env,), dtype=torch.float32, device=mem.memory_state.device)
        # v_next is V(s_{t+1}) for the "next time slice" during backward scan.
        
        v_tp1 = None

        for exp in backward_loader: 
            v_t = self.value_function.V(exp.state) # V(s_t)
            if v_tp1 is None: v_tp1 = self.value_function.V(exp.next_state) # V(s_{t+1})

            nonterminal = (~exp.done).to(torch.float32)

            # δ_t = r_t + γ * 1_{not done} * V(s_{t+1}) - V(s_t)
            delta = exp.reward.to(torch.float32) + self.gamma * nonterminal * v_tp1 - v_t

            # A_t = δ_t + γλ * 1_{not done} * A_{t+1}
            adv_t = delta + self.gamma * self.lamb * nonterminal * next_adv

            ret_t = adv_t + v_t

            mem["advantage", exp.indices] = adv_t
            mem["returns", exp.indices] = ret_t

            next_adv = adv_t.detach()
            v_tp1 = v_t.detach() # For next iteration