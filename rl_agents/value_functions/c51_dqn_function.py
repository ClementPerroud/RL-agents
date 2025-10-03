from rl_agents.value_functions.dqn_function import Q
from rl_agents.value_functions.value import Op
from rl_agents.memory.memory import Experience, Memory
from rl_agents.memory.sampler import RandomSampler
from rl_agents.policies.policy import Policy
from rl_agents.service import AgentService
from rl_agents.utils.distribution.distribution import Distribution, LinearAtomConfig, BaseAtomConfig, distribution_aware, distribution_mode
import torch
import matplotlib.pyplot as plt
import gymnasium as gym 
from abc import ABC
from typing import Protocol, runtime_checkable, Union

@runtime_checkable
class C51(Protocol):
    nb_atoms : int
    v_min : float
    v_max : float
    atom_config : BaseAtomConfig

class BaseC51(AgentService, ABC):
    def __init__(self,
            nb_atoms : int,
            v_min : float,
            v_max : float,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.nb_atoms = nb_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.atom_config = LinearAtomConfig(nb_atoms=nb_atoms, v_min=v_min, v_max=v_max)
        

class ContinuousC51Wrapper(BaseC51, Q, AgentService):
    def __init__(self,
            core_net : torch.nn.Module,
            action_space : gym.spaces.Box,
            nb_atoms : int,
            v_min : float,
            v_max : float,
            **kwargs
        ):
        assert isinstance(action_space, gym.spaces.Box), "action_space only support Box."
        super().__init__(nb_atoms=nb_atoms, v_min=v_min, v_max=v_max, **kwargs)
        # Core net : 
        # inputs [B, state_shape ... ], [B, action_shape] -> [B, hidden_dim]
        self.core_net = core_net
        self.action_space = action_space

        self.head = torch.nn.LazyLinear(self.nb_atoms)

    @distribution_aware
    def Q(self, state : torch.Tensor, action : torch.Tensor, **kwargs):
        x : torch.Tensor = self.core_net(state, action)
        # x : [B, H]
        dist = self.head(x).softmax(dim = -1)
        return Distribution(dist, atom_config=self.atom_config)
        # output : [B, Atoms]
    
    def Q_per_action(self, *args, **kwargs): raise ValueError("Continuous action cannot use Q_per_action.")

class DiscreteC51Wrapper(BaseC51, Q, AgentService):
    def __init__(self,
            core_net : torch.nn.Module,
            action_space : gym.spaces.Discrete,
            nb_atoms : int,
            v_min : float,
            v_max : float,
            **kwargs
        ):
        assert isinstance(action_space, gym.spaces.Discrete), "action_space must be Discrete"
        super().__init__(nb_atoms=nb_atoms, v_min=v_min, v_max=v_max, **kwargs)
        # Core net : 
        # inputs [B, state_shape ... ] -> [B, hidden_dim]
        self.core_net = core_net
        self.action_space = action_space
        self.nb_actions = self.action_space.n
        self.head = torch.nn.LazyLinear(self.nb_actions * self.nb_atoms)

        # Final outputs : [B, A, Atoms]
    @distribution_aware
    def Q_per_action(self, state : torch.Tensor, **kwargs):
        # state : [B, S shape ...]
        x = self.core_net(state)
        reshaped_x = torch.reshape(self.head(x), shape=(-1, self.nb_actions, self.nb_atoms)) # [B, A, Atoms]
        dist = reshaped_x.softmax(dim =-1)
        return Distribution(dist, atom_config=self.atom_config)
    
    @distribution_aware
    def Q(self, state : torch.Tensor, action : torch.Tensor, **kwargs):
        # state : [B, S shape ...]
        # action (discrete): [B]
        action = action.long().reshape(-1, 1, 1).expand(-1, -1, self.nb_atoms)
        p_per_action = self.Q_per_action(state=state, **kwargs) # [B, A, Atoms]
        dist = torch.gather(input=p_per_action, dim = 1, index= action).squeeze(1)
    
        return dist
        # output : [B, Atoms]

class C51Loss(torch.nn.modules.loss._Loss):
    @distribution_aware
    def forward(self, input: Distribution, target: Distribution):
        target = target.project_on(input.atom_config).to_tensor() # base
        input = input.to_tensor()
        input = input.clamp(min=1E-8, max = 1E8)
        return -(target * torch.log(input)).sum(-1)





@distribution_aware
@torch.no_grad()
def plot_q_distribution( c51dqn : Union[BaseC51, Q], policy : Policy, replay_memory : Memory, n = 1_000, quantile_range = (0.05, 0.95)):
    if n > len(replay_memory):
        raise ValueError(f"n (:{n}) must be greater than len(replay_memory) (:{len(replay_memory)}).")
    sampler = RandomSampler(replay_memory=replay_memory)
    batch = sampler.sample(batch_size=n)
    experience = replay_memory[batch]

    with distribution_mode(False):
        best_action = policy.pick_action(state = experience.next_state, op = Op.DQN_LOSS_TARGET_PICK_ACTION)
    values = c51dqn.Q(state=experience.state, action=best_action, op=Op.DQN_LOSS_TARGET_Q).to_tensor()

    quantiles = values.quantile(torch.as_tensor(quantile_range), dim = 0).cpu().numpy()
    mean = values.mean(dim=0).cpu().numpy()
    atoms = c51dqn.atom_config.get_atoms().cpu().numpy()

    plt.plot(atoms, mean, color="b")
    plt.fill_between(
            x= atoms, 
            y1= quantiles[0],
            y2= quantiles[1], 
            color= "b",
            alpha= 0.2)
    plt.show()


