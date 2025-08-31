from rl_agents.value_functions.dqn_function import DQN, Q, Trainable
from rl_agents.value_functions.value_manager import VManager
from rl_agents.replay_memory.replay_memory import ExperienceSample, Experience
from rl_agents.policies.policy import AbstractPolicy
from rl_agents.service import AgentService
from rl_agents.utils.mode import eval_mode
import torch
import matplotlib.pyplot as plt
import gymnasium as gym 
from abc import ABC, abstractmethod

class C51Q(Q, ABC):
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

        self._delta_atoms = (v_max - v_min) / (nb_atoms - 1)
        self.register_buffer("_atoms", torch.linspace(start = self.v_min, end = self.v_max, steps = self.nb_atoms, dtype = torch.float32))
    
    def get_value_from_prob(self, p : torch.Tensor):
        return (p * self._atoms).sum(dim = -1)
        
    @abstractmethod
    def Q_atoms(self, state : torch.Tensor, action : torch.Tensor, **kwargs):...
    def Q(self, state : torch.Tensor, action : torch.Tensor, **kwargs): return self.get_value_from_prob(self.Q_atoms(state=state, action=action))

    @abstractmethod
    def Q_per_action_atoms(self, state : torch.Tensor, **kwargs):...
    def Q_per_action(self, state : torch.Tensor, **kwargs): return self.get_value_from_prob(self.Q_per_action_atoms(state=state))

class ContinuousC51QWrapper(C51Q, Trainable):
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

    def Q_atoms(self, state : torch.Tensor, action : torch.Tensor, **kwargs):
        x : torch.Tensor = self.core_net(state, action)
        # x : [B, H]
        return self.head(x).softmax(dim = -1)
        # output : [B, Atoms]

class DiscreteC51QWrapper(C51Q):
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
    
    def Q_per_action_atoms(self, state : torch.Tensor, **kwargs):
        # state : [B, S shape ...]
        x = self.core_net(state)
        reshaped_x = torch.reshape(self.head(x), shape=(-1, self.nb_actions, self.nb_atoms)) # [B, A, Atoms]
        return reshaped_x.softmax(dim =-1)
    
    def Q_atoms(self, state : torch.Tensor, action : torch.Tensor, **kwargs):
        # state : [B, S shape ...]
        # action (discrete): [B]
        action = action.long().reshape(-1, 1, 1).expand(-1, -1, self.nb_atoms)
        p_per_action = self.Q_per_action_atoms(state=state, **kwargs) # [B, A, Atoms]
        return torch.gather(input=p_per_action, dim = 1, index= action).squeeze(1)
        # output : [B, Atoms]

class C51Loss(torch.nn.modules.loss._Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self, input : torch.Tensor, target : torch.Tensor):
        return - (target * torch.log(input + 1E-8)).sum(-1)

class C51DQN(C51Q, DQN):
    def __init__(self,
            nb_atoms : int,
            v_min : float,
            v_max : float,
            net : AgentService,
            gamma : float,
            loss_fn : torch.nn.modules.loss._Loss = C51Loss(),
            policy : AbstractPolicy = None,
            manager : VManager = VManager(),
            **kwargs
        ):
        assert isinstance(net, C51Q), "net must implement C51Q"
        # super().__init__(nb_atoms=nb_atoms, v_min=v_min, v_max=v_max, net=net, gamma=gamma, loss_fn = loss_fn, policy=policy, manager=manager)
        super().__init__(net=net, gamma=gamma, loss_fn = loss_fn, policy=policy, manager=manager, nb_atoms=nb_atoms, v_min=v_min, v_max=v_max, **kwargs)

    def Q_atoms(self, state, action, target = False, **kwargs): return self.manager.get_net(target=target).Q_atoms(state, action, **kwargs)
    def Q_per_action_atoms(self, state, target = False, **kwargs): return self.manager.get_net(target=target).Q_per_action_atoms(state, **kwargs)

    def V(self, state: torch.Tensor, q_target = False, **kwargs) -> torch.Tensor:
        action = self.policy.pick_action(state = state, **kwargs)
        return self.Q(state=state, action=action, target = q_target)

    def compute_td_errors(self, loss_input : torch.Tensor, loss_target : torch.Tensor):
        return (self.get_value_from_prob(loss_target) - self.get_value_from_prob(loss_input)).abs()

    def compute_loss_input(self, experience : Experience) -> torch.Tensor:
        return self.Q_atoms(state=experience.state, action=experience.action)

    @torch.no_grad()
    def compute_loss_target(self, experience : Experience, **kwargs) -> None:
        batch_size = experience.state.size(0)

        a = self.policy.pick_action(state=experience.next_state, **kwargs)
        p_next = self.Q_atoms(state=experience.next_state, action = a, target=True) #[batch, nb_atoms]
        
        # reward ([batch, 1]) + atoms([1, nb_atoms])
        T_z = torch.clip(
            experience.reward.unsqueeze(-1).to(p_next.dtype)
            + (1 - experience.done.to(p_next.dtype)).unsqueeze(-1)* self.gamma * self._atoms.unsqueeze(0).to(p_next.dtype), 
            min = self.v_min, max = self.v_max
        ) #[batch, nb_atoms]

        b = (T_z - self.v_min) / self._delta_atoms  #[batch, nb_atoms]
        l = torch.floor(b).long().clamp_max(self.nb_atoms - 1) #[batch, nb_atoms]
        u = torch.ceil(b).long().clamp_max(self.nb_atoms - 1) #[batch, nb_atoms]

        m = torch.zeros(size = (batch_size, self.nb_atoms), dtype=p_next.dtype, device=p_next.device) # [batch, nb_atoms]
        m.scatter_add_(1, l, (p_next * (u.float() - b)).float())
        m.scatter_add_(1, u, (p_next * (b - l.float())).float())
        return m