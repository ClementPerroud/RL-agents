from rl_agents.value_functions.dqn_function import DQNFunction
from rl_agents.policies.policy import AbstractPolicy
from rl_agents.replay_memory.replay_memory import ExperienceSample, Experience
from rl_agents.service import AgentService
from rl_agents.trainers.trainer import Trainer
from rl_agents.utils.mode import eval_mode
import torch
import matplotlib.pyplot as plt

class DistributionalLoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        
    def forward(self, input : torch.Tensor, target : torch.Tensor):
        return - (target * (input + 1E-8).log()).sum(-1)

class DistributionalDQNFunction(DQNFunction):
    def __init__(self,
            nb_atoms : int,
            v_min : float,
            v_max : float,
            net : AgentService,
            gamma : float,
            loss_fn : torch.nn.modules.loss._Loss,
            policy : AbstractPolicy
        ):
        super().__init__(net=net, gamma=gamma, loss_fn = loss_fn, policy=policy)
        self.nb_atoms = nb_atoms
        self.v_min = v_min
        self.v_max = v_max

        self._delta_atoms = (v_max - v_min) / (nb_atoms - 1)
        self.register_buffer("_atoms", torch.linspace(start = self.v_min, end = self.v_max, steps = self.nb_atoms, dtype = torch.float32))

    def compute_td_errors(self, loss_inputs : tuple[torch.Tensor, torch.Tensor]):
        y_pred, y_true = loss_inputs
        return self.get_value_from_atom_probs(y_true) - self.get_value_from_atom_probs(y_pred)
    
    def get_value_from_atom_probs(self, inputs):
        return (inputs * self._atoms).sum(dim = -1)

    def compute_loss_inputs(self, experience : Experience) -> None:
        batch_size = experience.state.size(0)
        y_pred = self.Q(state = experience.state, action = experience.action, return_atoms= True)

        with torch.no_grad(), eval_mode(self):
            p_next = self.Q_all_actions(state=experience.next_state, return_atoms= True) #[batch, nb_actions, nb_atoms]
            q_next = self.get_value_from_atom_probs(p_next) # [batch, nb_actions]
            a_star = self.policy.pick_action(state=experience.next_state) #[batch]
            p_next = p_next.gather(1, a_star.unsqueeze(-1).expand(-1, 1, self.nb_atoms)).squeeze(1) # [batch, nb_atoms]

            # reward ([batch, 1]) + atoms([1, nb_atoms])
            T_z = torch.clip(
                experience.reward.unsqueeze(-1) + (1 - experience.done.to(p_next.dtype)).unsqueeze(-1)* self.gamma * self._atoms.unsqueeze(0), 
                min = self.v_min, max = self.v_max
            ) #[batch, nb_atoms]
            
            b = (T_z - self.v_min) / self._delta_atoms  #[batch, nb_atoms]
            l = torch.floor(b).long().clamp_max(self.nb_atoms - 1) #[batch, nb_atoms]
            u = torch.ceil(b).long().clamp_max(self.nb_atoms - 1) #[batch, nb_atoms]
    
            m = torch.zeros(batch_size, self.nb_atoms, dtype=torch.float32, device=p_next.device)# [batch, nb_atoms]

            m.scatter_add_(1, l, (p_next * (u.float() - b)).float())
            m.scatter_add_(1, u, (p_next * (b - l.float())).float())
            y_true = m

        return (y_pred, y_true) #both : [batch, nb_atoms]

    def Q_all_actions(self, state: torch.Tensor, return_atoms = False) -> torch.Tensor:
        q_logits = self.net.forward(state)
        q_prob = torch.softmax(q_logits, dim=-1) #[batch/nb_env, nb_actions, nb_atoms]
        return q_prob if return_atoms else self.get_value_from_atom_probs(q_prob)
    
    def Q(
        self, state: torch.Tensor, action: torch.Tensor, return_atoms = False
    ) -> torch.Tensor:
        # actions : [batch]
        return self.net(state, action)
        q_values = self.Q_all_actions(state, return_atoms=return_atoms) #[batch/nb_env, nb_actions, nb_atoms] if return_atoms is True else [batch/nb_env, nb_actions]
        batch_idx = torch.arange(q_values.size(0), device=q_values.device)
        return q_values[batch_idx, action.long()]  # shape [B] ou [B, N]
    
    
