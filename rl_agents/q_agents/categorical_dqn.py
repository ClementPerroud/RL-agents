from rl_agents.q_agents.dqn import DQNAgent
from rl_agents.agent import AbstractAgent
from rl_agents.replay_memory.replay_memory import AbstractReplayMemory, MultiStepReplayMemory
from rl_agents.q_agents.deep_q_model import AbstractDeepQNeuralNetwork
from rl_agents.action_strategy.action_strategy import AbstractActionStrategy
from rl_agents.q_agents.q_agent import AbstractQAgent
import torch
import matplotlib.pyplot as plt

class CategoricalLoss(torch.nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        
    def forward(self, input : torch.Tensor, target : torch.Tensor):
        return - (target * (input + 1E-8).log()).sum(-1)

class CategoricalDQNAgent(DQNAgent):
    def __init__(self,
            nb_atoms : int,
            v_min : float,
            v_max : float,
            **dqn_kwargs,
        ):
        super().__init__(**dqn_kwargs)
        self.nb_atoms = nb_atoms
        self.v_min = v_min
        self.v_max = v_max

        self._delta_atoms = (v_max - v_min) / (nb_atoms - 1)
        self._atoms : torch.Tensor = torch.linspace(start = self.v_min, end = self.v_max, steps = self.nb_atoms, dtype = torch.float32, device= self.device)

    def compute_td_errors(
        self,
        y_true: torch.Tensor,  # [batch, state_shape ...] obtained at t
        y_pred: torch.Tensor,  # [batch] obtained at t+multi_steps
    ):
        return (
            y_true * (y_true.add(1e-8).log() - y_pred.add(1e-8).log())
        ).sum(dim=-1)

    def compute_target_predictions(
        self,
        state: torch.Tensor,  # [batch, state_shape ...] obtained at t
        action: torch.Tensor,  # [batch] obtained at t+multi_steps
        reward: torch.Tensor,  # [batch] obtained between t+1 and t+multi_step (then summed up using discounted sum)
        next_state: torch.Tensor,  # [batch, state_shapes ...] obtained at t+multi_steps
        done: torch.Tensor,  # [batch] obtained at t+multi_steps
    ):  
        # TODO : check if we should use target model or not
        batch_size = state.size(0)
        y_pred = self.Q_a(state = state, actions = action, target= False, return_atoms= True)

        with torch.no_grad():
            p_next = self.Q(state=next_state, target= True, return_atoms= True) #[batch, nb_actions, nb_atoms]
            q_next = (p_next * self._atoms).sum(dim = -1) # [batch, nb_actions]
            a_star = torch.argmax(q_next, dim = 1, keepdim= True) #[batch]
            p_next = p_next.gather(1, a_star.unsqueeze(-1).expand(-1, 1, self.nb_atoms)).squeeze(1) # [batch, nb_atoms]


            # reward ([batch, 1]) + atoms([1, nb_atoms])
            T_z = torch.clip(reward.unsqueeze(-1) + 
                    (1 - done.to(p_next.dtype)).unsqueeze(-1)* self.gamma * self._atoms.unsqueeze(0), min = self.v_min, max = self.v_max
            ) #[batch, nb_atoms]
            b = (T_z - self.v_min) / self._delta_atoms  #[batch, nb_atoms]
            l = torch.floor(b).long().clamp_max(self.nb_atoms - 1) #[batch, nb_atoms]
            u = torch.ceil(b).long().clamp_max(self.nb_atoms - 1) #[batch, nb_atoms]
    
            
            m = torch.zeros(size = (batch_size, self.nb_atoms), device=self.device) # [batch, nb_atoms]
            m.scatter_add_(1, l, p_next * (u.float() - b))
            m.scatter_add_(1, u, p_next * (b - l.float()))
            y_true = m
        if torch.isnan(y_pred).any() or torch.isnan(y_true).any():
            print("Catch")
        return y_true, y_pred #both : [batch, nb_atoms]

    def Q(self, state: torch.Tensor, target=False, return_atoms = False) -> torch.Tensor:
        q_logits = self.q_net.forward(state, target=target)
        q_prob = torch.softmax(q_logits, dim=-1) #[batch/nb_env, nb_actions, nb_atoms]
        return q_prob if return_atoms else q_prob @ self._atoms
    
    def Q_a(
        self, state: torch.Tensor, actions: torch.Tensor, target=False, return_atoms = False
    ) -> torch.Tensor:
        # actions : [batch]
        q_values = self.Q(state, target=target, return_atoms=return_atoms) #[batch/nb_env, nb_actions, nb_atoms] if return_atoms is True else [batch/nb_env, nb_actions]
        batch_idx = torch.arange(q_values.size(0), device=q_values.device)
        return q_values[batch_idx, actions.long()]  # shape [B] ou [B, N]
    
    def plot_atoms_distributions(self, n_samples: int = 10_000, max_batch: int = 512):
        """
        Estime la distribution C51 moyenne P(z|a) sur `n_samples` transitions,
        sans jamais dépasser `max_batch` exemples simultanément sur le GPU.
        """
        if self.replay_memory.size() == 0:
            print("ReplayMemory is empty.")
            return

        self.eval()
        nb_actions, nb_atoms = self.action_strategy.action_space.n, self.nb_atoms
        sum_p = torch.zeros(nb_actions, nb_atoms, device=self.device)
        seen = 0

        with torch.no_grad():
            while seen < n_samples:
                batch_sz = min(max_batch, n_samples - seen)
                batch    = self.replay_memory.sample(batch_sz)
                if batch is None:          # pas assez de données stockées
                    break

                states = batch["state"].to(self.device)
                p      = self.Q(states, return_atoms=True)   # (B, A, N)
                sum_p += p.sum(dim=0)                        # accumulate sur B
                seen  += p.size(0)

            if seen == 0:
                print("Pas assez de transitions pour échantillonner.")
                return

            mean_p = sum_p / seen                           # (A, N)

        # ---------- Plot ----------
        z = self._atoms.cpu().numpy()
        plt.figure(figsize=(8, 4))
        for a in range(nb_actions):
            plt.plot(z, mean_p[a].cpu().numpy(),
                    label=f"action {a}", linewidth=2)
        plt.title(f"Distribution moyenne sur {seen} transitions")
        plt.xlabel("z (valeurs atomiques)")
        plt.ylabel("Probabilité")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()
        self.train()
if __name__ == "__main__":
    agent = CategoricalDQNAgent(
        nb_atoms= 51,
        v_min= 0,
        v_max= 51,
        nb_env= 1, action_strategy= None, gamma= 0.99, train_every= 10, replay_memory= None, q_net= None, loss_fn= None, optimizer= None, batch_size= 54
    )