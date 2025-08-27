from rl_agents.value_functions.q_function import AbstractQFunction
from rl_agents.value_functions.v_function import AbstractVFunction
from rl_agents.replay_memory.replay_memory import AbstractReplayMemory, ExperienceSample, Experience
from rl_agents.policies.policy import AbstractPolicy
from rl_agents.policies.value_policy import QValuePolicy
from rl_agents.service import AgentService
from rl_agents.utils.mode import eval_mode, train_mode
from contextlib import nullcontext

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent

from typing import Callable
import torch

    
class DVNFunction(AbstractVFunction):
    def __init__(self,
            net : AgentService,
            gamma : float,
            loss_fn : torch.nn.modules.loss._Loss
        ):
        torch.nn.Module.__init__(self)
        self.net = net
        self.gamma = gamma
        self.loss_fn = loss_fn
        loss_fn.reduction= "none"

    
    def V(self, state: torch.Tensor, compute_target = False) -> torch.Tensor:
        # state : [batch/nb_env, state_shape ...]
        if compute_target:
            with eval_mode(self): return self.net.forward(state).squeeze(-1)
        return self.net.forward(state).squeeze(-1)

        # Return Q Values : [batch/nb_env]
    

    def compute_td_errors(self, loss_inputs : tuple[torch.Tensor, torch.Tensor]):
        y_pred, y_true = loss_inputs
        return (y_true - y_pred).abs()


    def compute_loss_inputs(self, experience : Experience) -> None:
        y_pred = self.V(experience.state)  # [batch/nb_env]
        with torch.no_grad():
            y_true = experience.reward + (1 - experience.done.float()) * self.gamma * self.V(experience.next_state, compute_target=True) # is meant to predict the end of the mathematical sequence
        return (y_pred, y_true)

    def get_loss(self, agent : "AbstractAgent", experience : ExperienceSample) -> float:
        # state: torch.Tensor,  # [batch, state_shape ...] obtained at t
        # action: torch.Tensor,  # [batch] obtained at t+multi_steps
        # reward: torch.Tensor,  # [batch] obtained between t+1 and t+multi_step (then summed up using discounted sum)
        # next_state: torch.Tensor,  # [batch, state_shapes ...] obtained at t+multi_steps
        # done: torch.Tensor,  # [batch] obtained at t+multi_steps
        # weight: torch.Tensor = None,
        # replay_memory_callbacks: Callable

        loss_inputs = self.compute_loss_inputs(experience)

        loss :torch.Tensor = self.loss_fn(*loss_inputs)
    
        with torch.no_grad():
            agent.sampler.update_experiences(
                agent = agent, indices = experience.indices, td_errors = self.compute_td_errors(loss_inputs=loss_inputs)
            )

        return loss

class DQNFunction(DVNFunction, AbstractQFunction):
    def __init__(self,
            net : AgentService,
            gamma : float,
            loss_fn : torch.nn.modules.loss._Loss,
            policy : AbstractPolicy
        ):
        super().__init__(net = net, gamma= gamma, loss_fn=loss_fn)
        self.policy = policy

    # def Q(self, state: torch.Tensor) -> torch.Tensor:
    #     # state : [batch/nb_env, state_shape ...]
    #     return self.net.forward(state)
    #     # Return Q Values : [batch/nb_env, nb_actions]

    def Q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_values = self.net(state, action)
        # print("Test2 : ", actions.shape, q_values.shape)
        return q_values
    
    def V(self, state: torch.Tensor, compute_target = False) -> torch.Tensor:
        best_action = self.policy.pick_action(state= state)
        #  If compute target is True : Q_θ'(s_t, argmax_{a} (Q_θ(s_t, a)))
        with eval_mode(self) if compute_target else nullcontext():
            return self.Q(state=state, action=best_action) # We use Q'
        
    
    def compute_loss_inputs(self, experience : ExperienceSample) -> None:
        y_pred = self.Q(experience.state, experience.action)  # [batch/nb_env]

        with torch.no_grad():
            y_true = experience.reward + (1 - experience.done.float()) * self.gamma * self.V(experience.next_state, compute_target=True)  # is meant to predict the end of the mathematical sequence
        return y_pred, y_true