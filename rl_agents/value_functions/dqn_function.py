from rl_agents.value_functions.q_function import AbstractQFunction
from rl_agents.value_functions.v_function import AbstractVFunction
from rl_agents.replay_memory.replay_memory import AbstractReplayMemory
from rl_agents.service import AgentService
from rl_agents.trainers.trainer import Trainer
from rl_agents.trainers.trainable import Trainable
from rl_agents.utils.mode import eval_mode

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.agent import AbstractAgent

from typing import Callable
import torch

    
class DVNFunction(AbstractVFunction):
    def __init__(self,
            net : AgentService,
            gamma : float,
            multi_steps = None,
            trainer : Trainer = None,
        ):
        torch.nn.Module.__init__(self)
        Trainable.__init__(self, trainer=trainer)
        self.net = net.connect(self)
        self.trainer = trainer.connect(self)
        self.gamma = gamma
        if multi_steps is not None: self.gamma **= multi_steps


    def V(self, state: torch.Tensor, training : bool) -> torch.Tensor:
        # state : [batch/nb_env, state_shape ...]
        return self.net.forward(state, training=training)
        # Return Q Values : [batch/nb_env]
    

    def compute_td_errors(
        self,
        y_true: torch.Tensor,  # [batch]
        y_pred: torch.Tensor,  # [batch]
    ):
        return (y_true - y_pred).abs()

    def out_to_value(self, inputs):
        return inputs

    def compute_loss_inputs(
        self,
        state: torch.Tensor,  # [batch, state_shape ...] obtained at t
        action: torch.Tensor,  # [batch] obtained at t+multi_steps
        reward: torch.Tensor,  # [batch] obtained between t+1 and t+multi_step (then summed up using discounted sum)
        next_state: torch.Tensor,  # [batch, state_shapes ...] obtained at t+multi_steps
        done: torch.Tensor,  # [batch] obtained at t+multi_steps
        **kwargs
    ):
        
        y_pred = self.V(state, training=True)  # [batch/nb_env]
        with torch.no_grad() and eval_mode(self):
            y_true = reward.unsqueeze(-1) + (1 - done.float()).unsqueeze(-1) * self.gamma * self.V(next_state, training=False) # is meant to predict the end of the mathematical sequence
        return y_true, y_pred

    def train_service(self, agent : "AbstractAgent"):
        # Training evert x steps
        with torch.no_grad():
            samples = self.trainer.replay_memory.sample(agent = agent, batch_size=self.trainer.batch_size)
        if samples is None: return None

        q_loss = self.train_step(**samples)
        return q_loss

    def train_step(self,
            state: torch.Tensor,  # [batch, state_shape ...] obtained at t
            action: torch.Tensor,  # [batch] obtained at t+multi_steps
            reward: torch.Tensor,  # [batch] obtained between t+1 and t+multi_step (then summed up using discounted sum)
            next_state: torch.Tensor,  # [batch, state_shapes ...] obtained at t+multi_steps
            done: torch.Tensor,  # [batch] obtained at t+multi_steps
            weight: torch.Tensor = None,
            replay_memory_callbacks: list[Callable] = [],
            **kwargs
            ) -> None:
        self.trainer.optimizer.zero_grad()

        y_true, y_pred = self.compute_loss_inputs(
            state=state, action=action, reward=reward, next_state=next_state, done=done
        )

        loss :torch.Tensor = self.trainer.loss_fn(y_pred, y_true)
        
        loss = self.trainer.apply_weight(loss, weight)
        loss = loss.mean()

        loss.backward()
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 10)
        self.trainer.optimizer.step()

        with torch.no_grad():
            td_errors = self.compute_td_errors(y_true=y_true, y_pred=y_pred)
            for callback in replay_memory_callbacks:
                callback(td_errors = td_errors)

        return loss.item()


class DQNFunction(DVNFunction, AbstractQFunction):
    def __init__(self,
            net : AgentService,
            gamma : float,
            trainer : Trainer = None,
            multi_steps = None,
        ):
        super().__init__(net = net, trainer=trainer, gamma= gamma, multi_steps=multi_steps)

    def Q(self, state: torch.Tensor, training : bool) -> torch.Tensor:
        # state : [batch/nb_env, state_shape ...]
        return self.net.forward(state, training=training)
        # Return Q Values : [batch/nb_env, nb_actions]

    def Q_a(self, state: torch.Tensor, action: torch.Tensor, training : bool) -> torch.Tensor:
        q_values = self.Q(state, training=training)
        # print("Test2 : ", actions.shape, q_values.shape)
        return q_values.gather(dim=1, index=action.long().unsqueeze(1)).squeeze(1)
    
    def V(self, state: torch.Tensor, training : bool) -> torch.Tensor:
        return torch.amax(self.Q(state=state, training=training), dim = -1)
    
    def compute_loss_inputs(
        self,
        state: torch.Tensor,  # [batch, state_shape ...] obtained at t
        action: torch.Tensor,  # [batch] obtained at t+multi_steps
        reward: torch.Tensor,  # [batch] obtained between t+1 and t+multi_step (then summed up using discounted sum)
        next_state: torch.Tensor,  # [batch, state_shapes ...] obtained at t+multi_steps
        done: torch.Tensor,  # [batch] obtained at t+multi_steps
        **kwargs
    ):
        y_pred = self.Q_a(state, action, training=True)  # [batch/nb_env]

        with torch.no_grad():
            y_true = reward
            y_true += torch.where(
                done,
                0,
                self.gamma * self.V(next_state, training= False),  # is meant to predict the end of the mathematical sequence
            )
        return y_true, y_pred