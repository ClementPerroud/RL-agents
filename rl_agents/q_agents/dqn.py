from rl_agents.agent import AbstractAgent
from rl_agents.replay_memory.replay_memory import AbstractReplayMemory, MultiStepReplayMemory
from rl_agents.q_agents.deep_q_model import AbstractDeepQNeuralNetwork
from rl_agents.action_strategy.action_strategy import AbstractActionStrategy
from rl_agents.q_agents.q_agent import AbstractQAgent

import torch
import numpy as np
import logging

class DQNAgent(
    torch.nn.Module,
    AbstractReplayMemory,
    AbstractQAgent,
):
    def __init__(
        self,
        nb_env: int,
        action_strategy: AbstractActionStrategy,
        gamma: float,
        train_every: int,
        replay_memory: AbstractReplayMemory,
        q_net: AbstractDeepQNeuralNetwork,
        loss_fn: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        device : torch.DeviceObjType,
    ):

        torch.nn.Module.__init__(self)
        AbstractAgent.__init__(self, nb_env=nb_env, action_strategy=action_strategy, device= device)
        self.gamma = gamma
        self.replay_memory = replay_memory.connect(self)
        if isinstance(self.replay_memory, MultiStepReplayMemory):
            assert abs(self.replay_memory.gamma - self.gamma) < 1E-8, f"gamma from both {self.__class__.__name__} and {self.replay_memory.__class__.__name__} must be equal. ({self.gamma} != {self.replay_memory.gamma})"
            self.gamma = gamma ** self.replay_memory.multi_step
        
        self.batch_size = batch_size
        self.train_every = train_every

        self.q_net = q_net.connect(self).to(device=device)
        self.loss_fn: torch.nn.MSELoss = loss_fn.to(device=device)
        self.loss_fn.reduction = "none"
        self.optimizer = (
            torch.optim.Adam(params=self.q_net.parameters(), lr=1e-3)
            if optimizer is None
            else optimizer
        )

    def _pick_deterministic_action(self, state: torch.Tensor) -> np.ndarray:
        return torch.argmax(self.Q(torch.as_tensor(state)), dim=-1).detach().cpu().numpy()

    def sample(self):
        return self.replay_memory.sample(batch_size=self.batch_size)

    def store(self, **kwargs):
        assert self.training, "Cannot store any memory during eval."
        for key, value in kwargs.items():
            kwargs[key] = torch.as_tensor(value)
            if self.nb_env == 1: kwargs[key] = kwargs[key][None, ...] # Uniformize the shape, so first dim is always nb_env 
        self.replay_memory.store(agent=self, **kwargs)

    def train_agent(self) -> float:
        self.train()
        if self.step % self.train_every == 0:
            with torch.no_grad():
                samples = self.replay_memory.sample(batch_size=self.batch_size)
            if samples is None:
                return None
            return self.train_step(**samples)
    
    def compute_td_errors(
        self,
        y_true: torch.Tensor,  # [batch, state_shape ...] obtained at t
        y_pred: torch.Tensor,  # [batch] obtained at t+multi_steps
    ):
        return (y_true - y_pred).abs()
        
    def compute_target_predictions(
        self,
        state: torch.Tensor,  # [batch, state_shape ...] obtained at t
        action: torch.Tensor,  # [batch] obtained at t+multi_steps
        reward: torch.Tensor,  # [batch] obtained between t+1 and t+multi_step (then summed up using discounted sum)
        next_state: torch.Tensor,  # [batch, state_shapes ...] obtained at t+multi_steps
        done: torch.Tensor,  # [batch] obtained at t+multi_steps
    ):
        y_pred = self.Q_a(state, action)  # [batch/nb_env]

        with torch.no_grad():
            y_true = reward
            y_true += torch.where(
                done,
                0,
                self.gamma * torch.amax(
                    self.Q(next_state, target=True), dim=-1
                ),  # is meant to predict the end of the mathematical sequence
            )
        return y_true, y_pred

    def train_step(
        self,
        state: torch.Tensor,  # [batch, state_shape ...] obtained at t
        action: torch.Tensor,  # [batch] obtained at t+multi_steps
        reward: torch.Tensor,  # [batch] obtained between t+1 and t+multi_step (then summed up using discounted sum)
        next_state: torch.Tensor,  # [batch, state_shapes ...] obtained at t+multi_steps
        done: torch.Tensor,  # [batch] obtained at t+multi_steps
        weight: torch.Tensor = None,
    ):
        self.optimizer.zero_grad()

        y_true, y_pred = self.compute_target_predictions(
            state=state, action=action, reward=reward, next_state=next_state, done=done
        )

        loss :torch.Tensor = self.loss_fn(y_pred, y_true)
        
        # Handle weights
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=torch.float32)
            if loss.size(0) == weight.size(0):loss = loss * weight
            elif loss.size(0) <= weight.size(0): logging.error(f'Your loss reduction is set to : {self.loss_fn.reduction}. Please set it the "none".')
            
        loss = loss.mean()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 10)
        self.optimizer.step()

        with torch.no_grad():
            self.replay_memory.train_callback(
                agent=self, infos={"td_errors": self.compute_td_errors(y_true=y_true, y_pred=y_pred)}
            )
        # print(y_pred, y_true, loss.item())
        return loss.item()

    def Q(self, state: torch.Tensor, target=False) -> torch.Tensor:
        # state : [batch/nb_env, state_shape ...]
        return self.q_net.forward(state, target=target)
        # Return Q Values : [batch/nb_env, nb_actions]

    def Q_a(
        self, state: torch.Tensor, actions: torch.Tensor, target=False
    ) -> torch.Tensor:
        q_values = self.Q(state, target=target)
        # print("Test2 : ", actions.shape, q_values.shape)
        return q_values.gather(dim=1, index=actions.long().unsqueeze(1)).squeeze(1)
