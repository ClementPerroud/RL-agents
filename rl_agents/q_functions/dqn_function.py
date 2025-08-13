from rl_agents.q_functions.q_function import AbstractQFunction
from rl_agents.service import AgentService

from abc import ABC, abstractmethod
from typing import Callable
import torch
import logging
import gymnasium as gym

class AbstractDeepQNeuralNetwork(torch.nn.Module, AgentService, ABC): ...

class DQNFunction(AbstractQFunction, torch.nn.Module):
    def __init__(self,
            q_net : AbstractDeepQNeuralNetwork,
            optimizer : torch.optim.Optimizer,
            loss_fn: torch.nn.modules.loss._Loss,
            gamma : float
        ):
        torch.nn.Module.__init__(self)
        self.q_net = q_net.connect(self)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_fn.reduction = "none"
        self.gamma = gamma


    def Q(self, state: torch.Tensor, training : bool) -> torch.Tensor:
        # state : [batch/nb_env, state_shape ...]
        return self.q_net.forward(state, training=training)
        # Return Q Values : [batch/nb_env, nb_actions]

    def Q_a(self, state: torch.Tensor, action: torch.Tensor, training : bool) -> torch.Tensor:
        q_values = self.Q(state, training=training)
        # print("Test2 : ", actions.shape, q_values.shape)
        return q_values.gather(dim=1, index=action.long().unsqueeze(1)).squeeze(1)
    

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
        y_pred = self.Q_a(state, action, training=True)  # [batch/nb_env]

        with torch.no_grad():
            y_true = reward
            y_true += torch.where(
                done,
                0,
                self.gamma * torch.amax(
                    self.Q(next_state, training=False), dim=-1
                ),  # is meant to predict the end of the mathematical sequence
            )
        return y_true, y_pred

    def train_q_function(self,
            state: torch.Tensor,  # [batch, state_shape ...] obtained at t
            action: torch.Tensor,  # [batch] obtained at t+multi_steps
            reward: torch.Tensor,  # [batch] obtained between t+1 and t+multi_step (then summed up using discounted sum)
            next_state: torch.Tensor,  # [batch, state_shapes ...] obtained at t+multi_steps
            done: torch.Tensor,  # [batch] obtained at t+multi_steps
            weight: torch.Tensor = None,
            callbacks_q_function_training: list[Callable] = []) -> None:

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
            td_errors = self.compute_td_errors(y_true=y_true, y_pred=y_pred)
            for callback in callbacks_q_function_training:
                callback(td_errors = td_errors)
        # print(y_pred, y_true, loss.item())
        return loss.item()