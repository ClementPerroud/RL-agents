from rl_agents.agent import AbstractAgent
from rl_agents.replay_memory import AbstractReplayMemory
from rl_agents.q_model.deep_q_model import AbstractDeepQNeuralNetwork
from rl_agents.action_strategy.action_strategy import AbstractActionStrategy

import torch
import numpy as np
from abc import ABC, abstractmethod
import typing

class AbstractQAgent(AbstractAgent, ABC):
    @abstractmethod
    def Q(self, state : torch.Tensor): ...

    @abstractmethod
    def Q_a(self, state : torch.Tensor, action : torch.Tensor): ...

class DQNAgent(
        torch.nn.Module,
        AbstractReplayMemory,
        AbstractQAgent,
        ):
    def __init__(self, 
            nb_env : int,
            action_strategy : AbstractActionStrategy,
            gamma : float,
            n_steps : int,
            train_every : int,
            replay_memory : AbstractReplayMemory,
            q_net : AbstractDeepQNeuralNetwork,
            loss_fn : torch.nn.MSELoss = torch.nn.MSELoss(),
            optimizer : torch.optim.Optimizer = None,
            batch_size : int = 64):
                
        torch.nn.Module.__init__(self)
        AbstractAgent.__init__(self, nb_env = nb_env, action_strategy = action_strategy)
        self.n_steps = n_steps
        self.gamma = gamma
        self.replay_memory = replay_memory.connect(self)
        self.batch_size = batch_size
        self.train_every = train_every

        self.q_net = q_net.connect(self)
        self.loss_fn : torch.nn.MSELoss = loss_fn
        self.loss_fn.reduction = "none"
        self.optimizer = torch.optim.Adam(params= self.q_net.parameters(), lr = 1E-3) if optimizer is None else optimizer


    def _pick_deterministic_action(self, state: torch.Tensor) -> np.ndarray:
        return torch.argmax(self.Q(torch.Tensor(state)), dim = -1)
    
    def sample(self):
        return self.replay_memory.sample(batch_size=self.batch_size)
    
    def store(self, **kwargs):
        assert self.training, "Cannot store any memory during eval."
        self.replay_memory.store(**kwargs)

    def train_agent(self) -> float:
        if self.step % self.train_every == 0:
            with torch.no_grad(): samples = self.replay_memory.sample(batch_size=self.batch_size)
            if samples is None: return None
            return self.train_step(**samples)

    def train_step(
        self,
        state : torch.Tensor, # [batch, state_shape ...] obtained at t
        action : torch.Tensor, # [batch] obtained at t+multi_steps
        reward : torch.Tensor, # [batch] obtained between t+1 and t+multi_step (then summed up using discounted sum) 
        next_state : torch.Tensor, # [batch, state_shapes ...] obtained at t+multi_steps
        done : torch.Tensor, # [batch] obtained at t+multi_steps
        weight : torch.Tensor = 1
    ):
        self.optimizer.zero_grad()

        y_pred = self.Q_a(state, action) # [batch/nb_env]

        if reward.ndim == 1: reward[..., None] # Handle both single step (shape : [batch]) and multi-step (shape : [batch, n_steps])
        
        with torch.no_grad():
            y_true = reward
            y_true += torch.where(
                done,
                0,
                (self.gamma ** self.n_steps) * torch.amax(self.Q(next_state, target = True), dim = -1), # is meant to predict the end of the mathematical sequence

            )
        
        loss = (self.loss_fn(y_pred, y_true) * weight).mean()
        loss.backward()

        self.optimizer.step()

        return loss.item()

    def Q(self, state : torch.Tensor, target = False) -> torch.Tensor:
        # state : [batch/nb_env, state_shape ...]
        return self.q_net.forward(state, target = target)
        # Return Q Values : [batch/nb_env, nb_actions]
    
    def Q_a(self, state: torch.Tensor, actions: torch.Tensor, target=False) -> torch.Tensor:
        q_values = self.Q(state, target=target)
        return q_values.gather(dim=1, index=actions.long().unsqueeze(1)).squeeze(1)
    
# ---- Training
# agent.train()
# while True:
#   action = agent.pick_action()
#   env.act(action)
#   agent.store()
#   agent.train()

# ---- Training
# agent.eval()
# while True:
#   action = agent.pick_action()
#   env.act(action)
