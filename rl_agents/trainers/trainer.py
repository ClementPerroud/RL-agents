from rl_agents.replay_memory.replay_memory import AbstractReplayMemory
from rl_agents.service import AgentService
from rl_agents.trainers.trainable import Trainable

import logging
import torch

class Trainer(AgentService):
    def __init__(self,
            replay_memory: AbstractReplayMemory = None,
            optimizer : torch.optim.Optimizer = None,
            loss_fn: torch.nn.modules.loss._Loss = None,
            batch_size: int = None,
        ):
        super().__init__()
        if replay_memory is not None: self.replay_memory = replay_memory
        if optimizer is not None: self.optimizer = optimizer
        if loss_fn is not None:
            self.loss_fn = loss_fn
            self.loss_fn.reduction = "none"
        if batch_size is not None:self.batch_size = batch_size

    def _apply_weights(self, loss : torch.Tensor, weight : torch.Tensor | float |None):
        # Handle weights
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=torch.float32)
            try: return loss * weight
            except BaseException as e: 
                raise ValueError(f"Could not apply weight from a loss of shape {loss.shape} and {weight.shape}. Initial Exception : {e}")
        return loss