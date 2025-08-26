from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rl_agents.trainers.trainer import Trainer

class Trainable(ABC):
    def __init__(self, trainer : 'Trainer'):
        super().__init__()
        self._trainer = trainer
    @abstractmethod
    def train_service(self): ...
    
    @abstractmethod
    def compute_loss_inputs(self, experience): ...

    @abstractmethod
    def compute_td_errors(self): ...

    _trainer = None
    @property
    def trainer(self) -> int:
        if self._trainer is None: raise AttributeError(f"{self.__class__.__name__}.trainer is not set")
        return self._trainer
    @trainer.setter
    def trainer(self, val): self._trainer = val