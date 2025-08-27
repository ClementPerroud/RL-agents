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

    @property
    def trainer(self):
        try:
            if self._trainer is not None: return self._trainer
            raise ValueError(f"Please provide a Trainer to {self.__class__.__name__}")
        except AttributeError as e:
            raise ValueError(f"Please call Trainable.__init__(self, trainer = trainer) in __init__ method of object {self.__class__.__name__}.")
        
    @trainer.setter
    def trainer(self, value : 'Trainer'):
        self._trainer = value