from rl_agents.service import AgentService

from abc import ABC, abstractmethod
import torch

class ActionModel(AgentService, ABC):
    @abstractmethod
    def pick_action(self, states) -> torch.Tensor:
        ...


# Use for testing purpose
class OTPActionModel(ActionModel):
    def __init__(self, nb_env, action = 0):
        self.action = action
        
        self.returned_action = torch.Tensor([action])

        self.repeat = [1] * self.returned_action.ndim


    def pick_action(self, states : torch.Tensor):
        self.repeat[0] = states.shape[0]
        return self.returned_action.repeat(self.repeat)
    
# if __name__ == "__main__":
#     model  = OTPActionModel(nb_env=4, action= -1)
#     print(model.pick_action(None))