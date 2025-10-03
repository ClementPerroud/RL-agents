from rl_agents.service import AgentService

import torch

class VWrapper(AgentService):
    def __init__(self, core_net : torch.nn.Module):
        super().__init__()
        self.core_net = core_net
        self.head = torch.nn.LazyLinear(1)

    def V(self, state : torch.Tensor) -> torch.Tensor:
        x = self.core_net(state)
        return self.head(x).squeeze(1)