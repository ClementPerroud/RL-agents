from rl_agents.service import AgentService
from rl_agents.utils.noisynet import NoisyLinear, NoisyLazyLinear
from rl_agents.utils.mode import train_mode
import torch
import torch.nn as nn

class NoisyNetTransformer(AgentService):
    """
    In-place wrapper: replaces all nn.Linear layers in q_net by torchrl.modules.NoisyLinear.
    Keeps device/dtype and training mode; does NOT transfer weights (fresh noisy layers).
    """
    def __init__(self, std_init: float = 0.1):
        super().__init__()
        self.std_init = std_init

    def __call__(self, module: AgentService):
        self._patch(module=module)
        return module

    def _patch(self, module: nn.Module):
        for name, child in list(module.named_children()):
            print(child)
            if isinstance(child, nn.LazyLinear) and not isinstance(child, NoisyLazyLinear):
                noisy = NoisyLazyLinear(
                    out_features=child.out_features,
                    bias=(child.bias is not None),
                    std_init=self.std_init,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                noisy.train(child.training)
                setattr(module, name, noisy)
            elif isinstance(child, nn.Linear) and not isinstance(child, NoisyLinear):
                noisy = NoisyLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=(child.bias is not None),
                    std_init=self.std_init,
                    device=child.weight.device,
                    dtype=child.weight.dtype,
                )
                noisy.train(child.training)
                setattr(module, name, noisy)
            else:
                self._patch(child)