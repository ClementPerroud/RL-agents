import copy
import torch
import torch.nn as nn

class NoisyNetStrategy(nn.Module):
    """
    Wrap a model and replace all nn.Linear layers by torchrl.modules.NoisyLinear.

    Args:
        module: the base model to convert (deep-copied by default)
        std_init: TorchRL NoisyLinear initial std (default 0.1)
        auto_sample: if True, resample noise before every forward when training
        copy: deep-copy the incoming module before modifying (default True)
        include_last: if False, keeps the very last Linear deterministic
    """
    def __init__(self,
                 module: nn.Module,
                 std_init: float = 0.1,
                 auto_sample: bool = True,
                 copy: bool = True,
                 include_last: bool = True):
        super().__init__()
        try:
            from torchrl.modules import NoisyLinear  # noqa: F401
        except Exception as e:
            raise ImportError(
                "TorchRL is required. Install with: pip install torchrl"
            ) from e

        self.model = copy.deepcopy(module) if copy else module
        self._handles = []
        self._NoisyLinear = __import__("torchrl.modules", fromlist=["NoisyLinear"]).NoisyLinear

        self._replace_linears(std_init=std_init, include_last=include_last)

        # Auto-resample via forward_pre_hook
        if auto_sample:
            for m in self.model.modules():
                if isinstance(m, self._NoisyLinear):
                    h = m.register_forward_pre_hook(lambda mod, _: mod.reset_noise() if mod.training else None)
                    self._handles.append(h)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def reset_noise(self):
        """Resample noise on all NoisyLinear layers."""
        for m in self.model.modules():
            if isinstance(m, self._NoisyLinear):
                m.reset_noise()

    # Convenience: expose wrapped attributes
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def __del__(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass

    # ---- internals ----
    def _replace_linears(self, std_init: float, include_last: bool):
        linear_nodes = []
        def collect(parent):
            for name, child in parent.named_children():
                if isinstance(child, nn.Linear):
                    linear_nodes.append((parent, name, child))
                collect(child)
        collect(self.model)

        keep_idx = len(linear_nodes) - 1 if (not include_last and len(linear_nodes) > 0) else -1

        for i, (parent, name, lin) in enumerate(linear_nodes):
            if i == keep_idx:
                continue
            noisy = self._NoisyLinear(
                lin.in_features, lin.out_features, bias=(lin.bias is not None), std_init=std_init
            )
            with torch.no_grad():
                noisy.weight_mu.copy_(lin.weight.detach())
                if lin.bias is not None:
                    noisy.bias_mu.copy_(lin.bias.detach())
            setattr(parent, name, noisy)