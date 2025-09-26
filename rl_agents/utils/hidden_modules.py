from __future__ import annotations
import weakref
import torch.nn as nn

class HiddenModulesUtilsMixin:
    """Provite the attribute hidden that contain reference to Module (or other) that will be hidden from torch modules."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden = _HiddenModulesContainer()
    

class _HiddenModulesContainer:
    def __getattr__(self, name : str):
        slot = f"__hiddenref__{name}"
        value = getattr(self, slot, None)
        return value
        
    def __setattr__(self, name, value):
        object.__setattr__(self, f"__hiddenref__{name}", value)