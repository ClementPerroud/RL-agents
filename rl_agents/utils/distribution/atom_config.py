from torch.overrides import resolve_name
from rl_agents.utils.class_analysis import instrument_methods
from rl_agents.utils.dispatch import dispatch

import sys
import torch
import warnings
import dataclasses
from typing import Protocol, runtime_checkable, Any, Self, TYPE_CHECKING
import contextlib
from contextvars import ContextVar
from functools import wraps

if TYPE_CHECKING:
    from rl_agents.utils.distribution.distribution import Distribution

@runtime_checkable
class AtomConfig(Protocol):
    nb_atoms : int
    def get_atoms(self, **kwargs) -> torch.Tensor: ...
    def projection_distribution(self, dist : "Distribution") -> torch.Tensor: ...

    @staticmethod
    def are_compatible(config1 : "AtomConfig", config2 : "AtomConfig"):
        return config1.nb_atoms == config2.nb_atoms


class BaseAtomConfig(AtomConfig):
    def __init__(self, nb_atoms : int, atoms : torch.Tensor):
        self.nb_atoms = int(nb_atoms)
        self.atoms = atoms
        atoms.requires_grad_(False)

    def __repr__(self): return f"BaseAtomConfig([{self.atoms.min()};{self.atoms.max()}]; {self.nb_atoms} atoms; shape : {self.atoms.shape})"
    def get_atoms(self, **kwargs): return self.atoms
    def __add__(self, other : float | int): return BaseAtomConfig(nb_atoms=self.nb_atoms, atoms = self.atoms + other)
    def __mul__(self, other : float | int): return BaseAtomConfig(nb_atoms=self.nb_atoms, atoms = self.atoms * other)
    def __eq__(self, config : Self):
        if not isinstance(config, AtomConfig): return False
        return (config.get_atoms() == self.get_atoms()).all()
    
    def projection_distribution(self, dist):
        return NotImplemented


class LinearAtomConfig(BaseAtomConfig):
    def __init__(self, nb_atoms : int, v_min : float, v_max : float, tensor_kwargs = {}):
        self.nb_atoms = int(nb_atoms)
        self.v_min, self.v_max = float(v_min), float(v_max)
        self._delta_atoms = (self.v_max - self.v_min) / (self.nb_atoms - 1)
        self.tensor_kwargs = tensor_kwargs
        self._atoms = None
    def __repr__(self): return f"LinearAtomConfig([{self.v_min};{self.v_max}]; {self.nb_atoms} atoms)"
    def __add__(self, other : float | int): return LinearAtomConfig(nb_atoms=self.nb_atoms, v_min=self.v_min + other, v_max=self.v_max + other)
    def __mul__(self, other : float | int): return LinearAtomConfig(nb_atoms=self.nb_atoms, v_min=self.v_min * other, v_max=self.v_max * other)

    def __eq__(self, config : Self):
        if not isinstance(config, AtomConfig): return False
        if isinstance(config, LinearAtomConfig): return self.nb_atoms == config.nb_atoms and self.v_min == config.v_min and self.v_max == config.v_max
        return super().__eq__(config=config)

    def get_atoms(self, **kwargs):
        if self._atoms is None: self._atoms = torch.linspace(start=self.v_min, end=self.v_max, steps=self.nb_atoms, requires_grad=False, **kwargs)
        return self._atoms
    
    def projection_distribution(
            self,
            dist : "Distribution"
        ) -> torch.Tensor:
        assert AtomConfig.are_compatible(dist.atom_config, self), "Config dist.atom_config is not comptatible with the config you try to project on."

        T_z = dist.atom_config.get_atoms(dtype=dist.dtype, device=dist.device).clamp(min=self.v_min, max=self.v_max)  # [..., A]
        b = (T_z - self.v_min) / self._delta_atoms  # [..., A]  (dtype=self.dtype)

        # Broadcast p and b to a common shape
        target_shape = torch.broadcast_shapes(dist.shape, b.shape)
        b = b.expand(target_shape).to(dtype=dist.dtype)
        p = dist.to_tensor().expand(target_shape).to(dtype=dist.dtype)

        l = torch.floor(b).to(torch.long)
        u = l + 1

        m = torch.zeros(target_shape, dtype=p.dtype, device=p.device)

        m.scatter_add_(-1, l.clamp_max(self.nb_atoms - 1), p * (u.to(p.dtype) - b))
        m.scatter_add_(-1, u.clamp_max(self.nb_atoms - 1), p * (b - l.to(p.dtype)))
        return m
