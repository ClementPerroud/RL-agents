from rl_agents.utils.dispatch import dispatch
from rl_agents.utils.distribution.atom_config import AtomConfig, BaseAtomConfig, LinearAtomConfig

import torch
import warnings
import dataclasses
import contextlib
from contextvars import ContextVar
from functools import wraps
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from rl_agents.memory.memory import MemoryField



_DIST_MODE = ContextVar("dist_mode", default=False)
_DIST_DEBUG_MODE = ContextVar("dist_debug_mode", default=False)

@contextlib.contextmanager
def distribution_mode(mode: bool):
    token = _DIST_MODE.set(mode)
    try:
        yield
    finally:
        _DIST_MODE.reset(token)

def distribution_aware(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with distribution_mode(mode=True):
            return fn(*args, **kwargs)
    return wrapper

@contextlib.contextmanager
def debug_mode():
    _DIST_DEBUG_MODE.set(True)
    try:
        yield
    finally:
        _DIST_DEBUG_MODE.set(False)

def debug_dispatch(fn):
    @wraps(fn)
    def wrapper(cls, func, types, args=(), kwargs=None):
        if _DIST_DEBUG_MODE.get():
            print(" ")
            print("====================================================================")
            print("Call : ", func)
            for i, a in enumerate(args):
                print(f"Input (arg {i}): ",a)
            if kwargs is None: kwargs = {}
            for key, a in kwargs.items():
                print(f"Input (kwarg {key}): ",a)

        _result =  fn(cls, func, types, args=args, kwargs=kwargs)
    
        if _DIST_DEBUG_MODE.get():
            print("Result : ", _result)
        return _result
    return wrapper

@dataclasses.dataclass
class DistSpec:
    atom_config: AtomConfig
    keep_config: bool
    _initial_atom_config: AtomConfig

    def create_distribution(self, data) -> "Distribution":
        """Create a new distribution with the same parameters that the original distribution"""
        return Distribution(data=data, _spec = self)

class Distribution(torch.Tensor):
    _spec : DistSpec
    DISPATCH_MAPPING = {}
    # -------- Construction / metadata --------
    def __new__(cls, data, atom_config : AtomConfig = None, *, keep_config=False, _initial_atom_config=None, _spec = None,**kwargs):

        if isinstance(data, Distribution): base = data.to_tensor(**kwargs)
        elif not isinstance(data, torch.Tensor): base = torch.tensor(data, **kwargs)
        else: base = data

        
        obj = torch.Tensor.as_subclass(base, cls)

        if _spec:   obj._spec = _spec
        else:       obj._spec = DistSpec(atom_config=atom_config, keep_config=keep_config, _initial_atom_config=atom_config)

        if base.size(-1) != obj._spec.atom_config.nb_atoms:
            warnings.warn(f"Distribution automatically converted to a Tensor : last dim ({data.size(-1)}) != atom_config.nb_atoms ({obj.atom_config.nb_atoms}).")
            return base
        
        return obj

    def __repr__(self):
        base = self.to_tensor()
        return f"Distribution{str.capitalize(base.__repr__())[:-1]}, atom_config={self._spec.atom_config}, keep_config={self._spec.keep_config})"
    
    @property
    def atom_config(self): return self._spec.atom_config
    
    def to_tensor(self, **kwargs):
        with torch._C._DisableTorchDispatch():
            return self.as_subclass(torch.Tensor, **kwargs)
        


    def expectation(self) -> torch.Tensor:
        atoms = self._spec.atom_config.get_atoms(dtype=self.dtype, device=self.device)
        return (self.to_tensor() * atoms).sum(-1)
    
    def normalize(self) -> "Distribution":
        t = self.to_tensor()
        return self._spec.create_distribution(data=t / t.sum(-1, keepdim=True))


    def project_on(self, atom_config : AtomConfig, **kwargs):
        if atom_config != self._spec.atom_config:
            proj = atom_config.projection_distribution(self)  # expects Distribution or Tensor
            spec = dataclasses.replace(self._spec, atom_config = atom_config)
            return spec.create_distribution(data=proj, **kwargs)
        return self._spec.create_distribution(data = self.to_tensor())
    
    def _set_initial_atom_config(self):
        """In-place operation: reset AtomConfig to initial value and project data."""
        if self._spec._initial_atom_config != self._spec.atom_config:
            proj = self._spec._initial_atom_config.projection_distribution(self)
            # mutate in place (same storage) to keep autograd references consistent
            with torch._C._DisableTorchDispatch():
                self.to_tensor().copy_(proj)
            self._spec.atom_config = self._spec._initial_atom_config
        return self
    
    def _post_process_return(self, data: torch.Tensor, **kwargs):
        """If keep_config is True and atom_config changed, project back to self._spec.atom_config."""
        spec = dataclasses.replace(self._spec, **kwargs)
        not_projected_dist = spec.create_distribution(data=data)
        if self._spec.keep_config and spec.atom_config != self._spec.atom_config:
            return not_projected_dist.project_on(atom_config=self._spec.atom_config)
        return not_projected_dist
        

    @dispatch(torch.ops.aten.neg, torch.ops.aten.neg_)
    @staticmethod
    def _neg(input : "Distribution", **kwargs): # Called when : other (any) - input (Distribution)
        return input._post_process_return(data=-input.to_tensor())
    
    @dispatch(torch.ops.aten.rsub)
    @staticmethod
    def _rsub(input : Any, other : Any, **kwargs): # Called when : other (any) - input (Distribution) 
        return Distribution._add(input=input.__neg__(), other=other, **kwargs)

    @dispatch(torch.ops.aten.sub, torch.ops.aten.sub_)
    @staticmethod
    def _sub(input : Any, other : Any, **kwargs):
        return Distribution._add(input=input, other=other.__neg__(), **kwargs)
    
    @dispatch(torch.ops.aten.add, torch.ops.aten.add_)
    @staticmethod
    def _add(input : Any, other : Any, **kwargs):
        if not isinstance(input, Distribution): return Distribution._add(input=other, other=input, **kwargs)
        input : Distribution

        if isinstance(other, Distribution):
            input._set_initial_atom_config()
            if other._spec.atom_config != input._spec.atom_config: 
                other_data_projected  = input._spec.atom_config.projection_distribution(other)
            else: other_data_projected  = other.to_tensor()
            new_data = other_data_projected + input.to_tensor()
            return input._post_process_return(data=new_data)
        
        if isinstance(other, torch.Tensor):
            # If `other` already has the atom dimension, treat as elementwise add on data
            if other.ndim > 0 and other.size(-1) == input._spec.atom_config.nb_atoms:
                new_data = input.to_tensor() + other
                return input._post_process_return(data=new_data)
            
            full_atoms = input._spec.atom_config.get_atoms(dtype = input.dtype, device = input.device).broadcast_to(input.shape)
            other = other.unsqueeze(-1)
            new_full_atoms = full_atoms + other
            new_atom_config = BaseAtomConfig(nb_atoms= input._spec.atom_config.nb_atoms, atoms = new_full_atoms)
            return input._post_process_return(data=input.to_tensor(), atom_config=new_atom_config)
        
        if isinstance(other, float) or isinstance(other, int):
            return input._post_process_return(data=input.to_tensor(), atom_config=input._spec.atom_config + float(other))

        return NotImplemented

    @dispatch(torch.ops.aten.div, torch.ops.aten.div_)
    @staticmethod
    def _truediv(input : Any, other : Any, **kwargs):
        return Distribution._mul(input=input, other=other**(-1))

    @dispatch(torch.ops.aten.mul, torch.ops.aten.mul_)
    @staticmethod
    def _mul(input : Any, other : Any, **kwargs):
        if not isinstance(input, Distribution): return Distribution._mul(input=other, other=input)
        input : Distribution

        if isinstance(other, Distribution):
            input._set_initial_atom_config()
            if other._spec.atom_config != input._spec.atom_config: 
                other_data_projected  = input._spec.atom_config.projection_distribution(other)
            else: other_data_projected  = other.to_tensor()
            new_data = other_data_projected * input.to_tensor()
            return input._post_process_return(data=new_data)
        
        elif isinstance(other, torch.Tensor):
            # If `other` already has the atom dimension, treat as elementwise add on data
            if other.ndim > 0 and other.size(-1) == input._spec.atom_config.nb_atoms:
                new_data = input.to_tensor() * other
                return input._post_process_return(data=new_data)
            
            full_atoms = input._spec.atom_config.get_atoms(dtype = input.dtype, device = input.device).broadcast_to(input.shape)
            if other.ndim == 0 or other.size(-1) != input._spec.atom_config.nb_atoms:
                other = other.unsqueeze(-1)
            new_full_atoms = full_atoms * other
            new_atom_config = BaseAtomConfig(nb_atoms= input._spec.atom_config.nb_atoms, atoms = new_full_atoms)
            return input._post_process_return(data=input.to_tensor(), atom_config=new_atom_config)
        
        if isinstance(other, float) or isinstance(other, int):
            return input._post_process_return(data=input.to_tensor(), atom_config=input._spec.atom_config * float(other))
        return NotImplemented

    @dispatch(torch.ops.aten.mean)
    @staticmethod
    def _mean(input : 'Distribution', **kwargs):
        return torch.mean(input=input.expectation(), **kwargs)


    @dispatch(torch.ops.aten.std)
    @staticmethod
    def _std(input : 'Distribution', **kwargs):
        return torch.std(input=input.expectation(), **kwargs)

    @staticmethod
    def _dimension_reduction(func, input : 'Distribution', dim = None, *args, **kwargs):
        last_dim = input.ndim-1
        if dim is None:
            dim = list(range(0,last_dim))
        elif isinstance(dim, tuple) or isinstance(dim, list):
            dim = list(dim)
            for i, d in enumerate(dim):
                if d < 0: d = last_dim + d
                if d < 0 or last_dim <= d: raise ValueError(f"Dimension {dim[i]} is invalid. Dimensions must be between 0 (included) and {last_dim} (included ; atom dim can not be reduced).")
                dim[i] = d
        elif isinstance(dim, int):
            if dim < 0 or last_dim <= dim: raise ValueError(f"Dimension {dim[i]} is invalid. Dimensions must be between 0 (included) and {last_dim} (included ; atom dim can not be reduced).")
    
        return input._post_process_return(
            data = func(input.to_tensor(), dim = dim, *args, **kwargs)
        )

    SHAPE_OR_MUT_OPS = {
        # torch.ops.aten.view,
        # torch.ops.aten.reshape,
        # torch.ops.aten._unsafe_view,
        # torch.ops.aten.expand,
        # torch.ops.aten.permute,
        # torch.ops.aten.transpose,
        # torch.ops.aten.unsqueeze,
        # torch.ops.aten.squeeze,
        # torch.ops.aten.copy_,
        # torch.ops.aten.index_put,
        # torch.ops.aten.slice_scatter,
        # torch.ops.aten.scatter,
        # torch.ops.aten.scatter_,
        # add others you hit in practice
    }

    @classmethod
    @debug_dispatch
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if kwargs is None: kwargs = {}

        base_func = func._overloadpacket
        mode_dist = _DIST_MODE.get()
        if mode_dist: # mode_dist ACTIVED
            if base_func in cls.DISPATCH_MAPPING:
                replacing_func = cls.DISPATCH_MAPPING[base_func]
                return replacing_func(*args, **kwargs)
            
            # Func not overrided :
            # We use the Tensor method by :
            #  1 - converting distribution to Tensor
            #  2 - using the original method
            #  3 - rewrapping the returned Tensor into a Distribution
            args = list(args)
            _spec = None
            for i, arg in enumerate(args): 
                if isinstance(arg, Distribution):
                    args[i] = arg.to_tensor()
                    if _spec is None: _spec=arg._spec
            for key, arg in kwargs.items():
                if isinstance(arg, Distribution):
                    kwargs[key] = arg.to_tensor()
                    if _spec is None: _spec=arg._spec
            
            returned_tensor = func(*args, **kwargs)
            return _spec.create_distribution(data=returned_tensor)
        
        else:
            # mode_dist DESACTIVED : We convert distribution to their expectation, and go back working with Tensor.
            # if base_func in cls.SHAPE_OR_MUT_OPS:
            #     args2 = [a.to_tensor() if isinstance(a, Distribution) else a for a in args]
            #     kwargs2 = {k: (v.to_tensor() if isinstance(v, Distribution) else v) for k, v in kwargs.items()}
            #     return func(*args2, **kwargs2)

            args = list(args)
            for i, arg in enumerate(args): 
                if isinstance(arg, Distribution):
                    args[i] = arg.expectation()
            for key, arg in kwargs.items():
                if isinstance(arg, Distribution):
                    kwargs[key] = arg.expectation()
            return func(*args, **kwargs)
    
    def get_codec(self):
        self._set_initial_atom_config()
        return DistributionCodec(spec = self._spec)


class DistributionCodec:
    tensor_class = Distribution
    def __init__(self, spec: DistSpec):
        self.spec = spec

    def _uniform_vec(self, dtype, device) -> torch.Tensor:
        A = self.spec.atom_config.nb_atoms
        return torch.full((A,), 1.0 / A, dtype=dtype, device=device)

    def allocate(self, size, field : "MemoryField", device):
        # ignore `default`; use spec.init
        return  torch.zeros(size, dtype=field.dtype, device=device)

    def reset_fill(self, buf, field : "MemoryField"):
        buf.zero_()

    def encode(self, value, field : "MemoryField", device):
        # Accept Distribution or raw tensor of probs with last dim = A
        if isinstance(value, Distribution):
            v = value
            if v.atom_config != self.spec.atom_config:
                v = v.project_on(self.spec.atom_config)
            return v.to_tensor().to(dtype=field.dtype, device=device)
        # raw tensor
        v = torch.as_tensor(value, dtype=field.dtype, device=device)
        assert v.size(-1) == self.spec.atom_config.nb_atoms, f"{field.name}: last dim {v.size(-1)} != A={self.spec.atom_config.nb_atoms}"
        return v

    def decode(self, tensor, field : "MemoryField"):
        # Wrap back as Distribution for consumers
        return Distribution(tensor, atom_config=self.spec.atom_config, keep_config=self.spec.keep_config)
    
def main():
    _DIST_MODE.set(True)
    t1 = Distribution([[0.1, 0.2, 0.4, 0.2, 0.1]], atom_config=LinearAtomConfig(v_min = -10, v_max = 10, nb_atoms=5), requires_grad = True)
    t2 = Distribution([[0.2, 0.2, 0.2, 0.2, 0.2]], atom_config=LinearAtomConfig(v_min = -10, v_max = 10, nb_atoms=5), requires_grad = True)
    print(t1.shape)
    t3 = t1.unsqueeze(0).tile(6,10,1)
    print(1, t1 + t2)
    print(2, t2 + 1 )
    print(2.5, 1 + t2 )
    print(3, t2 - 1)
    # print(3.5, 1 - t2)
    print(4, t2 * 10)
    print(5, 10 * t2)
    print(6, (t2 / 2))
    print(7, t1.unsqueeze(1).shape)
    loss = 2 * t1 + 3 * t2
    loss = loss.expectation().mean()
    loss.backward()
    print("Grad t1 : ", t1.grad)

if __name__ == "__main__":
    main()
