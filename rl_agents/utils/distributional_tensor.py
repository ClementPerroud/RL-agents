import sys
sys.setrecursionlimit(200)
import torch
from torch.overrides import resolve_name
from rl_agents.utils.class_analysis import instrument_methods
from rl_agents.utils.dispatch import dispatch

from typing import Protocol, runtime_checkable, Any, Self

@runtime_checkable
class AtomConfig(Protocol):
    nb_atoms : int
    def get_atoms(self, **kwargs) -> torch.Tensor: ...
    def projection_distribution(self, dist : "Distribution") -> torch.Tensor: ...

    @staticmethod
    def are_compatible(config1 : Self, config2 : Self):
        return config1.nb_atoms == config2.nb_atoms

@instrument_methods
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

@instrument_methods
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

@instrument_methods
class Distribution(torch.Tensor):
    # -------- Construction / metadata --------
    def __new__(cls, data, atom_config : AtomConfig, *, keep_config=False, _initial_atom_config=None, **kwargs):

        if isinstance(data, Distribution): base = data.to_tensor(**kwargs)
        elif not isinstance(data, torch.Tensor): base = torch.tensor(data, **kwargs)
        else: base = data

        # Create the subclass tensor with same requires_grad
        obj = torch.Tensor.as_subclass(base, cls)
        # attach metadata
        obj.atom_config = atom_config
        obj.keep_config = bool(keep_config)
        obj._initial_atom_config = atom_config if _initial_atom_config is None else _initial_atom_config
        # basic checks
        assert obj.size(-1) == obj.atom_config.nb_atoms, (
            f"data last dim ({obj.size(-1)}) must match atom_config.nb_atoms {obj.atom_config.nb_atoms}"
        )
        return obj

    def __repr__(self):
        base = self.to_tensor()
        return f"Distribution({base.__repr__()}, atom_config={self.atom_config}, keep_config={self.keep_config})"
    
    def to_tensor(self, **kwargs):
        with torch._C._DisableTorchDispatch():
            return self.as_subclass(torch.Tensor, **kwargs)
        
    def new_distribution(self, data, atom_config : AtomConfig):
        """Create a new distribution with the same parameters that the original distribution"""
        return Distribution(data=data, atom_config=atom_config, keep_config=self.keep_config, _initial_atom_config = self._initial_atom_config)

    def expectation(self) -> torch.Tensor:
        atoms = self.atom_config.get_atoms(dtype=self.dtype, device=self.device)
        return (self.to_tensor() * atoms).sum(-1)
    
    def project_on(self, atom_config, **kwargs):
        if atom_config != self.atom_config:
            proj = atom_config.projection_distribution(self)  # expects Distribution or Tensor
            return self.new_distribution(data=proj, atom_config=atom_config)
        return self

    def _set_initial_atom_config(self):
        """In-place operation: reset AtomConfig to initial value and project data."""
        if self._initial_atom_config != self.atom_config:
            proj = self._initial_atom_config.projection_distribution(self)
            # mutate in place (same storage) to keep autograd references consistent
            with torch._C._DisableTorchDispatch():
                self.to_tensor().copy_(proj)
            self.atom_config = self._initial_atom_config
        return self
    
    def _post_process_return(self, data: torch.Tensor, atom_config):
        """If keep_config is True and atom_config changed, project back to self.atom_config."""
        if self.keep_config and atom_config != self.atom_config:
            tmp = self.new_distribution(data=data, atom_config=atom_config)
            return tmp.project_on(atom_config=self.atom_config)
        return self.new_distribution(data=data, atom_config=atom_config)
        
    @dispatch(torch.ops.aten.neg.Scalar, torch.ops.aten.rsub.Tensor)
    @staticmethod
    def _neg(input : "Distribution"): # Called when : other (any) - input (Distribution) 
        return input._post_process_return(data=input.to_tensor(), atom_config=-input.atom_config)
    
    @dispatch(torch.ops.aten.rsub.Scalar, torch.ops.aten.rsub.Tensor)
    @staticmethod
    def _rsub(input : Any, other : Any, **kwargs): # Called when : other (any) - input (Distribution) 
        return Distribution._add(input=input.__neg__(), other=other, **kwargs)

    @dispatch(torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar)
    @staticmethod
    def _sub(input : Any, other : Any, **kwargs):
        return Distribution._add(input=input, other=other.__neg__(), **kwargs)
    
    @dispatch(torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar)
    @staticmethod
    def _add(input : Any, other : Any, **kwargs):
        if not isinstance(input, Distribution): return Distribution._add(input=other, other=input, **kwargs)
        input : Distribution

        if isinstance(other, Distribution):
            input._set_initial_atom_config()
            if other.atom_config != input.atom_config: 
                other_data_projected  = input.atom_config.projection_distribution(other)
            else: other_data_projected  = other.to_tensor()
            new_data = other_data_projected + input.to_tensor()
            new_data = new_data / new_data.sum(-1, keepdim=True)
            return input._post_process_return(data=new_data, atom_config=input.atom_config)
        if isinstance(other, torch.Tensor):
            full_atoms = input.atom_config.get_atoms(dtype = input.dtype, device = input.device).broadcast_to(input.shape)
            other = other.unsqueeze(-1)
            new_full_atoms = full_atoms + other
            new_atom_config = BaseAtomConfig(nb_atoms= input.atom_config.nb_atoms, atoms = new_full_atoms)
            return input._post_process_return(data=input.to_tensor(), atom_config=new_atom_config)
        if isinstance(other, float) or isinstance(other, int):
            return input._post_process_return(data=input.to_tensor(), atom_config=input.atom_config + float(other))

        return NotImplemented

    @dispatch(torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar)
    @staticmethod
    def _truediv(input : Any, other : Any):
        return Distribution._mul(input=input, other=1/other)

    @dispatch(torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar)
    @staticmethod
    def _mul(input : Any, other : Any):
        if not isinstance(input, Distribution): return Distribution._mul(input=other, other=input)
        input : Distribution
        if isinstance(other, Distribution):
            return NotImplemented
        elif isinstance(other, torch.Tensor):
            full_atoms = input.atom_config.get_atoms(dtype = input.dtype, device = input.device).broadcast_to(input.shape)
            if other.size(-1) != input.atom_config.nb_atoms:
                other = other.unsqueeze(-1)
            new_full_atoms = full_atoms * other
            new_atom_config = BaseAtomConfig(nb_atoms= input.atom_config.nb_atoms, atoms = new_full_atoms)
            return input._post_process_return(data=input.to_tensor(), atom_config=new_atom_config)
        
        if isinstance(other, float) or isinstance(other, int):
            return input._post_process_return(data=input.to_tensor(), atom_config=input.atom_config * float(other))
        return NotImplemented
    

    VIEW_OPS = {
        torch.ops.aten.view, torch.ops.aten.reshape, torch.ops.aten.unsqueeze, torch.ops.aten.squeeze.dim,
        torch.ops.aten.permute, torch.ops.aten.transpose, torch.ops.aten.contiguous, torch.ops.aten.detach
    }
    COUNT_OPS = {}
    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # print(func)
        if func.__str__() not in cls.COUNT_OPS: cls.COUNT_OPS[func.__str__()] = 0
        cls.COUNT_OPS[func.__str__()] += 1
        
        if kwargs is None: kwargs = {}
        if func in cls.DISPATCH_MAPPING:
            replacing_func = cls.DISPATCH_MAPPING[func]
            return replacing_func(*args, **kwargs)


        # super cheap path for view ops
        if func in cls.VIEW_OPS:
            with torch._C._DisableTorchDispatch():
                out = func(*[a.to_tensor() if isinstance(a, Distribution) else a for a in args], **{
                    k: (v.to_tensor() if isinstance(v, Distribution) else v) for k, v in (kwargs or {}).items()
                })
            # wrap only if last dim still matches
            proto = next((a for a in list(args)+list((kwargs or {}).values()) if isinstance(a, Distribution)), None)
            if isinstance(out, torch.Tensor) and out.ndim >= 1 and out.size(-1) == proto.atom_config.nb_atoms:
                return proto.new_distribution(data=out, atom_config=proto.atom_config)
            return out
    
        args = list(args)
        _proto = None
        for i, arg in enumerate(args): 
            if isinstance(arg, Distribution):
                args[i] = arg.to_tensor()
                if _proto is None: _proto=arg
        for key, arg in kwargs.items():
            if isinstance(arg, Distribution):
                kwargs[key] = arg.to_tensor()
                if _proto is None: _proto=arg
        return _proto.new_distribution(data=func(*args, **kwargs), atom_config=_proto.atom_config)
    
def main():
    t1 = Distribution([[0.1, 0.2, 0.4, 0.2, 0.1]], atom_config=LinearAtomConfig(v_min = -10, v_max = 10, nb_atoms=5), requires_grad = True)
    t2 = Distribution([[0.2, 0.2, 0.2, 0.2, 0.2]], atom_config=LinearAtomConfig(v_min = -10, v_max = 10, nb_atoms=5), requires_grad = True)
    print(t1)
    print(1, t1 + t2)
    print(2, t2 + 1 )
    print(2.5, 1 + t2 )
    print(3, t2 - 1)
    # print(3.5, 1 - t2)
    print(4, t2 * 10)
    print(5, 10 * t2)
    print(6, (t2 / 2)._set_initial_atom_config())
    print(7, t1.unsqueeze(1).shape)
    loss = 2 * t1 + 3 * t2
    loss = loss.expectation().mean()
    loss.backward()
    print("Grad t1 : ", t1.grad)

# class DistributionalTensor(torch.Tensor):
#     # -------- Construction / metadata --------
#     def __new__(cls, data, *, nb_atoms: int, v_min: float, v_max: float, requires_grad=None):
#         base = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
#         with torch.no_grad():
#             s = base.sum(-1)
#             assert torch.allclose(s, torch.ones_like(s)), "Distributions must sum to 1 on the last dim."
#             assert base.size(-1) == nb_atoms, f"nb_atoms={nb_atoms} != last dim={base.size(-1)}"
#         obj = torch.Tensor._make_subclass(cls, base, base.requires_grad if requires_grad is None else requires_grad)
#         # attach metadata
#         obj.nb_atoms = int(nb_atoms)
#         obj.v_min = float(v_min)
#         obj.v_max = float(v_max)
#         obj._delta_atoms = (obj.v_max - obj.v_min) / (obj.nb_atoms - 1)
#         return obj

#     # -------- Pretty print / base view --------
#     def __repr__(self):
#         with torch._C._DisableTorchDispatch():
#             base = torch.Tensor(self)
#         return f"Distributional{str.capitalize(base.__repr__())}"

#     def to_base(self) -> torch.Tensor:
#         """Zero-copy base view (same storage, no dispatch)."""
#         with torch._C._DisableTorchDispatch():
#             return torch.Tensor(self)

#     # -------- Helpers --------
#     def get_atoms(self) -> torch.Tensor:
#         return torch.linspace(
#             start=self.v_min, end=self.v_max, steps=self.nb_atoms,
#             dtype=self.dtype, device=self.device
#         )

#     def expectation(self) -> torch.Tensor:
#         p = self.to_base()
#         with torch._C._DisableTorchDispatch():
#             return (p * self.get_atoms()).sum(dim=-1)

#     def has_same_config(self, other: "DistributionalTensor") -> bool:
#         return (self.nb_atoms == other.nb_atoms) and (self.v_min == other.v_min) and (self.v_max == other.v_max)

#     def project_from(self, p: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
#         if isinstance(p, DistributionalTensor):
#             p = p.to_base()

#         assert atoms.size(-1) == self.nb_atoms, "Projection requires the same nb_atoms."
#         with torch._C._DisableTorchDispatch():
#             T_z = atoms.to(dtype=self.dtype, device=self.device).clamp(min=self.v_min, max=self.v_max)  # [..., A]
#             b = (T_z - self.v_min) / self._delta_atoms  # [..., A]  (dtype=self.dtype)

#             # Broadcast p and b to a common shape
#             target_shape = torch.broadcast_shapes(p.shape, b.shape)
#             b = b.expand(target_shape).to(dtype=self.dtype)
#             p = p.expand(target_shape).to(dtype=self.dtype)

#             l = torch.floor(b).to(torch.long)
#             u = l + 1

#             m = torch.zeros(target_shape, dtype=self.dtype, device=self.device)

#             m.scatter_add_(-1, l.clamp_max(self.nb_atoms - 1), p * (u.to(self.dtype) - b))
#             m.scatter_add_(-1, u.clamp_max(self.nb_atoms - 1), p * (b - l.to(self.dtype)))
#         return m

#     # -------- Custom binary ops (out-of-place) --------
#     @staticmethod
#     def _as_like(x: Any, ref: "DistributionalTensor") -> torch.Tensor:
#         return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)

#     @staticmethod
#     def _apply_inplace(dst: "DistributionalTensor", src_base: torch.Tensor) -> "DistributionalTensor":
#         # Write into dst storage in-place under base dispatcher to honor aliasing contracts.
#         dst.to_base().copy_(src_base)
#         return dst

#     @staticmethod
#     def _add_impl(input: "DistributionalTensor", other: Any, *, inplace: bool):
#         # Ensure 'input' is the DistributionalTensor
#         if isinstance(other, DistributionalTensor) and not isinstance(input, DistributionalTensor):
#             return DistributionalTensor._add_impl(input=other, other=input, inplace=inplace)

#         # Dist + Dist : project 'other' onto input's support, elementwise add, renormalize
#         if isinstance(input, DistributionalTensor) and isinstance(other, DistributionalTensor):
#             other_proj = input.project_from(p=other, atoms=other.get_atoms())  # base tensor
#             base = input.to_base() + other_proj
#             base = base / base.sum(-1, keepdim=True)
#             return DistributionalTensor._apply_inplace(input, base) if inplace else base

#         # Dist + scalar/tensor : shift atoms by 'other', then project
#         if isinstance(input, DistributionalTensor) and not isinstance(other, DistributionalTensor):
#             other = DistributionalTensor._as_like(other, input)
#             new_atoms = input.get_atoms().broadcast_to(other.shape + torch.Size([input.nb_atoms]))
#             new_atoms = new_atoms + other.unsqueeze(-1)
#             base = input.project_from(p=input, atoms=new_atoms)
#             return DistributionalTensor._apply_inplace(input, base) if inplace else base

#         raise TypeError("Unsupported operands for distributional add.")

#     @staticmethod
#     def _sub_impl(input: "DistributionalTensor", other: Any, *, inplace: bool):
#         if isinstance(other, DistributionalTensor):
#             # You could implement Dist - Dist as add with negative mass or raise explicitly.
#             raise TypeError("Distributional subtraction (dist - dist) is undefined in this semantics.")
#         return DistributionalTensor._add_impl(input=input, other=(-other), inplace=inplace)

#     @staticmethod
#     def _mul_impl(input: "DistributionalTensor", other: Any, *, inplace: bool):
#         if isinstance(other, DistributionalTensor) and not isinstance(input, DistributionalTensor):
#             return DistributionalTensor._mul_impl(input=other, other=input, inplace=inplace)
        
#         # Dist * scalar/tensor : scale atoms by 'other', then project
#         if isinstance(input, DistributionalTensor) and not isinstance(other, DistributionalTensor):
#             other = DistributionalTensor._as_like(other, input)
#             if other.size(-1) != input.nb_atoms:
#                 other = other.unsqueeze(-1).broadcast_to(other.shape + torch.Size([input.nb_atoms]))
#             new_atoms = input.get_atoms().broadcast_to(input.shape)
#             new_atoms = new_atoms * other.broadcast_to(input.shape)
#             base = input.project_from(p=input, atoms=new_atoms)
#             return DistributionalTensor._apply_inplace(input, base) if inplace else base

#         raise TypeError("Unsupported operands for distributional mul.")

#     @staticmethod
#     def _div_impl(input: "DistributionalTensor", other: Any, *, inplace: bool):
#         other = other if isinstance(other, torch.Tensor) else float(other)
#         return DistributionalTensor._mul_impl(input=input, other=(1.0 / other), inplace=inplace)

#     # -------- Dispatch table --------
#     MAPPING = {
#         # out-of-place only
#         "aten.add.Tensor":  _add_impl,
#         "aten.add.Scalar":  _add_impl,
#         "aten.sub.Tensor":  _sub_impl,
#         "aten.sub.Scalar":  _sub_impl,
#         "aten.mul.Tensor":  _mul_impl,
#         "aten.mul.Scalar":  _mul_impl,
#         "aten.div.Tensor":  _div_impl,
#         "aten.div.Scalar":  _div_impl,
#     }
#     # -------- __torch_dispatch__ --------
#     @classmethod
#     def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
#         if kwargs is None: kwargs = {}
#         name = resolve_name(func)

#         if name in cls.MAPPING:
#             result = cls.MAPPING[name](*args, **kwargs, inplace=False)
#         else:
#             result = func(*args, **kwargs)

#         # ... rewrap as you already do
#         proto = cls._first_proto(args, kwargs)
#         def wrap(obj):
#             if isinstance(obj, torch.Tensor) and not isinstance(obj, cls):
#                 return cls._wrap_with_meta(obj, proto, cls)
#             if isinstance(obj, (tuple, list)):
#                 return type(obj)(wrap(x) for x in obj)
#             if isinstance(obj, dict):
#                 return {k: wrap(v) for k, v in obj.items()}
#             return obj
#         return wrap(result)

#     # -------- proto & wrap helpers --------
#     @staticmethod
#     def _first_proto(args, kwargs):
#         def it(obj):
#             if isinstance(obj, (tuple, list)):
#                 for x in obj: yield from it(x)
#             elif isinstance(obj, dict):
#                 for v in obj.values(): yield from it(v)
#             else:
#                 yield obj
#         for a in it(args):
#             if isinstance(a, DistributionalTensor):
#                 return a
#         for a in it(kwargs):
#             if isinstance(a, DistributionalTensor):
#                 return a
#         return None

#     @staticmethod
#     def _wrap_with_meta(x: torch.Tensor, proto: "DistributionalTensor", cls):
#         # Only wrap distributions (last dim = nb_atoms)
#         if proto is not None and x.ndim >= 1 and x.size(-1) == proto.nb_atoms:
#             y = x.as_subclass(cls)
#             y.nb_atoms = proto.nb_atoms
#             y.v_min = proto.v_min
#             y.v_max = proto.v_max
#             y._delta_atoms = proto._delta_atoms
#             return y
#         return x


# def main():
#     t1 = DistributionalTensor([[0.1, 0.2, 0.4, 0.2, 0.1]], v_min = -10, v_max = 10, nb_atoms=5)
#     t2 = DistributionalTensor([[0.2, 0.2, 0.2, 0.2, 0.2]], v_min = -10, v_max = 10, nb_atoms=5)
#     print(t2 + 1 )
#     print(t2 - 1)
#     print(t2 * 10)
#     print(10 * t2)
#     print(t2 / 2)

#     # Autograd test
#     p = DistributionalTensor([[0.1,0.2,0.4,0.2,0.1]], nb_atoms=5, v_min=-10, v_max=10, requires_grad=True)
#     q = DistributionalTensor([[0.2,0.2,0.2,0.2,0.2]], nb_atoms=5, v_min=-10, v_max=10, requires_grad=True)
#     s = torch.tensor(1.0, requires_grad=True, device=p.device)
#     k = torch.tensor(2.0, requires_grad=True, device=p.device)

#     loss = ( (p + q).expectation() + (p + s).expectation() + (p * k).expectation() ).sum()
#     loss.backward()
#     assert p.grad is not None and q.grad is not None and s.grad is not None and k.grad is not None
if __name__ == "__main__":
    main()
