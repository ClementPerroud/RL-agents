import time
import inspect
import threading
import contextvars
from dataclasses import dataclass, field
from functools import wraps
from typing import Callable, Dict, Optional, Any

# ---------- per-task call stack (works for threads & asyncio tasks) ----------
_call_stack_cv: contextvars.ContextVar[list] = contextvars.ContextVar(
    "instrument_call_stack", default=None
)

@dataclass
class _Frame:
    owner: type            # class that owns the method
    name: str              # method name
    start: float           # start time (clock)
    child_time: float = 0  # accumulated inclusive time of direct children


# ---------- tree node structure ----------------------------------------------
@dataclass
class _CallNode:
    name: str
    calls: int = 0
    total_time: float = 0.0   # inclusive
    self_time: float = 0.0    # exclusive
    children: Dict[str, "_CallNode"] = field(default_factory=dict)

    def child(self, name: str) -> "_CallNode":
        node = self.children.get(name)
        if node is None:
            node = self.children[name] = _CallNode(name)
        return node

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "total_time": self.total_time,
            "self_time": self.self_time,
            "avg_total": (self.total_time / self.calls) if self.calls else 0.0,
            "avg_self": (self.self_time / self.calls) if self.calls else 0.0,
            "children": {k: v.to_dict() for k, v in self.children.items()},
        }

    def reset(self) -> None:
        self.calls = 0
        self.total_time = 0.0
        self.self_time = 0.0
        for ch in self.children.values():
            ch.reset()


# ---------- flat metrics container (kept from your version) -------------------
@dataclass
class MethodMetric:
    calls: int = 0
    total_time: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def avg_time(self) -> float:
        return self.total_time / self.calls if self.calls else 0.0

    def update(self, dt: float) -> None:
        with self._lock:
            self.calls += 1
            self.total_time += dt


# ---------- decorator ---------------------------------------------------------
def instrument_methods(
    cls=None,
    *,
    include_private: bool = False,          # include names starting with "_"
    methods: Optional[set[str]] = None,     # only these names if provided
    exclude: Optional[set[str]] = None,     # skip these names
    clock: Callable[[], float] = time.perf_counter,
):
    """
    Class decorator that measures:
      • Flat per-method metrics (calls, total_time, avg_time)
      • A hierarchical call tree (inclusive/exclusive times) by inner call stack
    Works for instance methods, @classmethod, @staticmethod, sync & async.
    """

    def _decorate(_cls):
        # flat metrics
        flat_metrics: Dict[str, MethodMetric] = {}
        setattr(_cls, "__metrics__", flat_metrics)

        # tree metrics
        root = _CallNode("<ROOT>")
        tree_lock = threading.Lock()
        setattr(_cls, "__metrics_tree_root__", root)
        setattr(_cls, "__metrics_tree_lock__", tree_lock)

        # ---- helpers exposed on the class -----------------------------------
        @classmethod
        def get_metrics(cls) -> dict:
            return {
                name: {"calls": m.calls, "total_time": m.total_time, "avg_time": m.avg_time}
                for name, m in cls.__metrics__.items()
            }

        @classmethod
        def reset_metrics(cls, names: Optional[set[str]] = None) -> None:
            target = names or set(cls.__metrics__.keys())
            for n in target:
                if n in cls.__metrics__:
                    mm = cls.__metrics__[n]
                    with mm._lock:
                        mm.calls = 0
                        mm.total_time = 0.0

        @classmethod
        def get_call_tree(cls) -> dict:
            with cls.__metrics_tree_lock__:
                # return a deep snapshot
                return {k: v.to_dict() for k, v in cls.__metrics_tree_root__.children.items()}

        @classmethod
        def reset_call_tree(cls) -> None:
            with cls.__metrics_tree_lock__:
                cls.__metrics_tree_root__.reset()
                # Keep children structure to preserve shape, or wipe it:
                # cls.__metrics_tree_root__.children.clear()

        setattr(_cls, "get_metrics", get_metrics)
        setattr(_cls, "reset_metrics", reset_metrics)
        setattr(_cls, "get_call_tree", get_call_tree)
        setattr(_cls, "reset_call_tree", reset_call_tree)

        # ---- selection predicates -------------------------------------------
        def want(name: str) -> bool:
            if exclude and name in exclude:
                return False
            if methods is not None and name not in methods:
                return False
            if not include_private and name.startswith("_"):
                return False
            return True

        # ---- core wrapping ---------------------------------------------------
        def wrap_callable(fn, name: str):
            flat_metrics.setdefault(name, MethodMetric())

            def _enter_frame():
                stack = _call_stack_cv.get()
                if stack is None:
                    stack = []
                    _call_stack_cv.set(stack)
                frame = _Frame(owner=_cls, name=name, start=clock(), child_time=0.0)
                stack.append(frame)
                return frame, stack

            def _exit_frame(frame, stack, dt: float):
                # account child time into parent, for exclusive timing
                stack.pop()
                if stack:
                    stack[-1].child_time += dt

                self_time = dt - frame.child_time

                # flat metrics
                flat_metrics[name].update(dt)

                # build the path of *this class's* frames in the stack (bottom → top),
                # including the current frame we just exited.
                path_names = [f.name for f in stack if f.owner is _cls] + [frame.name]

                # update tree under class root
                with tree_lock:
                    node = root
                    for nm in path_names:
                        node = node.child(nm)
                    node.calls += 1
                    node.total_time += dt
                    node.self_time += self_time

            if inspect.iscoroutinefunction(fn):
                @wraps(fn)
                async def wrapper(*args, **kwargs):
                    frame, stack = _enter_frame()
                    try:
                        return await fn(*args, **kwargs)
                    finally:
                        dt = clock() - frame.start
                        _exit_frame(frame, stack, dt)
                return wrapper
            else:
                @wraps(fn)
                def wrapper(*args, **kwargs):
                    frame, stack = _enter_frame()
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        dt = clock() - frame.start
                        _exit_frame(frame, stack, dt)
                return wrapper

        # instrument only attributes defined on this class
        for name, attr in list(_cls.__dict__.items()):
            if not want(name):
                continue

            if inspect.isfunction(attr):
                setattr(_cls, name, wrap_callable(attr, name))
            elif isinstance(attr, staticmethod):
                fn = attr.__func__
                setattr(_cls, name, staticmethod(wrap_callable(fn, name)))
            elif isinstance(attr, classmethod):
                fn = attr.__func__
                setattr(_cls, name, classmethod(wrap_callable(fn, name)))
            # properties/others are ignored intentionally

        return _cls

    return _decorate if cls is None else _decorate(cls)


# --------------------------- Example -----------------------------------------
if __name__ == "__main__":
    @instrument_methods  # or: @instrument_methods(include_private=True)
    class Demo:
        def a(self, n=20000):
            s = 0
            for _ in range(3):
                s += self.b(n // 2)
            return s

        def b(self, n=10000):
            return self.c(n) + self.c(n)

        @staticmethod
        def c(n=8000):
            x = 0
            for i in range(n):
                x += i
            return x

        @classmethod
        def d(cls, n=5000):
            return cls.c(n)

    d = Demo()
    for _ in range(5):
        d.a(24000)
        Demo.d(7000)

    # Flat metrics
    print("FLAT:", Demo.get_metrics())

    # Tree metrics
    from pprint import pprint
    print("\nTREE:")
    pprint(Demo.get_call_tree())
