import numpy as np
from collections.abc import Iterable


epsilon = 1e-6  # Small constant to avoid zero‑probability issues

class SumTree:
    """Binary SumTree for Prioritized Experience Replay (PER).

    Each leaf stores a *non‑negative* priority p_i. Internal nodes store the sum
    of their two children. Thanks to this property, we can sample a leaf in
    O(log N) via a cumulative‑sum lookup.

    Parameters
    ----------
    size : int
        Maximum number of elements that can be stored (capacity). When the tree
        is full, `add()` overwrites in circular fashion (FIFO).
    """

    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("size must be a strictly positive integer")

        self.size = int(size)
        self.length = 0          # Number of elements actually inserted

        # ----- Build layered array of sums (bottom → top) -----
        self.node_layers: list[np.ndarray] = [np.zeros(size, dtype=np.float32)]
        layer_size = size
        while layer_size > 1:
            layer_size = (layer_size + 1) // 2
            self.node_layers.append(np.zeros(layer_size, dtype=np.float32))

        self.n_layers = len(self.node_layers)

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _propagate(self, leaf_idx: int, change: float) -> None:
        """Propagate *change* from a leaf up to the root."""
        for layer in range(1, self.n_layers):
            leaf_idx //= 2
            self.node_layers[layer][leaf_idx] += change

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def add(self, value: float | Iterable[float]):
        """Insert a priority (or a batch of priorities) at the next position.

        If the capacity is reached, we overwrite in FIFO order.
        """
        if isinstance(value, Iterable):
            for v in value:
                self.add(v)
            return

        v = float(max(value, epsilon))  # Clamp to strictly positive
        leaf_idx = self.length % self.size
        self.length += 1

        change = v - self.node_layers[0][leaf_idx]
        self.node_layers[0][leaf_idx] = v
        self._propagate(leaf_idx, change)

    def __setitem__(self, idx: int | np.ndarray, value):
        """Update one or several priorities in‑place."""
        if isinstance(idx, (list, np.ndarray)):
            idx = np.asarray(idx, dtype=int)
            value = np.asarray(value, dtype=np.float32)
            if idx.shape != value.shape:
                raise ValueError("index and value must have the same shape")
            for i, v in zip(idx, value):
                self.__setitem__(int(i), float(v))
            return

        if idx >= self.length:
            raise IndexError(
                f"Cannot set value at index {idx} (current length = {self.length})")

        v = float(max(value, epsilon))
        change = v - self.node_layers[0][idx]
        self.node_layers[0][idx] = v
        self._propagate(idx, change)

    def __getitem__(self, idx: int | np.ndarray):
        return self.node_layers[0][idx]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sum(self) -> float:
        """Return the total priority (value stored in the root)."""
        return float(self.node_layers[-1][0])

    def sample(self, batch_size: int) -> list[int]:
        """Sample *batch_size* leaf indices proportionally to their priorities.
        Raises ValueError if the tree is empty.
        """
        if self.length == 0:
            raise ValueError("Cannot sample from an empty SumTree")

        total = self.sum()
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError(f"Invalid total priority: {total}")

        cumsums = np.random.rand(batch_size) * (total - epsilon)
        indices: list[int] = []

        for cs in cumsums:
            idx = 0  # Start from the root.
            for layer in range(self.n_layers - 1, 0, -1):
                left_idx = idx * 2
                left_sum = self.node_layers[layer - 1][left_idx]
                if cs < left_sum:
                    idx = left_idx
                else:
                    idx = left_idx + 1
                    cs -= left_sum
            # `idx` is now a leaf index.
            if idx >= self.length:
                # This can happen only if some of the last leaves were never
                # filled (length < capacity). Draw again.
                idx = np.random.randint(0, self.length)
            indices.append(idx)

        return indices