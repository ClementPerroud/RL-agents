import torch
from collections.abc import Iterable

epsilon = 1e-6  # strictly positive clamp


class SumTree:
    """
    Flat-array SumTree for Prioritized Experience Replay (PER), using torch tensors.
    """

    def __init__(self, size: int, dtype: torch.dtype = torch.float32):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("size must be a strictly positive integer")

        self.size = int(size)                  # capacity
        self.length = 0                        # how many items have ever been written (clamped to size)
        self.write = 0                         # next leaf slot in [0, size)
        self.leaf_start = self.size - 1        # first leaf index in flat tree

        # One contiguous tensor for the entire tree (internal + leaves)
        # We keep it on CPU and out of autograd on purpose.
        self.tree = torch.zeros(2 * self.size - 1, dtype=dtype)
        self.tree.requires_grad_(False)

    # ----------------------------- helpers ---------------------------------

    @torch.no_grad()
    def _update_leaf(self, leaf_pos: int, new_p: float) -> None:
        """Set leaf at logical position (0..size-1) to new_p and propagate."""
        leaf_idx = self.leaf_start + int(leaf_pos)

        # Clamp to strictly positive to avoid dead leaves
        v = float(new_p)
        if not torch.isfinite(torch.tensor(v)):
            raise ValueError(f"priority must be finite, got {v}")
        if v <= 0:
            v = epsilon

        change = v - float(self.tree[leaf_idx])
        if change == 0.0:
            return  # nothing to do

        self.tree[leaf_idx] = v

        # Propagate change up to the root
        idx = leaf_idx
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    # ------------------------------ API ------------------------------------

    @torch.no_grad()
    def add(self, value: float | Iterable[float]) -> None:
        """
        Insert a priority (or batch). Overwrites in FIFO when full.
        """
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for v in value:
                self.add(float(v))
            return

        pos = self.write
        self._update_leaf(pos, float(value))

        # Move circular pointer & adjust logical length
        self.write = (self.write + 1) % self.size
        self.length = min(self.length + 1, self.size)

    @torch.no_grad()
    def __setitem__(self, idx, value) -> None:
        """
        Update one or several priorities in-place.
        - idx can be int, list[int], torch.Tensor[int]
        - value can be float or a same-shaped container
        """
        # Vector/batch case
        if isinstance(idx, (list, tuple)) or (isinstance(idx, torch.Tensor) and idx.ndim >= 1):
            idx_t = torch.as_tensor(idx, dtype=torch.long)
            val_t = torch.as_tensor(value, dtype=self.tree.dtype)
            if idx_t.shape != val_t.shape:
                raise ValueError("index and value must have the same shape")
            for i, v in zip(idx_t.tolist(), val_t.tolist()):
                self.__setitem__(int(i), float(v))
            return

        # Scalar case
        i = int(idx)
        if i < 0 or i >= self.length:
            raise IndexError(f"Cannot set value at index {i} (current length = {self.length})")
        self._update_leaf(i, float(value))

    def __getitem__(self, idx):
        """
        Read leaf priority (or priorities).
        Returns torch.Tensor for vector input, Python float for scalar.
        """
        if isinstance(idx, (list, tuple)) or (isinstance(idx, torch.Tensor) and idx.ndim >= 1):
            idx_t = torch.as_tensor(idx, dtype=torch.long)
            leaf_idx = self.leaf_start + idx_t
            return self.tree.index_select(0, leaf_idx)
        else:
            i = int(idx)
            return float(self.tree[self.leaf_start + i])

    def sum(self) -> float:
        """Total priority (value at the root)."""
        return float(self.tree[0])

    @torch.no_grad()
    def sample(self, batch_size: int) -> list[int]:
        """
        Sample `batch_size` leaf indices proportionally to their priorities.
        Raises ValueError if empty or if total priority is non-positive.
        """
        if self.length == 0:
            raise ValueError("Cannot sample from an empty SumTree")

        total = self.sum()
        if not torch.isfinite(torch.tensor(total)) or total <= 0.0:
            raise ValueError(f"Invalid total priority: {total}")

        # Draw targets uniformly in [0, total)
        # (subtract tiny epsilon to avoid edge-case hitting exactly total)
        targets = torch.rand(batch_size) * max(total - epsilon, 0.0)

        out: list[int] = []
        for s in targets.tolist():
            idx = 0  # start at root
            # Descend until a leaf is reached
            while idx < self.leaf_start:  # while we're at an internal node
                left = 2 * idx + 1
                right = left + 1

                left_sum = float(self.tree[left])
                if s < left_sum:
                    idx = left
                else:
                    s -= left_sum
                    idx = right

            # Convert flat leaf index to logical position [0..size-1]
            pos = idx - self.leaf_start

            # If buffer is not yet full, the tail leaves are zero; they won't be hit
            # unless user set them explicitly. As a last guard:
            if pos >= self.length:
                # pick a valid index uniformly
                pos = int(torch.randint(0, self.length, (1,)).item())

            out.append(int(pos))

        return out

    # (Optional) helpers
    def __len__(self) -> int:
        return self.length

    @property
    def priorities(self) -> torch.Tensor:
        """View of all leaf priorities (length == size)."""
        return self.tree[self.leaf_start:self.leaf_start + self.size]