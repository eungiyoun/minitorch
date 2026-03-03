from __future__ import annotations
from typing import Any, Tuple
from ...tensor import Tensor


class Dataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Any:
        raise NotImplementedError


class TensorDataset(Dataset):
    """
    A minimal dataset that wraps one or more mini_torch Tensors.

    All tensors must have the same first dimension (number of samples).
    Each index returns a tuple of per-sample Tensors with requires_grad=False.
    """
    def __init__(self, *tensors: Tensor):
        if len(tensors) == 0:
            raise ValueError("TensorDataset requires at least one Tensor.")

        # Strict type check
        for t in tensors:
            if not isinstance(t, Tensor):
                raise TypeError(
                    f"TensorDataset expects Tensor, got {type(t)}"
                )

        # Check first dimension size consistency
        size0 = tensors[0].data.shape[0]
        for t in tensors:
            if t.data.shape[0] != size0:
                raise ValueError(
                    "All tensors must have the same size in dimension 0."
                )

        self.tensors = tensors

    def __len__(self) -> int:
        return self.tensors[0].data.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        return tuple(
            Tensor(t.data[idx], requires_grad=False)
            for t in self.tensors
        )
