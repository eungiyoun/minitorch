from __future__ import annotations
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence
import numpy as np

from .dataset import Dataset
from ...tensor import Tensor


def _default_collate(batch: Sequence[Any]) -> Any:
    """
    Default collate function used by DataLoader to form a mini-batch.

    It takes a list of samples returned by `Dataset.__getitem__` and merges them into
    batched Tensors by stacking along the first dimension (axis=0). 
    """
    if len(batch) == 0:
        raise ValueError("Cannot collate an empty batch.")

    first = batch[0]

    if isinstance(first, Tensor):
        arr = np.stack([b.data for b in batch], axis=0)
        return Tensor(arr, requires_grad=False)

    if isinstance(first, (tuple, list)):
        transposed = list(zip(*batch))
        return type(first)(_default_collate(samples) for samples in transposed)

    if isinstance(first, dict):
        return {k: _default_collate([d[k] for d in batch]) for k in first}

    raise TypeError(
        f"default_collate: unsupported type {type(first)}"
    )


class DataLoader(Iterable):
    """
    Simple DataLoader that iterates over a Dataset and yields mini-batches.

    Supports batching, optional shuffling (with a seed), optional drop_last, and a
    collate_fn to stack/merge samples into batched Tensors.
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None,
        collate_fn: Optional[Callable[[Sequence[Any]], Any]] = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a Dataset object")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.collate_fn = collate_fn or _default_collate

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        n = len(self.dataset)

        # Match PyTorch: shuffle each epoch
        indices = np.arange(n)

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        for start in range(0, n, self.batch_size):
            end = start + self.batch_size

            if end > n and self.drop_last:
                break

            batch_idx = indices[start:end]
            batch = [self.dataset[int(i)] for i in batch_idx]

            yield self.collate_fn(batch)
