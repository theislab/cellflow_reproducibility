from collections.abc import Iterable

import jax
from torch.utils.data import DataLoader


class ConditionalLoader:
    """Dataset for OT problems with conditions.

    This data loader wraps several data loaders and samples from them.

    Args:
      datasets: Datasets to sample from.
      seed: Random seed.
    """

    def __init__(
        self,
        dataloaders: Iterable[DataLoader],
        seed: int = 0,
    ):
        self.dataloaders = tuple(dataloaders)
        self._rng = jax.random.PRNGKey(seed)

    def __next__(self):
        rng, self._rng = jax.random.split(self._rng, 2)
        idx = int(jax.random.choice(rng, len(self.dataloaders)))
        dl = self.dataloaders[idx]

        return next(iter(dl))

    def __iter__(self) -> "ConditionalLoader":
        return self

    def __len__(self) -> int:
        return len(self.dataloaders)
