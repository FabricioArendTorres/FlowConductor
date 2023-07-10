"""
Based on https://github.com/bayesiains/nsf/blob/master/data/base.py
"""


from .plane import *

from torch.utils import data

def load_dataset(name, split, frac=None):
    """Loads and returns a requested dataset.

    Args:
        name: string, the name of the dataset.
        split: one of 'train', 'val' or 'test', the dataset split.
        frac: float between 0 and 1 or None, the fraction of the dataset to be returned.
            If None, defaults to the whole dataset.

    Returns:
        A Dataset object, the requested dataset.

    Raises:
         ValueError: If any of the arguments has an invalid value.
    """

    if split not in ['train', 'val', 'test']:
        raise ValueError('Split must be one of \'train\', \'val\' or \'test\'.')

    if frac is not None and (frac < 0 or frac > 1):
        raise ValueError('Frac must be between 0 and 1.')


def load_plane_dataset(name, num_points, flip_axes=False):
    """Loads and returns a plane dataset.

    Args:
        name: string, the name of the dataset.
        num_points: int, the number of points the dataset should have,
        flip_axes: bool, flip x and y axes if True.

    Returns:
        A Dataset object, the requested dataset.

    Raises:
         ValueError: If `name` an unknown dataset.
    """

    try:
        return {
            'gaussian': GaussianDataset,
            'crescent': CrescentDataset,
            'crescent_cubed': CrescentCubedDataset,
            'sine_wave': SineWaveDataset,
            'abs': AbsDataset,
            'sign': SignDataset,
            'four_circles': FourCircles,
            'diamond': DiamondDataset,
            'two_spirals': TwoSpiralsDataset,
            'checkerboard': CheckerboardDataset,
            "eight_gaussians": EightGaussianDataset,
            'two_circles': TwoCircles,
            'two_moons': TwoMoonsDataset,
            'pinwheel': PinWheelDataset,
            'swissroll': SwissRollDataset,
            'ConcentricRingsDataset' : ConcentricRingsDataset
        }[name](num_points=num_points, flip_axes=flip_axes)

    except KeyError:
        raise ValueError('Unknown dataset: {}'.format(name))


def batch_generator(loader, num_batches=int(1e10)):
    batch_counter = 0
    while True:
        for batch in loader:
            yield batch
            batch_counter += 1
            if batch_counter == num_batches:
                return


class InfiniteLoader(data.DataLoader):
    """A data loader that can load a dataset repeatedly."""

    def __init__(self, num_epochs=None, *args, **kwargs):
        """Constructor.

        Args:
            dataset: A `Dataset` object to be loaded.
            batch_size: int, the size of each batch.
            shuffle: bool, whether to shuffle the dataset after each epoch.
            drop_last: bool, whether to drop last batch if its size is less than
                `batch_size`.
            num_epochs: int or None, number of epochs to iterate over the dataset.
                If None, defaults to infinity.
        """
        super().__init__(
            *args, **kwargs
        )
        self.finite_iterable = super().__iter__()
        self.counter = 0
        self.num_epochs = float('inf') if num_epochs is None else num_epochs

    def __next__(self):
        try:
            return next(self.finite_iterable)
        except StopIteration:
            self.counter += 1
            if self.counter >= self.num_epochs:
                raise StopIteration
            self.finite_iterable = super().__iter__()
            return next(self.finite_iterable)

    def __iter__(self):
        return self

    def __len__(self):
        return None


def load_num_batches(loader, num_batches):
    """A generator that returns num_batches batches from the loader, irrespective of the length
    of the dataset."""
    batch_counter = 0
    while True:
        for batch in loader:
            yield batch
            batch_counter += 1
            if batch_counter == num_batches:
                return
