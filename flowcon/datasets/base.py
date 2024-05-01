"""
Based on https://github.com/bayesiains/nsf
"""

from .plane import *

from torch.utils import data

def load_plane_dataset(name, num_points, flip_axes=False, return_label=False):
    """Loads and returns a plane dataset.

    Args:
        name: string, the name of the dataset.
        num_points: int, the number of points the dataset should have,
        flip_axes: bool, flip x and y axes if True.

    Possible Names:
        'gaussian'
        'crescent'
        'crescent_cubed'
        'sine_wave'
        'abs'
        'sign'
        'four_circles'
        'diamond'
        'two_spirals'
        'checkerboard'
        "eight_gaussians"
        'two_circles'
        'two_moons'
        'pinwheel'
        'swissroll'
        'rings'

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
            'rings': ConcentricRingsDataset
        }[name](num_points=num_points, flip_axes=flip_axes, return_label=return_label)

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
