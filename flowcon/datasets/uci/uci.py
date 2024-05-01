import numpy as np


def load_uci_dataset(name, split, frac=None):
    """Loads and returns a requested UCI dataset.

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

    from flowcon.datasets.uci.gas import GasDataset
    from flowcon.datasets.uci.power import PowerDataset
    from flowcon.datasets.uci.hepmass import HEPMASSDataset
    from flowcon.datasets.uci.miniboone import MiniBooNEDataset
    from flowcon.datasets.uci.bsds300 import BSDS300Dataset

    if split not in ['train', 'val', 'test']:
        raise ValueError('Split must be one of \'train\', \'val\' or \'test\'.')

    if frac is not None and (frac < 0 or frac > 1):
        raise ValueError('Frac must be between 0 and 1.')

    try:
        return {
            'power': PowerDataset,
            'gas': GasDataset,
            'hepmass': HEPMASSDataset,
            'miniboone': MiniBooNEDataset,
            'bsds300': BSDS300Dataset
        }[name](split=split, frac=frac)

    except KeyError:
        raise ValueError('Unknown dataset: {}'.format(name))


def get_uci_dataset_range(dataset_name):
    """
    Returns the per dimension (min, max) range for a specified UCI dataset.

    :param dataset_name:
    :return:
    """
    train_dataset = load_uci_dataset(dataset_name, split='train')
    val_dataset = load_uci_dataset(dataset_name, split='val')
    test_dataset = load_uci_dataset(dataset_name, split='test')
    train_min, train_max = np.min(train_dataset.data, axis=0), np.max(train_dataset.data, axis=0)
    val_min, val_max = np.min(val_dataset.data, axis=0), np.max(val_dataset.data, axis=0)
    test_min, test_max = np.min(test_dataset.data, axis=0), np.max(test_dataset.data, axis=0)
    min_ = np.minimum(train_min, np.minimum(val_min, test_min))
    max_ = np.maximum(train_max, np.maximum(val_max, test_max))
    return np.array((min_, max_))


def get_uci_dataset_max_abs_value(dataset_name):
    """
    Returns the max of the absolute values of a specified UCI dataset.

    :param dataset_name:
    :return:
    """
    range_ = get_uci_dataset_range(dataset_name)
    return np.max(np.abs(range_))
