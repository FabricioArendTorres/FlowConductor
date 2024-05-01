"""
Based on https://github.com/bayesiains/nsf/blob/master/data/plane.py
Various 2-dim datasets.
"""

import numpy as np
import torch

from torch.utils.data import Dataset
import sklearn.datasets
import flowcon.utils as utils
from sklearn.utils import shuffle as util_shuffle


class PlaneDataset(Dataset):
    def __init__(self, num_points, flip_axes=False, return_label=False):
        self.num_points = num_points
        self.flip_axes = flip_axes
        self.data = None
        self.label = None
        self.min_label = None
        self.max_label = None
        self.return_label = return_label
        self.reset()

    def __getitem__(self, item):
        if self.return_label:
            return self.data[item], self.label[item]
        else:
            return self.data[item]

    def __len__(self):
        return self.num_points

    def reset(self):
        self._create_data()
        if self.flip_axes:
            x1 = self.data[:, 0]
            x2 = self.data[:, 1]
            self.data = torch.stack([x2, x1]).t()

    def _create_data(self):
        raise NotImplementedError


class GaussianDataset(PlaneDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2 = 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x1, x2)).t()


class EightGaussianDataset(PlaneDataset):
    def _create_data(self):
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = np.array([(scale * x, scale * y) for x, y in centers])

        dataset = []
        points = np.random.randn(self.num_points * 2).reshape(-1, 2) * 0.5
        idx = np.random.randint(8, size=(self.num_points,))
        center = centers[idx]
        points[..., 0] = points[..., 0] + center[..., 0]
        points[..., 1] = points[..., 1] + center[..., 1]
        dataset = points / 1.414
        self.data = torch.tensor(dataset, dtype=torch.float32)
        self.label = torch.tensor(idx, dtype=torch.float32)
        self.min_label = 0.
        self.max_label = 7.


class CrescentDataset(PlaneDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = 0.5 * x1 ** 2 - 1
        x2_var = torch.exp(torch.Tensor([-2]))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x2, x1)).t()


class TwoMoonsDataset(PlaneDataset):
    def _create_data(self):
        data, label = sklearn.datasets.make_moons(n_samples=self.num_points, noise=0.1)
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2])
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)
        self.min_label = 0.
        self.max_label = 1.


class SwissRollDataset(PlaneDataset):
    def _create_data(self):
        data, label = sklearn.datasets.make_swiss_roll(n_samples=self.num_points, noise=1.0)
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)
        self.min_label = 0.
        self.max_label = 15.


class PinWheelDataset(PlaneDataset):
    def _create_data(self):
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = self.num_points // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = np.random.randn(num_classes * num_per_class, 2) \
                   * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        x = 2 * np.einsum("ti,tij->tj", features, rotations)
        idx_permuted = np.random.permutation(np.arange(x.shape[0]))
        x_permuted = x[idx_permuted]
        label_permuted = labels[idx_permuted]
        label_permuted = label_permuted / label_permuted.max()
        self.data = torch.tensor(x_permuted, dtype=torch.float32)  # , label_permuted
        self.label = torch.tensor(label_permuted, dtype=torch.float32)  # , label_permuted
        self.min_label = 0.
        self.max_label = 1.


class CrescentCubedDataset(PlaneDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = 0.2 * x1 ** 3
        x2_var = torch.ones(x1.shape)
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x2, x1)).t()


class SineWaveDataset(PlaneDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = torch.sin(5 * x1)
        x2_var = torch.exp(-2 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x1, x2)).t()


class AbsDataset(PlaneDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = torch.abs(x1) - 1.
        x2_var = torch.exp(-3 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x1, x2)).t()


class SignDataset(PlaneDataset):
    def _create_data(self):
        x1 = torch.randn(self.num_points)
        x2_mean = torch.sign(x1) + x1
        x2_var = torch.exp(-3 * torch.ones(x1.shape))
        x2 = x2_mean + x2_var ** 0.5 * torch.randn(self.num_points)
        self.data = torch.stack((x1, x2)).t()


class TwoCircles(PlaneDataset):
    def _create_data(self):
        data, label = sklearn.datasets.make_circles(n_samples=self.num_points, factor=.5, noise=0.08)
        data = data.astype("float32")
        data *= 3
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)

        self.min_label = 0.
        self.max_label = 1.


class FourCircles(PlaneDataset):
    def __init__(self, num_points, flip_axes=False):
        if num_points % 4 != 0:
            raise ValueError('Number of data points must be a multiple of four')
        super().__init__(num_points, flip_axes)

    @staticmethod
    def create_circle(num_per_circle, std=0.1):
        u = torch.rand(num_per_circle)
        x1 = torch.cos(2 * np.pi * u)
        x2 = torch.sin(2 * np.pi * u)
        data = 2 * torch.stack((x1, x2)).t()
        data += std * torch.randn(data.shape)
        return data

    def _create_data(self):
        num_per_circle = self.num_points // 4
        centers = [
            [-1, -1],
            [-1, 1],
            [1, -1],
            [1, 1]
        ]
        self.data = torch.cat(
            [self.create_circle(num_per_circle) - torch.Tensor(center)
             for center in centers]
        )


class DiamondDataset(PlaneDataset):
    def __init__(self, num_points, flip_axes=False, width=20, bound=2.5, std=0.04):
        # original values: width=15, bound=2, std=0.05
        self.width = width
        self.bound = bound
        self.std = std
        super().__init__(num_points, flip_axes)

    def _create_data(self, rotate=True):
        # probs = (1 / self.width**2) * torch.ones(self.width**2)
        #
        # means = torch.Tensor([
        #     (x, y)
        #     for x in torch.linspace(-self.bound, self.bound, self.width)
        #     for y in torch.linspace(-self.bound, self.bound, self.width)
        # ])
        #
        # covariance = self.std**2 * torch.eye(2)
        # covariances = covariance[None, ...].repeat(self.width**2, 1, 1)
        #
        # mixture_distribution = distributions.OneHotCategorical(
        #     probs=probs
        # )
        # components_distribution = distributions.MultivariateNormal(
        #     loc=means,
        #     covariance_matrix=covariances
        # )
        #
        # mask = mixture_distribution.sample((self.num_points,))[..., None].repeat(1, 1, 2)
        # samples = components_distribution.sample((self.num_points,))
        # self.data = torch.sum(mask * samples, dim=-2)
        # if rotate:
        #     rotation_matrix = torch.Tensor([
        #         [1 / np.sqrt(2), -1 / np.sqrt(2)],
        #         [1 / np.sqrt(2), 1 / np.sqrt(2)]
        #     ])
        #     self.data = self.data @ rotation_matrix
        means = np.array([
            (x + 1e-3 * np.random.rand(), y + 1e-3 * np.random.rand())
            for x in np.linspace(-self.bound, self.bound, self.width)
            for y in np.linspace(-self.bound, self.bound, self.width)
        ])

        covariance_factor = self.std * np.eye(2)

        index = np.random.choice(range(self.width ** 2), size=self.num_points, replace=True)
        noise = np.random.randn(self.num_points, 2)
        self.data = means[index] + noise @ covariance_factor
        if rotate:
            rotation_matrix = np.array([
                [1 / np.sqrt(2), -1 / np.sqrt(2)],
                [1 / np.sqrt(2), 1 / np.sqrt(2)]
            ])
            self.data = self.data @ rotation_matrix
        self.data = self.data.astype(np.float32)
        self.data = torch.Tensor(self.data)


class TwoSpiralsDataset(PlaneDataset):
    def _create_data(self):
        n = torch.sqrt(torch.rand(self.num_points // 2)) * 540 * (2 * np.pi) / 360
        d1x = -torch.cos(n) * n + torch.rand(self.num_points // 2) * 0.5
        d1y = torch.sin(n) * n + torch.rand(self.num_points // 2) * 0.5
        x = torch.cat([torch.stack([d1x, d1y]).t(), torch.stack([-d1x, -d1y]).t()])
        self.data = x / 3 + torch.randn_like(x) * 0.1


class ConcentricRingsDataset(PlaneDataset):
    def _create_data(self):
        n_samples4 = n_samples3 = n_samples2 = self.num_points // 4
        n_samples1 = self.num_points - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X)

        # Add noise
        X = X + np.random.normal(scale=0.08, size=X.shape)
        self.data = X.astype("float32")


class TestGridDataset(PlaneDataset):
    def __init__(self, num_points_per_axis, bounds):
        self.num_points_per_axis = num_points_per_axis
        self.bounds = bounds
        self.shape = [num_points_per_axis] * 2
        self.X = None
        self.Y = None
        super().__init__(num_points=num_points_per_axis ** 2)

    def _create_data(self):
        x = np.linspace(self.bounds[0][0], self.bounds[0][1], self.num_points_per_axis)
        y = np.linspace(self.bounds[1][0], self.bounds[1][1], self.num_points_per_axis)
        self.X, self.Y = np.meshgrid(x, y)
        data_ = np.vstack([self.X.flatten(), self.Y.flatten()]).T
        self.data = torch.tensor(data_).float()


class CheckerboardDataset(PlaneDataset):
    def _create_data(self):
        x1 = torch.rand(self.num_points) * 4 - 2
        x2_ = torch.rand(self.num_points) - torch.randint(0, 2, [self.num_points]).float() * 2
        x2 = x2_ + torch.floor(x1) % 2
        self.data = torch.stack([x1, x2]).t() * 2


def _test():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset = DiamondDataset(num_points=int(1e6), width=20, bound=2.5, std=0.04)

    from flowcon.utils import torchutils
    from matplotlib import pyplot as plt
    data = torchutils.tensor_to_np(dataset.data)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.scatter(data[:, 0], data[:, 1], s=2, alpha=0.5)
    bound = 4
    bounds = [[-bound, bound], [-bound, bound]]
    # bounds = [
    #     [0, 1],
    #     [0, 1]
    # ]
    ax.hist2d(data[:, 0], data[:, 1], bins=256, range=bounds)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    plt.show()


if __name__ == '__main__':
    _test()
