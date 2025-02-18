"""Tests for Base distributions."""

import pytest
import torch
from pytest_mock import MockerFixture

from flowcon.distributions import BaseDistribution


class TestInitialization:
    def test_no_forward(self):
        dist = BaseDistribution(1)
        with pytest.raises(NotImplementedError):
            dist.forward(None)  # type: ignore

    def test_positive_dimension(self):
        with pytest.raises(TypeError):
            BaseDistribution(-1)
        with pytest.raises(TypeError):
            BaseDistribution(0)
        with pytest.raises(TypeError):
            BaseDistribution(1.0)
        BaseDistribution(1)
        BaseDistribution(5)


class TestSample:
    @pytest.mark.parametrize("dim", [1, 5, 10])
    def test_shape(self, mocker: MockerFixture, dim: int):
        dist = BaseDistribution(dim)

        def mock_sample(num_samples):
            return torch.randn(num_samples, dim)

        mocker.patch.object(dist, "_sample", side_effect=mock_sample)

        with pytest.raises(TypeError):
            dist.sample(-1)

        with pytest.raises(TypeError):
            dist.sample(0)

        with pytest.raises(TypeError):
            dist.sample(1.0)

        dist.sample(1)
        dist.sample(5)

        # print(dist.sample(1).shape)
        # print(dist.sample(10).shape)
        # assert dist.sample(1).shape == torch.Size((1, dim))
        # assert dist.sample(10).shape == torch.Size((10, dim))


if __name__ == "__main__":
    pytest.main()
