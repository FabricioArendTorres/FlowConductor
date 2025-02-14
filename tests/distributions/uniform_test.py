import pytest
import torch
from torch.testing import assert_allclose

from flowcon.distributions import Uniform


def test_box_uniform_init():
    dim = 3
    low = torch.tensor([-2.0, -1.0, 0.0])
    high = torch.tensor([1.0, 2.0, 3.0])
    dist = Uniform(dim, low, high)

    assert dist.dim == dim
    assert_allclose(dist._low, low.unsqueeze(0))
    assert_allclose(dist._high, high.unsqueeze(0))
    assert dist._log_norm_const.shape == (1,)


def test_box_uniform_sampling():
    dim = 2
    low = torch.tensor([-1.0, 0.0])
    high = torch.tensor([1.0, 2.0])
    dist = Uniform(dim, low, high)
    samples = dist._sample(1000)

    assert samples.shape == (1000, dim)
    assert torch.all(samples >= low) and torch.all(samples < high)


def test_box_uniform_log_prob():
    dim = 2
    low = torch.tensor([0.0, 0.0])
    high = torch.tensor([1.5, 1.0])
    dist = Uniform(dim, low, high)

    inputs_inside = torch.rand(5, 2)
    inputs_outside = torch.rand(7, 2) + 2

    log_prob_inside = dist.log_prob(inputs_inside)
    log_prob_outside = dist.log_prob(inputs_outside)
    expected_log_prob_inside = (torch.ones(5, 2) * (1 / (high - low))).prod(-1)
    assert_allclose(log_prob_inside.exp(), expected_log_prob_inside)

    assert torch.all(log_prob_inside > float("-inf"))
    assert torch.all(log_prob_outside == float("-inf"))


def test_device():
    dim = 2
    dummy_device = torch.device("meta")
    low = torch.tensor(
        [0.0, 0.0],
    )
    high = torch.tensor([1.5, 1.0])
    dist = Uniform(dim, low, high)
    dist.to(dummy_device)

    inputs_inside = torch.rand(5, 2).to(dummy_device)
    log_prob_inside = dist.log_prob(inputs_inside)
    assert log_prob_inside.device == dummy_device

    with pytest.raises(RuntimeError):
        inputs_inside = torch.rand(5, 2)
        log_prob_inside = dist.log_prob(inputs_inside)

    assert dist.sample(10).device == dummy_device
    assert dist.sample_and_log_prob(2)[0].device == dummy_device
    assert dist.sample_and_log_prob(3)[1].device == dummy_device


@pytest.mark.expensive
def test_compile():
    dim = 2
    low = torch.tensor([0.0, 0.0])
    high = torch.tensor([1.5, 1.0])
    dist = Uniform(dim, low, high)
    logprob_c = torch.compile(dist.log_prob)

    inputs_inside = torch.rand(5, 2)
    inputs_outside = torch.rand(7, 2) + 2

    log_prob_inside = dist.log_prob(inputs_inside)
    log_prob_inside_c = logprob_c(inputs_inside)
    log_prob_outside = dist.log_prob(inputs_outside)
    log_prob_outside_c = logprob_c(inputs_outside)

    assert_allclose(log_prob_inside_c, log_prob_inside)
    assert_allclose(log_prob_outside_c, log_prob_outside)

    assert torch.all(log_prob_inside > float("-inf"))
    assert torch.all(log_prob_outside == float("-inf"))
