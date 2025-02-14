"""Tests for Normal distributions."""

from typing import Dict, Tuple

import numpy as np
import pytest
import torch
import torchtestcase
from scipy import stats
from scipy.stats import kstest, normaltest, pearsonr, shapiro
from torch import compile

from flowcon.distributions import normal


def t_standard_normality(
    X: np.ndarray,
) -> Tuple[Dict[str, Dict[str, float]], np.ndarray, Dict[str, Dict[str, float]]]:
    """
    Perform statistical tests to check if each column in X follows N(0,1).
    Uses Shapiro-Wilk test for small n, and Kolmogorov-Smirnov for large n.
    Also computes correlation matrix and tests for significant correlations.

    Parameters
    ----------
    X : np.ndarray
        A 2D NumPy array of shape (n, d), where n is the number of samples and d is the number of dimensions.

    Returns
    -------
    Tuple[Dict[str, Dict[str, float]], np.ndarray, Dict[str, Dict[str, float]]]
        - Dictionary containing normality test p-values for each dimension.
        - Correlation matrix of shape (d, d).
        - Dictionary containing Pearson correlation coefficients and p-values for each pair of dimensions.
    """
    n, d = X.shape
    p_values = {}
    for i in range(d):
        col_data = X[:, i]
        # Test for normality using different methods
        if n < 5000:
            stat, p = shapiro(col_data)  # Shapiro-Wilk Test (small samples)
        else:
            stat, p = kstest(col_data, "norm")  # KS Test (large samples)
        # Additional normality test
        _, p_dap = normaltest(col_data)  # D'Agostino-Pearson test
        p_values[f"Dimension {i}"] = {
            "Shapiro/KS p-value": p,
            "D'Agostino-Pearson p-value": p_dap,
        }
    # Compute correlation matrix
    correlation_matrix = np.corrcoef(X, rowvar=False)
    correlation_p_values = {}
    for i in range(d):
        for j in range(i + 1, d):
            r, p_corr = pearsonr(X[:, i], X[:, j])
            correlation_p_values[f"Corr({i},{j})"] = {"Pearson r": r, "p-value": p_corr}
    return p_values, correlation_matrix, correlation_p_values


class StandardNormalTest(torchtestcase.TorchTestCase):
    def test_log_prob(self):
        batch_size = 10
        input_dim = 2 * 3 * 4
        dist = normal.StandardNormal(input_dim)
        inputs = torch.randn(batch_size, input_dim)

        log_prob = dist.log_prob(inputs)
        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, torch.Size([batch_size]))
        self.assertFalse(torch.isnan(log_prob).any())
        self.assertFalse(torch.isinf(log_prob).any())
        ref_log_prob = stats.Normal(mu=0, sigma=1).logpdf(inputs.numpy()).sum(-1)

        np.testing.assert_allclose(ref_log_prob, log_prob)

    def test_sample(self):
        """
        ensure that samples come from the standard normal distribution
        """
        num_samples = 10
        input_dim = 3
        dist = normal.StandardNormal(input_dim)
        samples = dist.sample(num_samples)

        self.assertIsInstance(samples, torch.Tensor)
        self.assertFalse(torch.isnan(samples).any())
        self.assertFalse(torch.isinf(samples).any())
        self.assertEqual(samples.shape, torch.Size([num_samples, input_dim]))

        np.random.seed(42)
        torch.random.manual_seed(42)
        n = 10_000
        custom_samples = dist.sample(n).numpy()
        test_results, corr_matrix, corr_p_values = t_standard_normality(custom_samples)

        print(np.mean(custom_samples, -1))
        print(np.std(custom_samples, -1))
        for dim in test_results:
            for test in test_results[dim]:
                assert test_results[dim][test] > 0.05

        for dim_results in corr_p_values:
            assert corr_p_values[dim_results]["p-value"] > 0.05
            assert corr_p_values[dim_results]["Pearson r"] < 0.01

    def test_sample_and_log_prob(self):
        num_samples = 10
        input_shape = 2 * 3 * 4
        dist = normal.StandardNormal(input_shape)
        samples, log_prob_1 = dist.sample_and_log_prob(num_samples)
        log_prob_2 = dist.log_prob(samples)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(log_prob_1, torch.Tensor)
        self.assertIsInstance(log_prob_2, torch.Tensor)
        self.assertEqual(samples.shape, torch.Size([num_samples, input_shape]))
        self.assertEqual(log_prob_1.shape, torch.Size([num_samples]))
        self.assertEqual(log_prob_2.shape, torch.Size([num_samples]))
        self.assertEqual(log_prob_1, log_prob_2)

    def test_dim(self):
        for dim in [2, 3, 10]:
            dist = normal.StandardNormal(dim)
            assert dist.dim == dim

    @pytest.mark.expensive
    def test_torch_compile(self):
        batch_size = 10
        input_dim = 2 * 3 * 4
        dist = normal.StandardNormal(input_dim)
        compiled_log_prob = compile(dist.log_prob, mode="reduce-overhead")
        compiled_sample = compile(dist.sample, mode="reduce-overhead")

        inputs = torch.randn(batch_size, input_dim)

        # Check log_prob
        log_prob_eager = dist.log_prob(inputs)
        log_prob_compiled = compiled_log_prob(inputs)
        torch.testing.assert_close(log_prob_eager, log_prob_compiled)

        # Check sample
        sample_eager = dist.sample(100)
        sample_compiled = compiled_sample(100)
        assert sample_eager.shape == sample_compiled.shape
        assert not torch.isnan(sample_compiled).any()
        assert not torch.isinf(sample_compiled).any()

    def test_device(self):
        num_samples = 10
        dim = 2 * 3 * 4
        dummy_device = torch.device("meta")

        dist = normal.StandardNormal(dim).to(dummy_device)

        samples, log_prob_1 = dist.sample_and_log_prob(num_samples)
        assert samples.device == dummy_device
        assert log_prob_1.device == dummy_device

        log_prob_2 = dist.log_prob(samples)
        assert log_prob_2.device == dummy_device

        with pytest.raises(RuntimeError):
            inputs_cpu = torch.rand(5, dim)
            dist.log_prob(inputs_cpu)


class DiagonalNormalTestDefault(torchtestcase.TorchTestCase):
    def test_parameter_shape(self):
        with pytest.raises(AssertionError):
            dist = normal.DiagonalNormal(3, mean=torch.ones(1, 2, 1))

        with pytest.raises(AssertionError):
            dist = normal.DiagonalNormal(2, log_std=torch.ones(2, 1, 2))

        with pytest.raises(AssertionError):
            dist = normal.DiagonalNormal(
                2, mean=torch.ones(1, 2), log_std=torch.ones(1)
            )

        with pytest.raises(AssertionError):
            dist = normal.DiagonalNormal(
                1, mean=torch.ones(1, 2), log_std=torch.ones(1, 2)
            )

        dist = normal.DiagonalNormal(2, mean=torch.ones(1, 2), log_std=torch.ones(1, 2))
        dist = normal.DiagonalNormal(
            2, mean=torch.ones(1, 2, 1, 1), log_std=torch.ones(1, 2)
        )

        dist = normal.DiagonalNormal(2, mean=torch.ones(2), log_std=torch.ones(2))

    def test_log_prob(self):
        batch_size = 10
        input_dim = 2 * 3 * 4
        dist = normal.DiagonalNormal(
            input_dim,
            mean=torch.ones(1, input_dim),
            log_std=3 * torch.ones(1, input_dim),
        )
        inputs = torch.randn(batch_size, input_dim)

        log_prob = dist.log_prob(inputs)

        self.assertIsInstance(log_prob, torch.Tensor)
        self.assertEqual(log_prob.shape, torch.Size([batch_size]))
        self.assertFalse(torch.isnan(log_prob).any())
        self.assertFalse(torch.isinf(log_prob).any())

        ref_log_prob = (
            stats.Normal(
                mu=dist._mean.detach().numpy(),
                sigma=np.exp(dist._log_std.detach().numpy()),
            )
            .logpdf(inputs.detach().numpy())
            .sum(-1)
        )

        np.testing.assert_allclose(ref_log_prob, log_prob.detach().numpy())

    def test_requires_grad(self):
        assert normal.DiagonalNormal(2)._mean.requires_grad
        assert normal.DiagonalNormal(2, trainable_mean=True)._mean.requires_grad
        assert not normal.DiagonalNormal(2, trainable_mean=False)._mean.requires_grad

        assert normal.DiagonalNormal(2)._log_std.requires_grad
        assert normal.DiagonalNormal(2, trainable_log_std=True)._log_std.requires_grad
        assert not normal.DiagonalNormal(
            2, trainable_log_std=False
        )._log_std.requires_grad

    def test_sample(self):
        """
        ensure that samples come from the standard normal distribution
        """
        num_samples = 10
        input_dim = 3

        dist = normal.DiagonalNormal(
            input_dim,
            mean=torch.ones(1, input_dim),
            log_std=3 * torch.ones(1, input_dim),
        )

        mu = dist._mean
        inv_sigma = (-dist._log_std).exp()

        normalized_samples = (dist.sample(num_samples) - mu) * inv_sigma
        self.assertIsInstance(normalized_samples, torch.Tensor)
        self.assertFalse(torch.isnan(normalized_samples).any())
        self.assertFalse(torch.isinf(normalized_samples).any())
        self.assertEqual(normalized_samples.shape, torch.Size([num_samples, input_dim]))

        np.random.seed(42)
        torch.random.manual_seed(42)
        n = 10_000

        custom_samples = (dist.sample(n) - mu) * inv_sigma
        custom_samples = custom_samples.detach().numpy()

        test_results, corr_matrix, corr_p_values = t_standard_normality(custom_samples)

        print(np.mean(custom_samples, -1))
        print(np.std(custom_samples, -1))
        for dim in test_results:
            for test in test_results[dim]:
                assert test_results[dim][test] > 0.05

        for dim_results in corr_p_values:
            assert corr_p_values[dim_results]["p-value"] > 0.05
            assert corr_p_values[dim_results]["Pearson r"] < 0.01

    def test_sample_and_log_prob(self):
        num_samples = 10
        input_shape = 2 * 3 * 4
        dist = normal.DiagonalNormal(input_shape)
        samples, log_prob_1 = dist.sample_and_log_prob(num_samples)
        log_prob_2 = dist.log_prob(samples)
        self.assertIsInstance(samples, torch.Tensor)
        self.assertIsInstance(log_prob_1, torch.Tensor)
        self.assertIsInstance(log_prob_2, torch.Tensor)
        self.assertEqual(samples.shape, torch.Size([num_samples, input_shape]))
        self.assertEqual(log_prob_1.shape, torch.Size([num_samples]))
        self.assertEqual(log_prob_2.shape, torch.Size([num_samples]))
        self.assertEqual(log_prob_1, log_prob_2)

    def test_dim(self):
        for dim in [2, 3, 10]:
            dist = normal.DiagonalNormal(dim)
            assert dist.dim == dim

    @pytest.mark.expensive
    def test_torch_compile(self):
        batch_size = 10
        input_dim = 2 * 3 * 4
        dist = normal.DiagonalNormal(input_dim)
        compiled_log_prob = compile(dist.log_prob, mode="reduce-overhead")
        compiled_sample = compile(dist.sample, mode="reduce-overhead")

        inputs = torch.randn(batch_size, input_dim)

        # Check log_prob
        log_prob_eager = dist.log_prob(inputs)
        log_prob_compiled = compiled_log_prob(inputs)
        torch.testing.assert_close(log_prob_eager, log_prob_compiled)

        # Check sample
        sample_eager = dist.sample(100)
        sample_compiled = compiled_sample(100)
        assert sample_eager.shape == sample_compiled.shape
        assert not torch.isnan(sample_compiled).any()
        assert not torch.isinf(sample_compiled).any()

    def test_device(self):
        num_samples = 10
        dim = 2 * 3 * 4
        dummy_device = torch.device("meta")

        dist = normal.DiagonalNormal(dim).to(dummy_device)

        samples, log_prob_1 = dist.sample_and_log_prob(num_samples)
        assert samples.device == dummy_device
        assert log_prob_1.device == dummy_device

        log_prob_2 = dist.log_prob(samples)
        assert log_prob_2.device == dummy_device

        with pytest.raises(RuntimeError):
            inputs_cpu = torch.rand(5, dim)
            dist.log_prob(inputs_cpu)


# class ConditionalDiagonalNormalTest(torchtestcase.TorchTestCase):
#     def test_log_prob(self):
#         batch_size = 10
#         input_shape = [2, 3, 4]
#         context_shape = [2, 3, 8]
#         dist = normal.ConditionalDiagonalNormal(input_shape)
#         inputs = torch.randn(batch_size, *input_shape)
#         context = torch.randn(batch_size, *context_shape)
#         log_prob = dist.log_prob(inputs, context=context)
#         self.assertIsInstance(log_prob, torch.Tensor)
#         self.assertEqual(log_prob.shape, torch.Size([batch_size]))
#         self.assertFalse(torch.isnan(log_prob).any())
#         self.assertFalse(torch.isinf(log_prob).any())

#     def test_sample(self):
#         num_samples = 10
#         context_size = 20
#         input_shape = [2, 3, 4]
#         context_shape = [2, 3, 8]
#         dist = normal.ConditionalDiagonalNormal(input_shape)
#         context = torch.randn(context_size, *context_shape)
#         samples = dist.sample(num_samples, context=context)
#         self.assertIsInstance(samples, torch.Tensor)
#         self.assertEqual(
#             samples.shape, torch.Size([context_size, num_samples] + input_shape)
#         )
#         self.assertFalse(torch.isnan(samples).any())
#         self.assertFalse(torch.isinf(samples).any())

#     def test_sample_and_log_prob_with_context(self):
#         num_samples = 10
#         context_size = 20
#         input_shape = [2, 3, 4]
#         context_shape = [2, 3, 8]
#         dist = normal.ConditionalDiagonalNormal(input_shape)
#         context = torch.randn(context_size, *context_shape)
#         samples, log_prob = dist.sample_and_log_prob(num_samples, context=context)
#         self.assertIsInstance(samples, torch.Tensor)
#         self.assertIsInstance(log_prob, torch.Tensor)
#         self.assertEqual(
#             samples.shape, torch.Size([context_size, num_samples] + input_shape)
#         )
#         self.assertEqual(log_prob.shape, torch.Size([context_size, num_samples]))

#     def test_mean(self):
#         context_size = 20
#         input_shape = [2, 3, 4]
#         context_shape = [2, 3, 8]
#         dist = normal.ConditionalDiagonalNormal(input_shape)
#         context = torch.randn(context_size, *context_shape)
#         means = dist.mean(context=context)
#         self.assertIsInstance(means, torch.Tensor)
#         self.assertFalse(torch.isnan(means).any())
#         self.assertFalse(torch.isinf(means).any())
#         self.assertEqual(means.shape, torch.Size([context_size] + input_shape))


if __name__ == "__main__":
    pytest.main()
