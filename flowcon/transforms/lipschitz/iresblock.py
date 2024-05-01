"""
https://github.com/yperugachidiaz/invertible_densenets/blob/master/lib/layers/iresblock.py

MIT License

Copyright (c) 2019 Ricky Tian Qi Chen
Copyright (c) 2021 Yura Perugachi-Diaz
Copyright (c) 2022 Fabricio Arend Torres


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from flowcon.transforms import Transform
from flowcon.utils.torchutils import batch_jacobian, logabsdet
from flowcon.transforms.lipschitz.util import BiasedParameterGenerator, UnbiasedParameterGenerator, ParameterGenerator
from flowcon.nn.nets.invertible_densenet import *
from flowcon.nn.nets.invertible_densenet import _DenseNet

import logging
from typing import *

logger = logging.getLogger()

__all__ = ['iResBlock']


class iResBlock(Transform):
    def __init__(
            self,
            nnet:_DenseNet,
            brute_force=False,
            unbiased_estimator=True,
            **options
    ):
        """
        Args:
            nnet: a nn.Module
            brute_force: If true, computes the exact logdet.
            unbiased_estimator: If true, uses an unbiased estimator for the logdet, else a power series is used.
            options:
                trace_estimator: 'neumann' or 'basic'. Only used if brute_force=False.
        """
        super().__init__()
        self.nnet = nnet
        self.brute_force = brute_force
        self.unbiased_estimator = unbiased_estimator

        self.train_determinant_estimator = DeterminantEstimator.build(network=self.nnet,
                                                                      brute_force=self.brute_force,
                                                                      unbiased_power_series=unbiased_estimator,
                                                                      **options)

        self.test_time_determinant_estimator = DeterminantEstimator.build(network=self.nnet,
                                                                          brute_force=True,
                                                                          **options)

    def forward(self, x, context=None):
        g, logdetgrad = self._logabsdet(x, context=context)
        return x + g, logdetgrad.view(-1)

    @property
    def logabsdet_estimator(self):
        if self.training:
            return self.train_determinant_estimator
        else:
            return self.test_time_determinant_estimator

    def inverse(self, y, context=None):
        x = self._inverse_fixed_point(y, context)
        return x, -self._logabsdet(x, context=context)[1]

    def _inverse_fixed_point(self, y, context=None, atol=1e-5, rtol=1e-5):
        x, x_prev = y - self.nnet(y, context), y
        i = 0
        tol = atol + y.abs() * rtol
        while not torch.all((x - x_prev) ** 2 / tol < 1):
            x, x_prev = y - self.nnet(x, context), x
            i += 1
            if i > 1000:
                logger.info('Iterations exceeded 1000 for inverse.')
                break
        return x

    def _logabsdet(self, x, context=None):
        """Returns g(x) and logdet|d(x+g(x))/dx|."""
        with torch.enable_grad():
            g, logabsdet = self.logabsdet_estimator.logabsdet_and_g(x, training=self.training,
                                                                    context=context)

            return g, logabsdet

    def extra_repr(self):
        return 'dist={}, n_samples={}, n_power_series={}, neumann_grad={}, brute_force={}'.format(
            self.n_dist, self.n_samples, self.n_power_series, self.neumann_grad, self.brute_force
        )

    class Factory:
        def __init__(self):
            self.args_iResBlock = None
            self.densenet_factory = None

        def set_densenet(self, **kwargs):
            self.densenet_factory = DenseNet.factory(**kwargs)
            return self

        def set_logabsdet_estimator(self,
                                    brute_force=False,
                                    unbiased_estimator=True,
                                    **options):
            self.args_iResBlock = dict(brute_force=brute_force,
                                       unbiased_estimator=unbiased_estimator,
                                       **options)
            return self

        def build(self) -> 'iResBlock':
            assert self.args_iResBlock is not None, "iResBlock arguments not set. Call set_iresblock."
            assert self.densenet_factory is not None, "DenseNet arguments not set. Call set_densenet."
            return iResBlock(nnet=self.densenet_factory(),
                             **self.args_iResBlock)



class DeterminantEstimator(torch.nn.Module):
    def __init__(self, network:_DenseNet, parameter_generator: ParameterGenerator):
        super().__init__()
        self.nnet = network
        self.parameter_generator = parameter_generator

    def logabsdet_and_g(self, x, context=None, training=False, **kwargs):
        coeff_fn, n_power_series = self.parameter_generator.sample_parameters(training=training)
        g, logabsdet = self._g_and_logabsdet(coeff_fn=coeff_fn, n_power_series=n_power_series, x=x, context=context)
        return g, logabsdet.view(-1)

    @abc.abstractmethod
    def _g_and_logabsdet(self, coeff_fn, n_power_series, x, context=None) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @staticmethod
    def build(network, brute_force=False, unbiased_power_series=True, **options):
        if brute_force:
            determinant_estimator = BruteForceDeterminantEstimator(network=network)
        else:
            if unbiased_power_series:
                parameter_generator = UnbiasedParameterGenerator(n_exact_terms=options.get("n_exact_terms", 2),
                                                                 n_samples=options.get("n_samples", 1))
            else:
                parameter_generator = BiasedParameterGenerator(n_power_series=options.get("n_power_series", 5))

            determinant_estimator = ApproxTraceDeterminantEstimator(network=network,
                                                                    parameter_generator=parameter_generator,
                                                                    trace_estimator=options.get("trace_estimator",
                                                                                                "neumann"))
        return determinant_estimator


class BruteForceDeterminantEstimator(DeterminantEstimator):
    """
        Brute-force compute Jacobian determinant
    """

    def __init__(self, network:_DenseNet):
        super().__init__(network=network, parameter_generator=None)

    def logabsdet_and_g(self, x, context=None, training=False, **kwargs):
        return self._g_and_logabsdet(None, None, x, context=context)

    def _g_and_logabsdet(self, coeff_fn, n_power_series, x, context=None):
        x = x.requires_grad_(True)
        g = self.nnet(x, context)
        jac = batch_jacobian(g, x)
        identity = torch.eye(jac.shape[1]).unsqueeze(0).to(jac)
        return g, logabsdet(jac + identity)


class ApproxTraceDeterminantEstimator(DeterminantEstimator):
    """
        Power series with trace estimation.
    """

    def __init__(self, *args, trace_estimator="neumann", **kwargs):
        super().__init__(*args, **kwargs)

        if trace_estimator == "neumann":
            self.trace_estimator = self.neumann_logdet_estimator
        elif trace_estimator == "basic":
            self.trace_estimator = self.basic_logdet_estimator
        else:
            raise NotImplementedError(f"Unknown estimator '{trace_estimator}'. Has to be 'neumann' or 'basic'.")

    def logabsdet_and_g(self, x, context=None, training=False, **kwargs):
        coeff_fn, n_power_series = self.parameter_generator.sample_parameters(training=training)
        return self._g_and_logabsdet(coeff_fn=coeff_fn, n_power_series=n_power_series, x=x, context=context)

    def _g_and_logabsdet(self, coeff_fn, n_power_series, x, context=None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        vareps = torch.randn_like(x)
        x = x.requires_grad_(True)
        g = self.nnet(x, context)
        logdetgrad = self.trace_estimator(g, x, n_power_series, vareps, coeff_fn, self.training)
        return g, logdetgrad

    @staticmethod
    def basic_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
        vjp = vareps
        logdetgrad = torch.tensor(0.).to(x)
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[0]
            tr = torch.sum(vjp.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
            delta = (-1) ** (k + 1) / k * coeff_fn(k) * tr
            logdetgrad = logdetgrad + delta
        return logdetgrad

    @staticmethod
    def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
        vjp = vareps
        neumann_vjp = vareps
        with torch.no_grad():
            for k in range(1, n_power_series + 1):
                vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
                neumann_vjp = neumann_vjp + (-1) ** k * coeff_fn(k) * vjp
        vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
        logdetgrad = torch.sum(vjp_jac.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
        return logdetgrad
