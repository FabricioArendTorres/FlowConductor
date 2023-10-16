"""
https://github.com/yperugachidiaz/invertible_densenets/blob/master/lib/layers/iresblock.py

MIT License

Copyright (c) 2019 Ricky Tian Qi Chen
Copyright (c) 2021 Yura Perugachi-Diaz

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

import abc
import math
import numpy as np
import torch
import torch.nn as nn
from enflows.transforms import Transform
from enflows.utils.torchutils import batch_jacobian, batch_trace, safe_detach, logabsdet
from scipy.special import expit
import logging
import enflows.transforms.lipschitz.util as solvers
from typing import *

logger = logging.getLogger()

__all__ = ['iResBlock']


class Sampler():
    def rcdf_fn(self, k, offset):
        pass

    def sample_fn(self, m):
        pass

    @classmethod
    def build_sampler(cls, n_dist, **kwargs):
        if n_dist == 'geometric':
            return GeometricSampler(kwargs.get("geom_p"))
        elif n_dist == 'poisson':
            return GeometricSampler(kwargs.get("lamb"))
        else:
            raise NotImplementedError(f"Unknown sampler '{n_dist}'.")


class GeometricSampler(Sampler):
    def __init__(self, geom_p):
        self.geom_p = expit(geom_p)

    def sample_fn(self, m):
        return self.geometric_sample(self.geom_p, m)

    def rcdf_fn(self, k, offset):
        return self.geometric_1mcdf(self.geom_p, k, offset)

    @staticmethod
    def geometric_sample(p, n_samples):
        return np.random.geometric(p, n_samples)

    @staticmethod
    def geometric_1mcdf(p, k, offset):
        if k <= offset:
            return 1.
        else:
            k = k - offset
        """P(n >= k)"""
        return (1 - p) ** max(k - 1, 0)


class PoissonSampler(Sampler):
    def __init__(self, lamb):
        self.lamb = lamb

    def sample_fn(self, m):
        return self.poisson_sample(self.lamb, m)

    def rcdf_fn(self, k, offset):
        return self.poisson_1mcdf(self.lamb, k, offset)

    @staticmethod
    def poisson_sample(lamb, n_samples):
        return np.random.poisson(lamb, n_samples)

    @staticmethod
    def poisson_1mcdf(lamb, k, offset):
        if k <= offset:
            return 1.
        else:
            k = k - offset
        """P(n >= k)"""
        sum = 1.
        for i in range(1, k):
            sum += lamb ** i / math.factorial(i)
        return 1 - np.exp(-lamb) * sum


class ParameterGenerator(torch.nn.Module):
    def sample_parameters(self, training=True) -> Tuple[Callable, int]:
        pass

    @classmethod
    def buildParameterGenerator(cls, n_power_series, unbiased=True):
        if unbiased:
            return UnbiasedParameterGenerator(n_dist='geometric', n_exact_terms=n_power_series)
        else:
            return BiasedParameterGenerator(n_power_series=n_power_series)


class UnbiasedParameterGenerator(ParameterGenerator):
    geom_p = 0.5
    geom_p = np.log(geom_p) - np.log(1. - geom_p)

    def __init__(self, n_dist, n_exact_terms, n_samples):
        super().__init__()
        self.sampler = Sampler.build_sampler(n_dist=n_dist, geom_p=self.geom_p, lamb=2.)
        self.n_exact_terms = n_exact_terms
        self.n_samples = n_samples

        # store the samples of n.
        # self.register_buffer('last_n_samples', torch.zeros(self.n_samples))

    def sample_parameters(self, training=True):
        n_samples = self.sampler.sample_fn(m=self.n_samples)
        # n_samples = sample_fn(self.n_samples)
        n_power_series = max(n_samples) + self.n_exact_terms

        if not training:
            n_power_series += 20

        def coeff_fn(k):
            rcdf_term = self.sampler.rcdf_fn(k, self.n_exact_terms)
            return 1 / rcdf_term * sum(n_samples >= k - self.n_exact_terms) / len(n_samples)

        # if training:
        #     self.last_n_samples.copy_(torch.tensor(n_samples).to(self.last_n_samples))

        return coeff_fn, n_power_series


class BiasedParameterGenerator(ParameterGenerator):
    def __init__(self, n_power_series):
        super().__init__()

        self.n_power_series = n_power_series

    def sample_parameters(self, training=True):
        def coeff_fn(k):
            return 1

        return coeff_fn, self.n_power_series


class DeterminantEstimator(torch.nn.Module):
    def __init__(self, network: torch.nn.Module, parameter_generator: ParameterGenerator):
        super().__init__()
        self.nnet = network
        self.parameter_generator = parameter_generator

    def logabsdet_and_g(self, x, training, **kwargs):
        coeff_fn, n_power_series = self.parameter_generator.sample_parameters(training=training)
        g, logabsdet = self._g_and_logabsdet(coeff_fn=coeff_fn, n_power_series=n_power_series, x=x)
        return g, logabsdet.view(-1)

    @abc.abstractmethod
    def _g_and_logabsdet(self, coeff_fn, n_power_series, x) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @staticmethod
    def build_det_estimator(network, brute_force=False, exact_trace=True, unbiased_power_series=True, **options):
        if brute_force:
            determinant_estimator = BruteForceDeterminantEstimator(network=network)
        else:
            if unbiased_power_series:
                parameter_generator = UnbiasedParameterGenerator(n_dist=options.get("n_dist", "geometric"),
                                                                 n_exact_terms=options.get("n_exact_terms", 2),
                                                                 n_samples=options.get("n_samples", 1))
            else:
                parameter_generator = BiasedParameterGenerator(n_power_series=options.get("n_power_series", 5))

            if exact_trace:
                determinant_estimator = ExactTraceDeterminantEstimator(network=network,
                                                                       parameter_generator=parameter_generator)
            else:
                determinant_estimator = ApproxTraceDeterminantEstimator(network=network,
                                                                        parameter_generator=parameter_generator,
                                                                        grad_in_forward=options.get("grad_in_forward",
                                                                                                    False))
        return determinant_estimator


class ExactTraceDeterminantEstimator(DeterminantEstimator):
    """
    Power series with exact trace computation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _g_and_logabsdet(self, coeff_fn, n_power_series, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.requires_grad_(True)
        g = self.nnet(x)
        jac = batch_jacobian(g, x)
        logdetgrad = batch_trace(jac)
        jac_k = jac
        for k in range(2, n_power_series + 1):
            jac_k = torch.bmm(jac, jac_k)
            logdetgrad = logdetgrad + coeff_fn(k) * batch_trace(jac_k)
        return g, logdetgrad


class ApproxTraceDeterminantEstimator(DeterminantEstimator):
    """
        Power series with trace estimation.
    """

    def __init__(self, *args, trace_estimator="basic", grad_in_forward=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.grad_in_forward = grad_in_forward

        if trace_estimator == "neumann":
            self.trace_estimator = self.neumann_logdet_estimator
        elif trace_estimator == "basic":
            self.trace_estimator = self.basic_logdet_estimator
        else:
            raise NotImplementedError(f"Unknown estimator '{trace_estimator}'.")

    def logabsdet_and_g(self, x, training, time_multiplier=None, context=None, **kwargs):
        coeff_fn, n_power_series = self.parameter_generator.sample_parameters(training=training)
        if training and self.grad_in_forward:
            return self._logabsdet_and_g_grad_in_forward(coeff_fn=coeff_fn, n_power_series=n_power_series, x=x)
        else:
            return self._g_and_logabsdet(coeff_fn=coeff_fn, n_power_series=n_power_series, x=x,
                                         time_multiplier=time_multiplier,
                                         context=context)

    def _g_and_logabsdet(self, coeff_fn, n_power_series, x, time_multiplier=None, context=None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        vareps = torch.randn_like(x)
        x = x.requires_grad_(True)

        if context is not None:
            _x = torch.concat([x, context], -1)
        else:
            _x = x

        if time_multiplier is not None:
            g = self.nnet(_x) * time_multiplier
        else:
            g = self.nnet(_x)

        logdetgrad = self.trace_estimator(g, x, n_power_series, vareps, coeff_fn, self.training)
        return g, logdetgrad

    def _logabsdet_and_g_grad_in_forward(self, coeff_fn, n_power_series, x) -> Tuple[torch.Tensor, torch.Tensor]:
        # Do backprop-in-forward to save memory.
        vareps = torch.randn_like(x)
        g, logdetgrad = self.mem_eff_wrapper(
            self.trace_estimator, self.nnet, x, n_power_series, vareps, coeff_fn, self.training
        )
        return g, logdetgrad

    @staticmethod
    def mem_eff_wrapper(estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training):
        # We need this in order to access the variables inside this module,
        # since we have no other way of getting variables along the execution path.
        if not isinstance(gnet, nn.Module):
            raise ValueError('g is required to be an instance of nn.Module.')

        return MemoryEfficientLogDetEstimator.apply(
            estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training, *list(gnet.parameters())
        )

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


class BruteForceDeterminantEstimator(DeterminantEstimator):
    """
        Brute-force compute Jacobian determinant
    """

    def __init__(self, network):
        super().__init__(network=network, parameter_generator=None)

    def logabsdet_and_g(self, x, training, time_multiplier=None, context=None):
        return self._g_and_logabsdet(None, None, x, time_multiplier, context=context)

    def _g_and_logabsdet(self, coeff_fn, n_power_series, x, time_multiplier=None, context=None):
        x = x.requires_grad_(True)
        if context is not None:
            _x = torch.concat([x, context], -1)
        else:
            _x = x
        if time_multiplier is not None:
            g = self.nnet(_x) * time_multiplier
        else:
            g = self.nnet(_x)

        jac = batch_jacobian(g, x)
        identity = torch.eye(jac.shape[1]).unsqueeze(0).to(jac)
        return g, logabsdet(jac + identity)


class iResBlock(Transform):
    def __init__(
            self,
            nnet,
            time_nnet=None,
            context_features=0,
            # exact_trace=False,
            brute_force=False,
            unbiased_estimator=True,
            **options
    ):
        """
        Args:
            nnet: a nn.Module
            n_power_series: number of power series. If not None, uses a biased approximation to logdet.
            exact_trace: if False, uses a Hutchinson trace estimator. Otherwise computes the exact full Jacobian.
            brute_force: Computes the exact logdet. Only available for 2D inputs.
        """
        super().__init__()
        exact_trace = False  # not working with True..

        self.time_nnet = time_nnet
        self.nnet = nnet
        self.exact_trace = exact_trace
        self.brute_force = brute_force
        self.unbiased_estimator = unbiased_estimator
        self.context_features = context_features

        self.train_determinant_estimator = DeterminantEstimator.build_det_estimator(network=self.nnet,
                                                                                    brute_force=self.brute_force,
                                                                                    exact_trace=self.exact_trace,
                                                                                    unbiased_power_series=unbiased_estimator,
                                                                                    **options)

        self.test_time_determinant_estimator = DeterminantEstimator.build_det_estimator(network=self.nnet,
                                                                                        brute_force=True,
                                                                                        **options)

        # self.register_buffer('last_firmom', torch.zeros(1))
        # self.register_buffer('last_secmom', torch.zeros(1))

    def forward(self, x, context=None):
        if self.time_nnet is not None:
            time_multiplier = self.time_nnet(context)
        else:
            time_multiplier = None
        g, logdetgrad = self._logabsdet(x, time_multiplier=time_multiplier, context=context)
        return x + g, logdetgrad.view(-1)

    @property
    def logabsdet_estimator(self):
        if self.training:
            return self.train_determinant_estimator
        else:
            return self.test_time_determinant_estimator

    def inverse(self, y, context=None):
        if self.time_nnet is not None:
            time_multiplier = self.time_nnet(context)
        else:
            time_multiplier = None

        x = self._inverse_fixed_point(y, context)
        return x, -self._logabsdet(x, time_multiplier=time_multiplier, context=context)[1]

    def g_from_nnet(self, x, context=None):
        if context is not None:
            _x = torch.concat([x, context], -1)
        else:
            _x = x

        if self.time_nnet is not None:
            time_multiplier = self.time_nnet(context)
        else:
            time_multiplier = None

        if time_multiplier is not None:
            g = self.nnet(_x) * time_multiplier
        else:
            g = self.nnet(_x)
        return g

    def _inverse_fixed_point(self, y, context=None, atol=1e-5, rtol=1e-5):
        x, x_prev = y - self.g_from_nnet(y, context), y
        i = 0
        tol = atol + y.abs() * rtol
        while not torch.all((x - x_prev) ** 2 / tol < 1):
            x, x_prev = y - self.g_from_nnet(x, context), x
            i += 1
            if i > 1000:
                logger.info('Iterations exceeded 1000 for inverse.')
                break
        return x

    def _logabsdet(self, x, time_multiplier=None, context=None):
        """Returns g(x) and logdet|d(x+g(x))/dx|."""

        with torch.enable_grad():
            g, logabsdet = self.logabsdet_estimator.logabsdet_and_g(x, training=self.training,
                                                                    time_multiplier=time_multiplier,
                                                                    context=context)

            return g, logabsdet

    def extra_repr(self):
        return 'dist={}, n_samples={}, n_power_series={}, neumann_grad={}, exact_trace={}, brute_force={}'.format(
            self.n_dist, self.n_samples, self.n_power_series, self.neumann_grad, self.exact_trace, self.brute_force
        )


#####################
class MemoryEfficientLogDetEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training, *g_params):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = gnet(x)
            ctx.g = g
            ctx.x = x
            logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, training)

            if training:
                grad_x, *grad_params = torch.autograd.grad(
                    logdetgrad.sum(), (x,) + g_params, retain_graph=True, allow_unused=True
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        return safe_detach(g), safe_detach(logdetgrad)

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x

            # Precomputed gradients.
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]

            dg_x, *dg_params = torch.autograd.grad(g, [x] + g_params, grad_g, allow_unused=True)

        # Update based on gradient from logdetgrad.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple([g.mul_(dL) if g is not None else None for g in grad_params])

        # Update based on gradient from g.
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple([dg.add_(djac) if djac is not None else dg for dg, djac in zip(dg_params, grad_params)])

        return (None, None, grad_x, None, None, None, None) + grad_params
