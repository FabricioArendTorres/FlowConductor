'''
Adapted from Implicit Normalizing Flow (ICLR 2021).
Link: https://github.com/thu-ml/implicit-normalizing-flows/blob/master/lib/layers/broyden.py
'''

import torch
from torch import nn, nn as nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np
import math
import pickle
import sys
import os
from typing import *
from scipy.optimize import root
from scipy.special import expit
import time

import logging

logger = logging.getLogger()


def find_fixed_point_noaccel(f, x0, threshold=1000, eps=1e-5):
    b = x0.size(0)
    b_shape = (b,)
    for _ in range(1, len(x0.shape)):
        b_shape = b_shape + (1,)
    alpha = 0.5 * torch.ones(b_shape, device=x0.device)
    x, x_prev = (1 - alpha) * x0 + (alpha) * f(x0), x0
    i = 0
    tol = eps + eps * x0.abs()

    best_err = 1e9 * torch.ones(b_shape, device=x0.device)
    best_iter = torch.zeros(b_shape, dtype=torch.int64, device=x0.device)
    while True:
        fx = f(x)
        err_values = torch.abs(fx - x) / tol
        cur_err = torch.max(err_values.view(b, -1), dim=1)[0].view(b_shape)

        if torch.all(cur_err < 1.):
            break
        alpha = torch.where(torch.logical_and(cur_err >= best_err, i >= best_iter + 30),
                            alpha * 0.9,
                            alpha)
        alpha = torch.max(alpha, 0.1 * torch.ones_like(alpha))
        best_iter = torch.where(torch.logical_or(cur_err < best_err, i >= best_iter + 30),
                                i * torch.ones(b_shape, dtype=torch.int64, device=x0.device),
                                best_iter)
        best_err = torch.min(best_err, cur_err)

        x, x_prev = (1 - alpha) * x + (alpha) * fx, x
        i += 1
        if i > threshold:
            dx = torch.abs(f(x) - x)
            rel_err = torch.max(dx / tol).item()
            abs_err = torch.max(dx).item()
            if rel_err > 3 or abs_err > 3 * max(eps, 1e-9):
                logger.info('Relative/Absolute error maximum: %.10f/%.10f' % (rel_err, abs_err))
                logger.info('Iterations exceeded %d for fixed point noaccel.' % (threshold))
            break
    return x


def find_fixed_point(f, x0, threshold=1000, eps=1e-5):
    b = x0.size(0)

    def g(w):
        return f(w.view(x0.shape)).view(b, -1)

    with torch.no_grad():
        X0 = x0.view(b, -1)
        X1 = g(X0)
        Gnm1 = X1
        dXnm1 = X1 - X0
        Xn = X1

        tol = eps + eps * X0.abs()
        best_err = math.inf
        best_iter = 0
        n = 1
        while n < threshold:
            Gn = g(Xn)
            dXn = Gn - Xn
            cur_err = torch.max(torch.abs(dXn) / tol).item()
            if cur_err <= 1.:
                break
            if cur_err < best_err:
                best_err = cur_err
                best_iter = n
            elif n >= best_iter + 10:
                break

            d2Xn = dXn - dXnm1
            d2Xn_norm = torch.linalg.vector_norm(d2Xn, dim=1)
            mult = (d2Xn * dXn).sum(dim=1) / (d2Xn_norm ** 2 + 1e-8)
            mult = mult.view(b, 1)
            Xnp1 = Gn - mult * (Gn - Gnm1)

            dXnm1 = dXn
            Gnm1 = Gn
            Xn = Xnp1
            n = n + 1

        rel_err = torch.max(torch.abs(dXn) / tol).item()
        if rel_err > 1:
            abs_err = torch.max(torch.abs(dXn)).item()
            if rel_err > 10:
                return find_fixed_point_noaccel(f, x0, threshold=threshold, eps=eps)
            else:
                return find_fixed_point_noaccel(f, Xn.view(x0.shape), threshold=threshold, eps=eps)
        else:
            return Xn.view(x0.shape)




class ParameterGenerator(torch.nn.Module):
    def sample_parameters(self, training=True) -> Tuple[Callable, int]:
        pass



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


class UnbiasedParameterGenerator(ParameterGenerator):
    geom_p = 0.5
    geom_p = np.log(geom_p) - np.log(1. - geom_p)

    def __init__(self, n_exact_terms, n_samples):
        super().__init__()
        self.sampler = GeometricSampler(self.geom_p)
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

        return coeff_fn, n_power_series


class BiasedParameterGenerator(ParameterGenerator):
    def __init__(self, n_power_series):
        super().__init__()

        self.n_power_series = n_power_series

    def sample_parameters(self, training=True):
        def coeff_fn(k):
            return 1

        return coeff_fn, self.n_power_series


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x) / self.w0


def exists(val):
    return val is not None
