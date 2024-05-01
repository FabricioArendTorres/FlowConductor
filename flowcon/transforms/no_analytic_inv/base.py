from abc import ABC

from flowcon.transforms.base import Transform
import torch
from typing import Callable
import flowcon.utils.torchutils as torchutils


class MonotonicTransform(Transform, ABC):
    """
    Elementwise Inverse for monotonic (elementwise) transformations using newton root finding method.
    For the initial guess, use bisection method.
    """

    def __init__(self, num_iterations=20, num_newton_iterations=1, lim=10, ratio_multiplier=1.5):
        self.num_iterations = num_iterations
        self.num_newton_iterations = num_newton_iterations
        self.lim = lim
        self.atol = 1e-7
        self.ratio_multiplier = ratio_multiplier
        super(MonotonicTransform, self).__init__()

    def newton_inverse(self, z, context=None, forward_function=None):
        if forward_function is None:
            forward_function = self.forward

        with torch.enable_grad():
            x_guess = self.bisection_inverse(z, context=context,
                                             forward_function=forward_function)[0].requires_grad_(True)
            for i in range(2):
                f = forward_function(x_guess, context=context)[0] - z
                df_dx = torchutils.gradient(f, x_guess).view(f.shape)
                x_guess = x_guess - f / (df_dx + 1e-7)
        return x_guess, -self.forward_logabsdet(x_guess, context=context, forward_function=forward_function).reshape(-1)

    def bisection_inverse(self, z, context=None, forward_function=None):
        if forward_function is None:
            forward_function = self.forward

        x_max = torch.ones_like(z) * self.lim
        x_min = -torch.ones_like(z) * self.lim

        z_max, _ = forward_function(x_max, context)
        z_min, _ = forward_function(x_min, context)

        idx_maxdiff, idx_mindiff, maxdiff, mindiff = self.calc_diffs(z, z_max, z_min)

        while maxdiff > 0:
            ratio = (maxdiff + z_max.flatten()[idx_maxdiff]) / z_max.flatten()[idx_maxdiff]
            x_max = x_max * self.ratio_multiplier * ratio
            z_max, _ = forward_function(x_max, context)
            idx_maxdiff, idx_mindiff, maxdiff, mindiff = self.calc_diffs(z, z_max, z_min)

        x_max += 1
        while mindiff < 0:
            ratio = (mindiff + z_min.flatten()[idx_mindiff]) / z_min.flatten()[idx_mindiff]
            x_min = x_min * self.ratio_multiplier * ratio
            z_min, _ = forward_function(x_min, context)
            idx_maxdiff, idx_mindiff, maxdiff, mindiff = self.calc_diffs(z, z_max, z_min)
        x_min -= 1

        z_max, _ = forward_function(x_max, context)
        z_min, _ = forward_function(x_min, context)
        # Old inversion by binary search
        i = 0
        x_middle = (x_max + x_min) / 2
        while i < self.num_iterations and (x_middle - z).abs().max() > self.atol:
            # for i in range(self.num_iterations):
            x_middle = (x_max + x_min) / 2
            z_middle, _ = forward_function(x_middle, context)
            left = (z_middle > z).float()
            right = (z_middle < z).float()
            equal = 1 - (left + right)

            x_max = left * x_middle + right * x_max + equal * x_middle
            x_min = right * x_middle + left * x_min + equal * x_middle
            z_max = left * z_middle + right * z_max + equal * z_middle
            z_min = right * z_middle + left * z_min + equal * z_middle
            i += 1

        x = (x_max + x_min) / 2
        # z_pred, _ = self.forward(x_middle, context)
        return x, -self.forward_logabsdet(x, context=context, forward_function=forward_function).squeeze()

    def calc_diffs(self, z, z_max, z_min):
        diff = z - z_max
        idx_maxdiff = torch.argmax(diff)
        maxdiff = diff.flatten()[idx_maxdiff]
        diff = z - z_min
        idx_mindiff = torch.argmin(diff)
        mindiff = diff.flatten()[idx_mindiff]
        return idx_maxdiff, idx_mindiff, maxdiff, mindiff

    def forward_logabsdet(self, inputs, context=None, forward_function=None):
        if forward_function is None:
            forward_function = self.forward
        _, logabsdet = forward_function(inputs=inputs, context=context)
        return logabsdet

    def inverse(self, inputs, context=None, forward_function=None):
        if forward_function is None:
            forward_function = self.forward
        return self.newton_inverse(inputs, context=context, forward_function=forward_function)
