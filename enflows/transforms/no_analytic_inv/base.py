from abc import ABC

from enflows.transforms.base import Transform
import torch
from typing import Callable
import enflows.utils.torchutils as torchutils

class MonotonicTransform(Transform, ABC):
    """
    Elementwise Inverse for monotonic (elementwise) transformations using newton root finding method.
    For the initial guess, use bisection method.
    """
    def __init__(self, num_iterations=20, num_newton_iterations = 1, lim=10):
        self.num_iterations = num_iterations
        self.num_newton_iterations = num_newton_iterations
        self.lim = lim
        self.atol = 1e-7
        super(MonotonicTransform, self).__init__()

    def newton_inverse(self, z, context=None):
        with torch.enable_grad():
            x_guess = self.bisection_inverse(z, context=context)[0].requires_grad_(True)
            for i in range(2):
                f = self.forward(x_guess, context=context)[0] - z
                df_dx = torchutils.gradient(f, x_guess).view(f.shape)
                x_guess = x_guess - f/df_dx

        return x_guess, -self.forward_logabsdet(x_guess, context=context).reshape(-1)

    def bisection_inverse(self, z, context=None):
        x_max = torch.ones_like(z) * self.lim
        x_min = -torch.ones_like(z) * self.lim

        z_max, _ = self.forward(x_max, context)
        z_min, _ = self.forward(x_min, context)

        diff = z - z_max

        idx_maxdiff = torch.argmax(diff)
        maxdiff = diff.flatten()[idx_maxdiff]

        diff = z - z_min
        idx_mindiff = torch.argmin(diff)
        mindiff = diff.flatten()[idx_mindiff]

        if maxdiff > 0:
            ratio = (maxdiff + z_max.flatten()[idx_maxdiff]) / z_max.flatten()[idx_maxdiff]
            x_max = x_max * 1.5 * ratio
            z_max, _ = self.forward(x_max, context)

        if mindiff < 0:
            ratio = (mindiff + z_min.flatten()[idx_mindiff]) / z_min.flatten()[idx_mindiff]
            x_min = x_min * 1.5 * ratio
            z_min, _ = self.forward(x_min, context)
        # Old inversion by binary search
        i = 0
        x_middle = (x_max + x_min) / 2

        while i < self.num_iterations and (x_middle - z).abs().max() > self.atol:
        # for i in range(self.num_iterations):
            x_middle = (x_max + x_min) / 2
            z_middle, _ = self.forward(x_middle, context)
            left = (z_middle > z).float()
            right = 1 - left
            x_max = left * x_middle + right * x_max
            x_min = right * x_middle + left * x_min
            z_max = left * z_middle + right * z_max
            z_min = right * z_middle + left * z_min

            i+=1

        x = (x_max + x_min) / 2
        return x, -self.forward_logabsdet(x, context=context).squeeze()

    def forward_logabsdet(self, inputs, context=None):
        _, logabsdet = self.forward(inputs=inputs, context=context)
        return logabsdet

    def inverse(self, inputs, context=None):
        return self.newton_inverse(inputs, context=context)
