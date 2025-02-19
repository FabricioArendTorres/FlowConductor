from __future__ import annotations

from typing import Iterable, Iterator

import torch
import torch.nn as nn
from UMNN import NeuralIntegral, ParallelNeuralIntegral  # type: ignore


def _flatten(sequence: Iterator[torch.Tensor]) -> torch.Tensor:
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


class ELUPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.elu(x) + 1.0


class IntegrandNet(nn.Module):
    def __init__(self, hidden: list[int], cond_in: int):
        super(IntegrandNet, self).__init__()
        l1 = [1 + cond_in] + hidden
        l2 = hidden + [1]
        layers = []
        for h1, h2 in zip(l1, l2, strict=False):
            layers += [nn.Linear(h1, h2), nn.ReLU()]
        layers.pop()
        layers.append(ELUPlus())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        nb_batch, in_d = x.shape
        x = torch.cat((x, h), 1)
        x_he = x.view(nb_batch, -1, in_d).transpose(1, 2).contiguous().view(nb_batch * in_d, -1)
        y = self.net(x_he).view(nb_batch, -1)
        return y


class MonotonicNormalizer(nn.Module):
    def __init__(
        self,
        integrand_net: IntegrandNet | list[int] | Iterable[int],
        cond_size: int,
        nb_steps: int = 20,
        solver: str = "CC",
    ):
        super(MonotonicNormalizer, self).__init__()
        if type(integrand_net) is list:
            self.integrand_net = IntegrandNet(integrand_net, cond_size)
        elif type(integrand_net) is IntegrandNet:
            self.integrand_net = integrand_net
        else:
            raise TypeError(f"Unknown type of integrand net: {type(integrand_net)}")
        self.solver = solver
        self.nb_steps = nb_steps

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = torch.zeros(x.shape).to(x.device)
        xT = x
        z0 = h[:, :, 0]
        h = h.permute(0, 2, 1).contiguous().view(x.shape[0], -1)
        if self.solver == "CC":
            z = (
                NeuralIntegral.apply(
                    x0,
                    xT,
                    self.integrand_net,
                    _flatten(self.integrand_net.parameters()),
                    h,
                    self.nb_steps,
                )
                + z0
            )
        elif self.solver == "CCParallel":
            z = (
                ParallelNeuralIntegral.apply(
                    x0,
                    xT,
                    self.integrand_net,
                    _flatten(self.integrand_net.parameters()),
                    h,
                    self.nb_steps,
                )
                + z0
            )
        else:
            raise RuntimeError(f"Unknown solver {self.solver}")
        return z, self.integrand_net(x, h)

    def inverse_transform(
        self, z: torch.Tensor, h: torch.Tensor, context: torch.Tensor | None = None
    ):
        # Old inversion by binary search
        x_max = torch.ones_like(z) * 20
        x_min = -torch.ones_like(z) * 20
        z_max, _ = self.forward(x_max, h, context)
        z_min, _ = self.forward(x_min, h, context)
        for i in range(25):
            x_middle = (x_max + x_min) / 2
            z_middle, _ = self.forward(x_middle, h, context)
            left = (z_middle > z).float()
            right = 1 - left
            x_max = left * x_middle + right * x_max
            x_min = right * x_middle + left * x_min
            z_max = left * z_middle + right * z_max
            z_min = right * z_middle + left * z_min
        return (x_max + x_min) / 2
