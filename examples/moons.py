import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from torch import nn
from torch import optim

from enflows.flows.base import Flow
from enflows.nn.nets import *
from enflows.distributions.normal import StandardNormal
from enflows.transforms.base import CompositeTransform
from enflows.transforms import *
from enflows.transforms.autoregressive import *
from enflows.transforms.permutations import ReversePermutation
import numpy as np

device = "cuda"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


num_layers = 10
base_dist = StandardNormal(shape=[2])

transforms = []

hypernet_kwargs = dict(features=2, hidden_features=64, num_blocks=2)


densenet_builder = LipschitzDenseNetBuilder(input_channels=2,
                                            densenet_depth=3,
                                            activation_function=Sin(),
                                            lip_coeff=.97,
                                            )

for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2))
    # transforms.append(LULinear(features=2, identity_init=True))
    transforms.append(ActNorm(features=2))

    # transforms.append(
    #     MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=2, hidden_features=256, num_bins=8,
    #                                                             tails='linear', num_blocks=2, tail_bound=3,
    #                                                             ))

    # transforms.append(MaskedDeepSigmoidTransform(n_sigmoids=5, **hypernet_kwargs))
    # transforms.append(MaskedDeepSigmoidTransform(n_sigmoids=10, **hypernet_kwargs))
    transforms.append(iResBlock(densenet_builder.build_network(),
                                brute_force=True))
    # transforms.append(AdaptiveSigmoidFixedOffset(n_sigmoids=200, features=2))
    # transforms.append(ActNorm(features=2))
    #
    # transforms.append(QRLinear(features=2, num_householder=2))
    # transforms.append(AdaptiveSigmoid(features=2, n_sigmoids=200))

# transforms.append(LULinear(features=2))

# transforms.append(SylvesterTransform(features=2))


transform = CompositeTransform(transforms)

flow = Flow(transform, base_dist).to(device)
optimizer = optim.Adam(flow.parameters(), lr=1e-3)

num_iter = 4001
for i in range(num_iter):
    x, y = datasets.make_moons(4 * 8 * 512, noise=.1)
    x = torch.tensor(x, dtype=torch.float32).to(device)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x).mean()
    if (i % 50) == 0:
        print(f"{i:04}: {loss=:.3f}")
    loss.backward()
    optimizer.step()

    if (i + 1) % 250 == 0:
        flow.eval()
        nsamples = 100
        xline = torch.linspace(-1.5, 2.5, nsamples).to(device)
        yline = torch.linspace(-.75, 1.25, nsamples).to(device)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)

        with torch.no_grad():
            zgrid = flow.log_prob(xyinput).exp().reshape(nsamples, nsamples)

            # samples = flow.sample(num_samples=1_000)

        plt.contourf(xgrid.detach().cpu().numpy(), ygrid.detach().cpu().numpy(), zgrid.detach().cpu().numpy())
        # plt.scatter(*samples.detach().cpu().numpy().T, marker='+', alpha=0.5, color="black")
        plt.title('iteration {}'.format(i + 1))
        plt.show()
        flow.train()
