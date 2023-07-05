"""
A very basic application of the LFlows.

Interpolation of the two half moons of the two-moons data set.
Here, we train with a maximum likelihood objective and also visualize the velocity.

"""
import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from enflows.transforms.autoregressive import MaskedDeepSigmoidTransform
from torch import nn
from torch import optim

from enflows.flows import Flow
from enflows.distributions.normal import DiagonalNormal
from enflows.transforms.base import CompositeTransform
from enflows.transforms.conditional import *
from enflows.transforms import *
from enflows.transforms.permutations import ReversePermutation
from enflows.nn.nets import ResidualNet

num_layers = 4
base_dist = DiagonalNormal(shape=[2])

transforms = []
context_features = 5
# transforms.append(ConditionalLUTransform(features=2, context_features=1, hidden_features=32))
for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2))
    transforms.append(ActNorm(2))
    transforms.append(ConditionalShiftTransform(features=2, context_features=context_features, hidden_features=64))

    transforms.append(MaskedSumOfSigmoidsTransform(features=2, n_sigmoids=10,
                                                        context_features=context_features, hidden_features=64))
transforms.append(ConditionalShiftTransform(features=2, context_features=context_features, hidden_features=32))
transforms.append(ActNorm(2))

transform = CompositeTransform(transforms)

embedding_net = ResidualNet(in_features=1, out_features=context_features, hidden_features=32,
                            num_blocks=2,
                            activation=torch.nn.functional.silu)

flow = LagrangeFlow(transform, base_dist, embedding_net=embedding_net)
optimizer = optim.Adam(flow.parameters(), lr=1e-4)

num_iter = 5000
for i in range(num_iter):
    x, y = datasets.make_moons(128, noise=.1)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    optimizer.zero_grad()
    loss = -flow.log_prob(inputs=x, context=y).mean()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print('iteration {}; Loss: {:.2e}'.format(i + 1, loss.item()))

    if (i + 1) % 500 == 0:
        fig, ax = plt.subplots(1, 5,  figsize=(20, 4))
        ax = ax.flatten()
        xline = torch.linspace(-1.5, 2.5, 100)
        yline = torch.linspace(-.75, 1.25, 100)
        xgrid, ygrid = torch.meshgrid(xline, yline)
        xgrid_vel, ygrid_vel = torch.meshgrid(xline[::5], yline[::5])
        xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
        xyinput_vel = torch.cat([xgrid_vel.reshape(-1, 1), ygrid_vel.reshape(-1, 1)], dim=1)

        with torch.no_grad():

            zgrid0 = flow.log_prob(xyinput, torch.zeros(10000, 1)).exp().reshape(100, 100)
            zgrid1 = flow.log_prob(xyinput, 0.25 * torch.ones(10000, 1)).exp().reshape(100, 100)
            zgrid2 = flow.log_prob(xyinput, 0.5 * torch.ones(10000, 1)).exp().reshape(100, 100)
            zgrid3 = flow.log_prob(xyinput, 0.75 * torch.ones(10000, 1)).exp().reshape(100, 100)
            zgrid4 = flow.log_prob(xyinput, torch.ones(10000, 1)).exp().reshape(100, 100)

            vels = [flow.velocity(xyinput_vel, t*torch.ones(xyinput_vel.shape[0], 1)).detach().cpu().numpy() for t in [0., 0.25, 0.5, 0.75, 1.]]

        for i, (zgrid, vel) in enumerate(zip([zgrid0, zgrid1, zgrid2, zgrid3, zgrid4],
                                             vels)):
            ax[i].contourf(xgrid.numpy(), ygrid.numpy(), zgrid.numpy(), cmap="Blues")
            ax[i].quiver(xgrid_vel.numpy(), ygrid_vel.numpy(), vel[:, 0], vel[:, 1], angles='xy', scale_units='xy')
            # ax[i].axis('equal')
            ax[i].set_ylim(-1.5, 1.5)


        # ax[1].contourf(xgrid.numpy(), ygrid.numpy(), zgrid1.numpy())
        plt.title('iteration {}'.format(i + 1))
        # plt.show()
        plt.show()
