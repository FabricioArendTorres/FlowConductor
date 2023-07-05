import matplotlib.pyplot as plt
import sklearn.datasets as datasets

import torch
from torch import nn
from torch import optim

from enflows.flows.base import Flow
from enflows.distributions.normal import StandardNormal
from enflows.transforms.base import CompositeTransform
from enflows.transforms import *
from enflows.transforms.autoregressive import *
from enflows.transforms.permutations import ReversePermutation
from datasets.base import load_plane_dataset, InfiniteLoader

device = "cuda"

LOAD_MODEL = False
SAVE_MODEL = True
CONTINUE_TRAINING = False

base_dist = StandardNormal(shape=[2])
MB_SIZE = 2 * 8 * 1024
# selected_data = "diamond"
selected_data = "eight_gaussians"

num_layers = {"eight_gaussians": 6,
              "diamond": 10,
              "crescent": 4,
              "four_circles": 4,
              "two_circles": 6,
              "checkerboard": 4,
              "two_spirals": 6
              }.get(selected_data, 4)
num_sigmoids = {"eight_gaussians": 30,
                "diamond": 30,
                "crescent": 30,
                "four_circles": 30,
                "two_circles": 100,
                "checkerboard": 100,
                "two_spirals": 50
                }.get(selected_data, 30)

num_iter = {"eight_gaussians": 3_000,
            "diamond": 50_000,
            "crescent": 3_000,
            "four_circles": 3_000,
            "two_circles": 3_000
            }.get(selected_data, 6_000)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_model(flow, x):
    flow.eval()
    nsamples = 1_000
    x_min, x_max, y_min, y_max = dict(
        two_spirals=[-4, 4, -4, 4],
        checkerboard=[-4, 4, -4, 4]
    ).get(selected_data, [-4, 4, -4, 4])
    # x_min = torch.floor(x.min(0)[0][0]) - 1e-1
    # y_min = torch.floor(x.min(0)[0][1]) - 1e-1
    # x_max = torch.ceil(x.max(0)[0][0]) + 1e-1
    # y_max = torch.ceil(x.max(0)[0][1]) + 1e-1
    xline = torch.linspace(x_min, x_max, nsamples).to(device)
    yline = torch.linspace(y_min, y_max, nsamples).to(device)
    xgrid, ygrid = torch.meshgrid(xline, yline, indexing='xy')
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
    # flow.train()
    # flow.eval()
    with torch.no_grad():
        zgrid = flow.log_prob(xyinput).exp().reshape(nsamples, nsamples)

        samples = flow.sample(num_samples=1_000)
    # plt.contourf(xgrid.detach().cpu().numpy(), ygrid.detach().cpu().numpy(), zgrid.detach().cpu().numpy())
    plt.imshow(zgrid.detach().cpu().numpy(), origin='lower')
    plt.axis('off')
    plt.gca().set_aspect('equal', 'box')
    plt.tight_layout()
    # plt.scatter(*samples.detach().cpu().numpy().T, marker='+', alpha=0.5, color="black")
    # plt.title('iteration {}'.format(i + 1))
    plt.savefig(f"{selected_data}.png")


# create data
train_dataset = load_plane_dataset(selected_data, int(1e6))
train_loader = InfiniteLoader(
    dataset=train_dataset,
    batch_size=MB_SIZE,
    shuffle=True,
    drop_last=True,
    num_epochs=None
)
transforms = []

hypernet_kwargs = dict(features=2, hidden_features=128, num_blocks=2)
# made_RQ = MaskedPiecewiseRationalQuadraticAutoregressiveTransform(num_bins=8, tails='linear', tail_bound=3,
#                                                                   **hypernet_kwargs)
# made_sigmoids = MaskedSumOfSigmoidsTransform(n_sigmoids=8, **hypernet_kwargs)

for _ in range(num_layers):
    transforms.append(ReversePermutation(features=2))
    transforms.append(LULinear(features=2, identity_init=True))
    transforms.append(ActNorm(features=2))

    # transforms.append(
    #     MaskedPiecewiseRationalQuadraticAutoregressiveTransform(features=2, hidden_features=256, num_bins=8,
    #                                                             tails='linear', num_blocks=2, tail_bound=3,
    #                                                             ))
    transforms.append(MaskedSumOfSigmoidsTransform(n_sigmoids=num_sigmoids, **hypernet_kwargs))
    # transforms.append(MaskedDeepSigmoidTransform(n_sigmoids=num_sigmoids, **hypernet_kwargs))

transform = CompositeTransform(transforms)

if LOAD_MODEL:
    flow = torch.load(f"{selected_data}.pt")
else:
    flow = Flow(transform, base_dist).to(device)

x = next(train_loader).to(device)
if not LOAD_MODEL or CONTINUE_TRAINING:
    optimizer = optim.Adam(flow.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-6, T_max=num_iter)
    try:
        for i in range(num_iter):
            x = next(train_loader).to(device)
            optimizer.zero_grad()
            loss = -flow.log_prob(inputs=x).mean()
            if (i % 50) == 0:
                print(f"{i:04}: {loss=:.3f}")
            loss.backward()
            optimizer.step()
            scheduler.step()
            if (i + 1) % 500 == 0:
                plot_model(flow, x)

    except KeyboardInterrupt:
        pass
if SAVE_MODEL:
    torch.save(flow, f"{selected_data}.pt")
plot_model(flow, x)
