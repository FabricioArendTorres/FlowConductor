import matplotlib.pyplot as plt
import os

import torch
from torch import optim


import flowcon.flows as flows
import flowcon.distributions as distributions
import flowcon.transforms as transforms
import flowcon.nn as nn

from flowcon.nn.neural_odes import ODEnet, ODEfunc

from flowcon.datasets import load_plane_dataset, InfiniteLoader

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

CONTINUE_TRAINING = False

os.makedirs("figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

base_dist = distributions.StandardNormal(shape=[2])
MB_SIZE = 500
selected_data = "two_spirals"
num_layers = 1

num_iter = {"eight_gaussians": 3_000,
            "diamond": 50_000,
            "crescent": 3_000,
            "four_circles": 3_000,
            "two_circles": 3_000
            }.get(selected_data, 10_000)


def main():

    # create data
    train_dataset = load_plane_dataset(selected_data, int(1e7))
    train_loader = InfiniteLoader(
        dataset=train_dataset,
        batch_size=MB_SIZE,
        shuffle=True,
        drop_last=True,
        num_epochs=None
    )
    test_loader = InfiniteLoader(
        dataset=train_dataset,
        batch_size=10_000,
        shuffle=True,
        drop_last=True,
        num_epochs=None
    )

    transform = build_transform(num_layers=num_layers)
    flow = flows.Flow(transform, base_dist).to(device)


    optimizer = optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9995)
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
            if (i + 1) % 250 == 0:
                with torch.no_grad():
                    flow.eval()
                    x = next(test_loader).to(device)
                    test_loss = -flow.log_prob(inputs=x).mean()
                    print(f"{i:04}: {test_loss=:.3f} {scheduler.get_last_lr()}")

                    plot_model(flow)
                    flow.train()

    except KeyboardInterrupt:
        pass
    plot_model(flow)


def build_transform(n_features=2, num_layers=5) -> transforms.Transform:
    ode_net = ODEnet((32, 32, 32), input_shape=(n_features,),
                     layer_type="hyper")
    transform_list = []
    transform_list.append(transforms.neuralode.SimpleCNF(ode_net,
                                                         divergence_fn="brute_force"))
    transform = transforms.CompositeTransform(transform_list)
    return transform


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_model(flow):
    flow.eval()
    nsamples = 200
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axs = axs.flatten()
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
        zgrid = flow.log_prob(xyinput).reshape(nsamples, nsamples).exp()

        #samples = flow.sample(num_samples=1_000)
    # plt.contourf(xgrid.detach().cpu().numpy(), ygrid.detach().cpu().numpy(), zgrid.detach().cpu().numpy())
    axs[0].imshow(zgrid.detach().cpu().numpy(), origin='lower', extent=[-4, 4, -4, 4],
                  cmap="inferno")
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].set_aspect('equal', 'box')
    axs[1].set_aspect('equal', 'box')
    #axs[1].scatter(*samples.detach().cpu().numpy().T, marker='+', alpha=0.1, color="black")
    plt.tight_layout()
    # plt.title('iteration {}'.format(i + 1))
    plt.savefig(f"figures/{selected_data}.png")
    plt.close()
    print("Saved plot to figures/{}.png".format(selected_data))

if __name__ == "__main__":
    main()