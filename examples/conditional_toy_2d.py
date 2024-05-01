"""
A very basic application of the Conditional Flow.
"""
import matplotlib.pyplot as plt

from torch import optim

from flowcon.flows import Flow
from flowcon.distributions.normal import DiagonalNormal
from flowcon.transforms import (ActNorm,
                                iResBlock,
                                CompositeTransform,
                                )
from flowcon.nn.nets import *
from flowcon.utils.torchutils import *
from flowcon.datasets.base import load_plane_dataset, InfiniteLoader, PlaneDataset
import logging

logging.basicConfig(
    format='%(asctime)s  %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S')

logger = logging.getLogger()
device = "cpu"  # "cuda"

###########################################
selected_data = "pinwheel"
MB_SIZE = 1024
num_iter = 1000

assert selected_data in ["two_moons", "two_circles", "eight_gaussians", "swissroll", "pinwheel"]


def main():
    flow = build_flow()
    optimizer = optim.Adam(flow.parameters(),
                           lr=1e-3,
                           weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=1e-7, T_max=num_iter)

    # create data
    train_dataset = load_plane_dataset(selected_data, int(1e7), return_label=True)
    train_loader = InfiniteLoader(
        dataset=train_dataset,
        batch_size=MB_SIZE,
        shuffle=True,
        drop_last=True,
        num_epochs=None,
        num_workers=10
    )

    for i in range(num_iter):
        x, y = next(train_loader)
        x, y = x.to(device), y.to(device).view(-1, 1)

        # x, y = datasets.make_moons(128, noise=.1)
        optimizer.zero_grad()
        loss = -flow.log_prob(inputs=x, context=y).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 50 == 0:
            logger.info('iteration {}; Loss: {:.2e}'.format(i + 1, loss.item()))

        if (i + 1) % 100 == 0:
            plot_model(flow, train_dataset)


def build_flow(num_layers=5,
               num_shared_embedding=50):
    base_dist = DiagonalNormal(shape=[2])

    densenet_factory = iResBlock.Factory()
    # First set on how to estimate the logabsdet of the lipschitz constrained neural network.
    # Use brute_force for low dimensions (i.e. directly via autograd).
    # Else, use stochastic estimator.
    densenet_factory.set_logabsdet_estimator(brute_force=True,  # set this to false for high dimensions (>3)
                                             # unbiased_estimator=True,  # default;
                                             # trace_estimator="neumann"  # either "neumann" or "basic";
                                             )

    # Then select on how to condition the lipschitz constrained neural network
    # Combinations are possible, but not all of them implemented.
    densenet_factory.set_densenet(condition_input=True,
                                  condition_lastlayer=False,
                                  condition_multiplicative=True,
                                  ###
                                  dimension=2,
                                  densenet_depth=3,
                                  densenet_growth=32,
                                  c_embed_hidden_sizes=(128, 128, 10),
                                  m_embed_hidden_sizes=(128, 128),
                                  activation_function=Sin(10),
                                  lip_coeff=.97,
                                  context_features=num_shared_embedding)

    transforms = []
    for i in range(num_layers):
        transforms.append(ActNorm(2))
        transforms.append(densenet_factory.build())

    transform = CompositeTransform(transforms)
    embedding_net = ResidualNet(in_features=1, out_features=num_shared_embedding, hidden_features=32,
                                num_blocks=2,
                                activation=torch.nn.functional.silu)
    flow = Flow(transform, base_dist, embedding_net=embedding_net)
    return flow.to(device)


def plot_model(flow: Flow, dataset: PlaneDataset):
    fig, ax = plt.subplots(2, 10, figsize=(20, 4))
    ax = ax.flatten()
    xline = torch.linspace(-4., 4., 100).to(device)
    yline = torch.linspace(-4., 4., 100).to(device)
    xgrid, ygrid = torch.meshgrid(xline, yline)
    xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)], dim=1)
    flow.eval()
    with torch.no_grad():
        zgrids = []
        for i in np.linspace(dataset.min_label, dataset.max_label, 20):
            condition = i * torch.ones(10000, 1).to(device)
            zgrid = flow.log_prob(xyinput, condition).exp().reshape(100, 100)
            zgrids.append(zgrid)
    for i, zgrid in enumerate(zgrids):
        ax[i].contourf(tensor_to_np(xgrid),
                       tensor_to_np(ygrid),
                       tensor_to_np(zgrid), cmap="Blues")
        ax[i].set_ylim(-4, 4)
        ax[i].set_xlim(-4, 4)
    flow.train()
    # ax[1].contourf(xgrid.numpy(), ygrid.numpy(), zgrid1.numpy())
    plt.title('iteration {}'.format(i + 1))
    plt.tight_layout()
    # plt.show()
    plt.show()


if __name__ == "__main__":
    main()
