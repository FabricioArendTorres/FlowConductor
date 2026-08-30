import numpy as np
import torch
from torch import nn

from flowcon.transforms.base import Sequential, Transform
from flowcon.transforms.linear.linear import ScalarScale, ScalarShift
from flowcon.transforms.nonlinearities import Exp, Sigmoid, Softplus

fancy_exp_transform = Sequential(
    [
        Sigmoid(),
        ScalarScale(scale=80.0, trainable=False),
        Exp(),
        ScalarShift(1e-5, trainable=False),
    ]
)

fancy_softplus_transform = Sequential(
    [
        Sigmoid(),
        ScalarScale(scale=80.0, trainable=False),
        Softplus(),
        ScalarShift(1e-5, trainable=False),
    ]
)


class TransformDiagonal(Transform):
    def __init__(self, N, diag_transformation: Transform = Exp()):
        super().__init__()
        self.N = N
        self.diag_indices = np.diag_indices(self.N)
        self.diag_mask = nn.Parameter(torch.diag_embed(torch.ones(1, self.N)), requires_grad=False)
        self.diag_transform = diag_transformation

        # self.transform = CompositeTransform([Sigmoid(), ScalarScale(scale=self.MAX_EXP, trainable=False)])

    def forward(self, inputs, context=None):
        transformed_diag, logabsdet_diag = self.diag_transform(
            torch.diagonal(inputs, dim1=-2, dim2=-1)
        )
        outputs = torch.diagonal_scatter(inputs, transformed_diag, dim1=-2, dim2=-1)
        return outputs, logabsdet_diag

    def inverse(self, inputs, context=None):
        transformed_diag, logabsdet_diag = self.diag_transform.inverse(
            torch.diagonal(inputs, dim1=-2, dim2=-1)
        )
        outputs = torch.diagonal_scatter(inputs, transformed_diag, dim1=-2, dim2=-1)
        return outputs, logabsdet_diag


class TransformDiagonalExponential(TransformDiagonal):
    def __init__(self, N, eps=1e-5):
        super().__init__(
            N=N,
            diag_transformation=Sequential([Exp(), ScalarShift(eps, trainable=False)]),
        )


class TransformDiagonalSoftplus(TransformDiagonal):
    def __init__(self, N, eps=1e-5):
        super().__init__(
            N=N,
            diag_transformation=Sequential([Softplus(), ScalarShift(eps, trainable=False)]),
        )
