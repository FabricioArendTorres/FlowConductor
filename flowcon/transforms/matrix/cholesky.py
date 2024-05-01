import numpy as np
import torch
from torch import nn
from flowcon.transforms import Transform, ConditionalTransform, Exp, Sigmoid, ScalarScale, CompositeTransform, \
    ScalarShift, Softplus


class CholeskyOuterProduct(Transform):
    def __init__(self, N, checkargs=True, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.N = N
        self.eye = nn.Parameter(torch.diag_embed(torch.ones(self.N)).unsqueeze(0), requires_grad=False)
        self.powers = nn.Parameter(torch.arange(self.N, 0, -1).unsqueeze(0), requires_grad=False)
        self.checkargs = checkargs

    def forward(self, inputs, context=None):
        if self.checkargs:
            self.check_pos_low_triang(inputs)
        outputs = torch.bmm(inputs, inputs.mT)
        outputs = 0.5*(outputs + outputs.mT)
        diagonal = torch.diagonal(inputs, dim1=-2, dim2=-1)
        logabsdet = self.N * np.log(2.) + (self.powers * diagonal.log()).sum(-1)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        inputs_jitter = inputs + self.eye * self.eps
        if self.checkargs:
            self.check_pos_def(inputs_jitter)

        outputs = torch.linalg.cholesky(inputs_jitter, upper=False)
        diagonal = torch.diagonal(outputs, dim1=-2, dim2=-1)
        logabsdet = self.N * np.log(2.) + (self.powers * diagonal.log()).sum(1)

        return outputs, -logabsdet

    def check_pos_low_triang(self, inputs):
        assert inputs.shape[-2] == inputs.shape[-1], "input tensor must be mini batch of square matrices"
        upper_indices = np.triu_indices(self.N, k=1)
        assert torch.all(inputs[:, upper_indices[0], upper_indices[1]] == 0.), (
            "input tensor must be mini batch of lower triangular matrices")
        assert torch.all(torch.diagonal(inputs, dim1=-2, dim2=-1) > 0), (
            'input tensor must be mini batch of lower triangular matrices with positive diagonal elements')

    def check_pos_def(self, inputs):
        assert torch.all(inputs == inputs.mT), "Input matrix is not symmetric."
        assert torch.all(torch.linalg.eig(inputs)[0].real >= 0), (
            "Input matrix is not positive semi-definite in order to perform Cholesky decomposition"
        )
