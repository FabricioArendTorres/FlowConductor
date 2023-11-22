import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from enflows.transforms import Transform, ConditionalTransform, Exp, Sigmoid, ScalarScale, CompositeTransform, \
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

class PositiveDefiniteAndUnconstrained(Transform):

    def __init__(self, features, p_features, jitter=1e-7):
        super().__init__()
        self.jitter = jitter
        self.MAX_EXP = 88.
        self.features = features
        self.p_features = p_features
        self.P = self.calc_numb_ltri_dim(p_features)
        self.Q = (self.features - self.p_features) // self.P
        assert self.Q * self.P + self.p_features == self.features

        # indices for lower triangular entries
        self.lower_indices = np.tril_indices(self.P, k=0)

        # mask for sofplus over diagonal
        PxP_diag = torch.diag_embed(torch.ones(1, self.P))
        PxPplusQ_diag = F.pad(input=PxP_diag, pad=(0, self.Q, 0, 0), mode='constant', value=0)
        self.diag_mask_PxP = nn.Parameter(PxP_diag, requires_grad=False)
        self.diag_mask_PxPplusQ = nn.Parameter(PxPplusQ_diag, requires_grad=False)
        self.diag_indices = np.diag_indices(self.P)

        # powers for cholesky outer product jacobian determinant
        self.powers = nn.Parameter(torch.arange(self.P, 0, -1).unsqueeze(0), requires_grad=False)
        self.eye = nn.Parameter(torch.diag_embed(torch.ones(self.P)).unsqueeze(0), requires_grad=False)

    @staticmethod
    def calc_numb_ltri_dim(p):
        assert p > 0, "dimension must be positive number"
        temp = 1 + 8 * p
        assert np.square(
            np.floor(np.sqrt(temp))) == temp, "invalid dimension: can't be mapped to lower triangular matrix"
        N = int((-1 + np.floor(np.sqrt(temp))) // 2)

        return N

    def forward(self, inputs, context=None):
        mb, features = inputs.shape
        assert features == self.features

        # 1) fill P x (P + Q) matrix
        outputs_fill = inputs.new_zeros((mb, self.P, self.P + self.Q))
        outputs_fill[:, self.lower_indices[0], self.lower_indices[1]] = inputs[:, :self.p_features]
        outputs_fill[:, :, self.P:] = inputs[:, self.p_features:].view(-1, self.P, self.Q)

        if torch.any(torch.isnan(outputs_fill)): breakpoint()

        # logabsdet is zero

        # 2) take softplus over diagonal of submatrix P x P
        softplus_diag = F.softplus(outputs_fill) + self.jitter
        outputs_soft = self.diag_mask_PxPplusQ * softplus_diag + (1. - self.diag_mask_PxPplusQ) * outputs_fill

        if torch.any(torch.isnan(outputs_soft)): breakpoint()

        logabsdet_soft = - (F.softplus(-torch.diagonal(outputs_fill, dim1=-2, dim2=-1)) + self.jitter).sum(-1)  # maybe jitter creates problems here

        # 3) compute Cholesky outer product on submatrix P x P
        self.check_pos_low_triang(outputs_soft)

        diagonal = torch.diagonal(outputs_soft, dim1=-2, dim2=-1)
        outputs = inputs.new_zeros((mb, self.P, self.P + self.Q))
        outputs[:, :self.P, :self.P] = outputs_soft[:, :self.P, :self.P] @ outputs_soft[:, :self.P, :self.P].mT
        outputs[:, :self.P, self.P:] = outputs_soft[:, :self.P, self.P:]

        if torch.any(torch.isnan(outputs)): breakpoint()

        logabsdet_chol = self.P * np.log(2.) + (self.powers * diagonal.log()).sum(-1)

        return outputs, logabsdet_soft + logabsdet_chol

    def inverse(self, inputs, context=None):
        assert inputs.shape[1] == self.P and inputs.shape[2] == self.P + self.Q

        # 1) compute Cholesky decomposition
        inputs_jitter = inputs[:,:self.P, :self.P] + self.eye * self.jitter
        self.check_pos_def(inputs_jitter)

        outputs_low_tri = torch.linalg.cholesky(inputs_jitter, upper=False)
        if torch.any(torch.isnan(outputs_low_tri)): breakpoint()

        diagonal = torch.diagonal(outputs_low_tri, dim1=-2, dim2=-1)
        logabsdet_chol = self.N * np.log(2.) + (self.powers * diagonal.log()).sum(1)

        # 2) inverse softplus over the diagonal
        self.check_pos_low_triang(outputs_low_tri)

        inv_softplus_diag = (outputs_low_tri.abs().clamp(max=self.MAX_EXP).exp() - 1. + self.jitter).log()
        outputs_inv_soft = self.diag_mask_PxP * inv_softplus_diag + (1. - self.diag_mask_PxP) * outputs_low_tri

        if torch.any(torch.isnan(outputs_inv_soft)): breakpoint()

        der_inv_softplus = 1. - torch.diagonal(- outputs_low_tri, dim1=-2, dim2=-1).clamp(max=self.MAX_EXP).exp() + self.jitter
        logabsdet_inv_soft = - der_inv_softplus.log().sum(-1)

        # 3) ravel P x (P + Q) matrix into a vector

        mb = inputs.shape[0]
        outputs = inputs.new_zeros((mb, self.p_features + self.P * self.Q))
        outputs[:, :self.p_features] = outputs_inv_soft[:, self.lower_indices[0], self.lower_indices[1]]
        outputs[:,self.p_features:] = inputs[:, :, self.P:]

        # logabsdet is zero

        return outputs, logabsdet_chol + logabsdet_inv_soft

    def check_pos_low_triang(self, inputs):
        upper_indices = np.triu_indices(self.P, k=1)
        assert torch.all(inputs[:, upper_indices[0], upper_indices[1]] == 0.), (
            "input tensor must be mini batch of lower triangular matrices")
        assert torch.all(torch.diagonal(inputs, dim1=-2, dim2=-1) > 0), (
            'input tensor must be mini batch of lower triangular matrices with positive diagonal elements')

    def check_pos_def(self, inputs):
        assert torch.all(inputs[:, :self.P, :self.P] == inputs[:, :self.P, :self.P].mT), (
            "input matrix must be symmetric"
        )
        assert  torch.all(torch.linalg.eig(inputs[:,:self.P, :self.P])[0].real >= 0), (
            "input matrix must be symmetric positive semi-definite in order to perform Cholesky decomposition"
        )