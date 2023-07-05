import torch
import torch.nn as nn

from enflows.transforms.no_analytic_inv.base import MonotonicTransform
from enflows.utils import torchutils


class ExtendedSoftplus(torch.nn.Module):
    """
    Combination of a (shifted and scaled) softplus and the same softplus flipped around the origin

    Softplus(scale * (x-shift)) - Softplus(-scale * (x + shift))

    Linear outside of origin, flat around origin.
    """

    def __init__(self, features, shift=None):
        self.features = features
        super(ExtendedSoftplus, self).__init__()
        if shift is None:
            self.shift = torch.nn.Parameter(torch.ones(1, features) * 3, requires_grad=True)
            # self.log_scale = torch.nn.Parameter(torch.zeros(1, features), requires_grad=True)
        elif torch.is_tensor(shift):
            self.shift = shift.reshape(-1, features)
            # self.log_scale = log_scale.reshape(-1, features)
        else:
            self.shift = torch.nn.Parameter(torch.tensor(shift), requires_grad=True)
            # self.log_scale = torch.nn.Parameter(torch.tensor(log_scale), requires_grad=True)

        self._softplus = torch.nn.Softplus()

    # def get_shift_and_scale(self):
    #     # return self._softplus(self.shift), torch.exp(self.log_scale)
    #     return self.shift, torch.exp(self.log_scale) + 1e-3
    #     # return 5, torch.exp(self.log_scale)

    def get_shift(self):
        return self._softplus(self.shift) + 1e-3

    def softplus(self, x, shift):
        return self._softplus((x - shift))

    def softminus(self, x, shift):
        return - self._softplus(-(x + shift))

    def diag_jacobian_pos(self, x, shift):
        # (b e^(b x))/(e^(a b) + e^(b x))
        return torch.exp(x) / (torch.exp(shift) + torch.exp(x))

    def log_diag_jacobian_pos(self, x, shift):
        # -log(e^(a b) + e^(b x)) + b x + log(b)
        log_jac = -torch.logaddexp(shift, x) + x
        return log_jac

    def diag_jacobian_neg(self, x, shift):
        return torch.sigmoid(- (shift + x))

    def log_diag_jacobian_neg(self, x, shift):
        return - self._softplus((shift + x))

    def forward(self, inputs):
        # inputs = inputs.requires_grad_()
        shift = self.get_shift()
        outputs = self.softplus(inputs, shift) + self.softminus(inputs, shift)
        # ref_batch_jacobian = torchutils.batch_jacobian(outputs, inputs)
        # ref_logabsdet = torchutils.logabsdet(ref_batch_jacobian)
        # breakpoint()
        diag_jacobian = torch.logaddexp(self.log_diag_jacobian_pos(inputs, shift),
                                        self.log_diag_jacobian_neg(inputs, shift))
        return outputs, diag_jacobian  # torch.log(diag_jacobian).sum(-1)


class SumOfSigmoids(MonotonicTransform):
    PREACT_SCALE_MIN = .1
    PREACT_SCALE_MAX = 10.
    PREACT_SHIFT_MAX = 2

    def __init__(self, features, n_sigmoids=10, num_iterations=50, lim=120,
                 raw_params: torch.Tensor = None):
        self.n_sigmoids = n_sigmoids
        self.features = features

        super(SumOfSigmoids, self).__init__(num_iterations=num_iterations, lim=lim)
        if raw_params is None:
            self.shift_preact = nn.Parameter(torch.randn(1, features, self.n_sigmoids), requires_grad=True)
            self.log_scale_preact = nn.Parameter(torch.zeros(1, features, self.n_sigmoids), requires_grad=True)
            self.raw_softmax = nn.Parameter((torch.ones(1, features, self.n_sigmoids, requires_grad=False)))
            self.extended_softplus = ExtendedSoftplus(features=features)
        else:
            assert raw_params.shape[1:] == (features, 3 * self.n_sigmoids + 1)
            self.set_raw_params(features, raw_params)

        self.log_scale_postact = nn.Parameter(torch.log(torch.ones(1, device=self.shift_preact.device)),
                                              requires_grad=False)
        self.eps = 1e-6

    def get_raw_params(self):
        return torch.cat((self.shift_preact.reshape(-1, self.features, self.n_sigmoids),
                          self.log_scale_preact.reshape(-1, self.features, self.n_sigmoids),
                          self.raw_softmax.reshape(-1, self.features, self.n_sigmoids),
                          self.extended_softplus.shift.reshape(-1, self.features, 1),
                          # self.extended_softplus.log_scale.reshape(-1, self.features, 1)
                          ), dim=-1)

    def set_raw_params(self, features, raw_params):
        # 3 = shift, scale, softmax for sigmoids
        # 2 = log_scale, log_shift for extended softplus
        vals = torch.split(raw_params, [self.n_sigmoids, self.n_sigmoids, self.n_sigmoids, 1], dim=-1)
        self.shift_preact, self.log_scale_preact, self.raw_softmax = vals[:3]
        self.extended_softplus = ExtendedSoftplus(features=features, shift=vals[3])

    def get_sigmoid_params(self, features, n_features_x_sigmoids, unconstrained_params):
        shift_preact = unconstrained_params[:, :features * self.n_sigmoids]
        shift_preact = shift_preact.view(-1, features, self.n_sigmoids)

        log_scale_preact = unconstrained_params[:, n_features_x_sigmoids: 2 * n_features_x_sigmoids]
        log_scale_preact = log_scale_preact.view(-1, features, self.n_sigmoids)

        raw_softmax_preact = unconstrained_params[:, 2 * n_features_x_sigmoids: 3 * n_features_x_sigmoids]
        raw_softmax_preact = raw_softmax_preact.view(-1, features, self.n_sigmoids)

        return shift_preact, log_scale_preact, raw_softmax_preact

    def sigmoid_log_derivative(self, x):
        return x - 2 * torch.nn.functional.softplus(x)

    def forward(self, inputs, context=None):
        output_sum_of_sigmoids, log_diag_jac_sigmoids = self.sum_of_sigmoids(inputs)
        output_extended_softplus, log_diag_jac_esoftplus = self.extended_softplus(inputs)

        output = output_sum_of_sigmoids + output_extended_softplus
        logabsdet = torch.logaddexp(log_diag_jac_sigmoids, log_diag_jac_esoftplus).sum(-1)

        return output, logabsdet

    def sum_of_sigmoids(self, inputs):
        shift_preact, scale_preact, scale_postact = self.get_params()

        pre_act = scale_preact * (inputs.unsqueeze(-1) - shift_preact)

        sigmoids_expanded = scale_postact * torch.sigmoid(pre_act)
        log_jac_sigmoid_expanded = torch.log(scale_postact) + torch.log(scale_preact) + self.sigmoid_log_derivative(
            pre_act)
        tmp = sigmoids_expanded.sum(-1) / (scale_postact.sum(-1))

        return tmp, torch.logsumexp(log_jac_sigmoid_expanded, -1)

    def get_params(self):
        soft_max = torch.nn.functional.softmax(self.raw_softmax, dim=-1) + self.eps
        soft_max /= soft_max.sum(-1).unsqueeze(-1)
        scale_postact = torch.exp(self.log_scale_postact) *soft_max

        scale_preact = torch.sigmoid(self.log_scale_preact)
        scale_preact = scale_preact * (self.PREACT_SCALE_MAX - self.PREACT_SCALE_MIN) + self.PREACT_SCALE_MIN

        shift_preact = torch.tanh(self.shift_preact) * self.PREACT_SHIFT_MAX

        return shift_preact, scale_preact, scale_postact
