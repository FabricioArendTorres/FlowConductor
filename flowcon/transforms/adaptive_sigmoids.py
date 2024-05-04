import abc

import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod

from flowcon.transforms.no_analytic_inv.base import MonotonicTransform
from flowcon.transforms.base import Transform
from flowcon.transforms.nonlinearities import ExtendedSoftplus, Sigmoid, Softplus


class SumOfSigmoids(MonotonicTransform):
    """
    Implements non-linear elementwise transformation as the sum of multiple shifted scaled sigmoid functions,
    which are combined with an extended softplus function to get linear behaviour far away from the origin.

    See Appendix A.1 in [1] for more details.

    [1] Negri, Marcello Massimo, Fabricio Arend Torres, and Volker Roth. "Conditional Matrix Flows for Gaussian Graphical Models." Advances in Neural Information Processing Systems 36 (2024).
    """
    PREACT_SCALE_MIN = .1
    PREACT_SCALE_MAX = 10.
    PREACT_SHIFT_MAX = 10

    def __init__(self, features, n_sigmoids=10, iterations_bisection_inverse=50, lim_bisection_inverse=120,
                 raw_params: torch.Tensor = None):
        """
        Initialize the SumOfSigmoids transformation.

        Parameters
        ----------
        features : int
            The number of features for each input. This defines the dimensionality of the input space.
        n_sigmoids : int, optional
            Number of sigmoid functions to apply per feature. This controls the complexity of the transformation. Default is 10.
        iterations_bisection_inverse : int, optional
            Max number of iterations for computing the numerical inverse with bisection search if it doesn't converge. Default is 50.
        lim_bisection_inverse : int, optional
            [-lim_bisection_inverse, lim_bisection_inverse] provides the search region for the inverse via bisection search . Default is 120.
        raw_params : torch.Tensor, optional
            A tensor containing pre-initialized parameters for the transformation. If provided, the parameters are set directly from this tensor; otherwise, they are initialized internally.

        Raises
        ------
        AssertionError
            If `raw_params` is provided but does not match the required shape of (features, 3 * n_sigmoids + 1).

        Notes
        -----
        This constructor sets up the transformation by initializing parameters, either from the `raw_params` tensor
        or by creating new parameters if `raw_params` is None.
        """
        self.n_sigmoids = n_sigmoids
        self.features = features

        super(SumOfSigmoids, self).__init__(num_iterations=iterations_bisection_inverse, lim=lim_bisection_inverse)
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
        """
        Concatenate and return all raw parameters of the transformation in a single tensor.
        The Tensor is of shape [self.n_sigmoids, self.features, -1].

        Returns
        -------
        torch.Tensor
            A concatenated tensor of all raw parameters, including shifts, log scales for
            the sigmoid functions, softmax weights, and the shift from the extended softplus.
        """
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
        scale_postact = torch.exp(self.log_scale_postact) * soft_max

        scale_preact = torch.sigmoid(self.log_scale_preact)
        scale_preact = scale_preact * (self.PREACT_SCALE_MAX - self.PREACT_SCALE_MIN) + self.PREACT_SCALE_MIN

        shift_preact = torch.tanh(self.shift_preact) * self.PREACT_SHIFT_MAX

        return shift_preact, scale_preact, scale_postact


class DeepSigmoidModule(Transform):
    @staticmethod
    def softmax(x, dim=-1):
        e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
        out = e_x / e_x.sum(dim=dim, keepdim=True)
        return out

    def __init__(self, n_sigmoids=4, mollify=0., eps=1e-4, num_inverse_iterations=100, lim=10):
        super(DeepSigmoidModule, self).__init__()

        self.n_sigmoids = n_sigmoids

        self.act_a = torch.nn.Softplus()
        self.act_b = torch.nn.Identity()
        self.act_w = torch.nn.Softmax(dim=-1)
        self._mollify = mollify
        self.eps = eps

        self.softplus_ = nn.Softplus()
        self.softplus = lambda x: self.softplus_(x) + self.eps
        self.sigmoid_ = nn.Sigmoid()
        self.sigmoid = lambda x: self.sigmoid_(x) * (
                1 - self.delta) + 0.5 * self.delta
        self.logsigmoid = lambda x: -self.softplus(-x)
        self.log = lambda x: torch.log(x * 1e2) - np.log(1e2)
        self.logit = lambda x: self.log(x) - self.log(1 - x)

    @abstractmethod
    def forward(self, inputs, context=None):
        dsparams = self.get_params(inputs, context)
        return self.forward_given_params(inputs, dsparams=dsparams)

    def forward_given_params(self, inputs, dsparams=None):
        scale_ = self.act_a(self.raw_scales(dsparams))
        shift_ = self.act_b(self.raw_shifts(dsparams))
        weight = self.act_w(self.raw_weights(dsparams))

        scale, shift = self.mollify(scale_, shift_)

        pre_sigm = scale * inputs.unsqueeze(-1) + shift
        x_pre = torch.sum(weight * torch.sigmoid(pre_sigm), dim=-1)
        x_pre_clipped = x_pre * (1 - self.eps) + self.eps * 0.5
        x_ = self.logit(x_pre_clipped)
        outputs = x_

        logdet = self._forward_logabsdet(scale, dsparams, self.n_sigmoids, pre_sigm, x_pre_clipped)

        return outputs, logdet

    def raw_scales(self, dsparams):
        return dsparams[..., 0 * self.n_sigmoids:1 * self.n_sigmoids]

    def raw_shifts(self, dsparams):
        return dsparams[..., 1 * self.n_sigmoids:2 * self.n_sigmoids]

    def raw_weights(self, dsparams):
        return dsparams[..., 2 * self.n_sigmoids:3 * self.n_sigmoids]

    def _forward_logabsdet(self, a, dsparams, ndim, pre_sigm, x_pre_clipped):
        logj = torch.nn.functional.log_softmax(self.raw_weights(dsparams), dim=-1) + \
               self.logsigmoid(pre_sigm) + \
               self.logsigmoid(-pre_sigm) + self.log(a)

        logj = torch.logsumexp(logj, -1)
        logabsdet_ = logj + np.log(1 - self.eps) - (self.log(x_pre_clipped) + self.log(-x_pre_clipped + 1))
        return logabsdet_.sum(-1)

    def mollify(self, a_, b_):
        a = a_ * (1 - self._mollify) + 1.0 * self._mollify
        b = b_ * (1 - self._mollify) + 0.0 * self._mollify
        return a, b

    def inverse(self, inputs, context=None):
        raise NotImplementedError("..")


class DeepSigmoid(DeepSigmoidModule):
    def __init__(self, features, *args, **kwargs):
        self.features = features
        super().__init__(*args, **kwargs)
        _a_preact = -2 * torch.ones(self.features, self.n_sigmoids)  # scale
        _b_preact = torch.zeros(self.features, self.n_sigmoids)  # shift
        _w_preact = torch.ones(self.features, self.n_sigmoids)  # softmax

        self.dsparams = torch.nn.Parameter(torch.concatenate([_a_preact + 1e-5 * torch.randn_like(_a_preact),
                                                              _b_preact + 1e-5 * torch.randn_like(_b_preact),
                                                              _w_preact + 1e-3 * torch.randn_like(_w_preact)], -1),
                                           requires_grad=True)

    def forward(self, inputs, context=None) -> torch.Tensor:
        return self.forward_given_params(inputs=inputs, dsparams=self.dsparams)
