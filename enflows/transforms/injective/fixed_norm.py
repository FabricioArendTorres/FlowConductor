import torch
from torch.autograd.functional import jacobian
from torch.nn import functional as F
from torch import nn
from torch.nn import init

import sympy as sym
import numpy as np
import sympytorch
from enflows.transforms import Transform, ConditionalTransform, Sigmoid, ScalarScale, CompositeTransform, ScalarShift
import matplotlib.pyplot as plt



def r_given_norm(thetas, norm, q):
    assert thetas.shape[1] >= 1
    eps = 1e-8
    n = thetas.shape[-1]
    assert not torch.any(torch.isnan(thetas))
    sin_thetas = torch.sin(thetas)
    cos_thetas = torch.cos(thetas)

    norm_1 = (torch.abs(cos_thetas[:, 0]) + eps) ** q
    assert not torch.any(torch.isnan(norm_1))
    norm_3 = (torch.abs(torch.prod(sin_thetas, dim=-1)) + eps) ** q
    assert not torch.any(torch.isnan(norm_3))
    if thetas.shape[1] == 1:
        r = norm / ((norm_1 + norm_3) ** (1. / q))
        assert  not torch.any(torch.isnan(r))
        return norm / ((norm_1 + norm_3 + eps) ** (1. / q))
    else:
        norm_2_ = [(torch.abs(torch.prod(sin_thetas[..., :k - 1], dim=-1) * cos_thetas[..., k - 1]) + eps) ** q for k in
                   range(2, n + 1)]
        norm_2 = torch.stack(norm_2_, dim=1).sum(-1)
        assert not torch.any(torch.isnan(norm_2))
        r = norm / ((norm_1 + norm_2 + norm_3 + eps) ** (1. / q))
        assert not torch.any(torch.isnan(r))
        return r


def inflate_radius(inputs, norm, q):
    r = r_given_norm(inputs, norm, q)
    # if context is not None:
    #     theta_r = torch.cat([inputs, r], dim=1)
    # else:
    #     theta_r = torch.cat([inputs, r.unsqueeze(-1)], dim=1)
    theta_r = torch.cat([inputs, r.unsqueeze(-1)], dim=1)
    return theta_r
def sph_to_cart_sympy(spherical):
    # sympy implementation of change of variables from spherical to cartesian
    # [theta_1, theta_2, ..., theta_n-1, r] --> [x1, x2, ..., x_n]

    n = len(spherical)
    thetas = spherical[:-1]
    r = spherical[-1]

    sin_thetas = [sym.sin(theta) for theta in thetas]
    cos_thetas = [sym.cos(theta) for theta in thetas]

    x = r * cos_thetas[0]
    x_ = [r * sym.prod(sin_thetas[:k - 1], cos_thetas[k - 1]) for k in range(2, n)]
    x_n = r * sym.prod(sin_thetas)

    return [x, *x_, x_n]

def sph_to_cart_jacobian_sympy (n):
    # analytic expression of the jacobian of the change of variables from spherical to cartesian

    spherical_names = [f'theta_{k}' for k in range(1, n)] + ['r']
    spherical = [sym.Symbol(name, real=True) for name in spherical_names]

    output = sym.Matrix(sph_to_cart_sympy(spherical))
    jacobian = output.jacobian(spherical)

    return spherical_names, jacobian


def spherical_to_cartesian_torch(arr):
    # meant for batches of vectors, i.e. arr.shape = (mb, n)
    assert arr.shape[1] >= 2
    assert not torch.any(torch.isnan(arr))
    eps = 1e-5
    r = arr[:, -1:]
    angles = arr[:, :-1]

    # print(angles[:,:-1][angles[:, :-1] >= np.pi -eps])
    # print(angles[:,:-1][angles[:, :-1] <= eps])
    # print(angles[:, -1][angles[:, -1] >= 2 * np.pi - eps])
    # print(angles[:, -1][angles[:, -1] <= eps])

    _0_pi = torch.clamp(angles[:,:-1], min=eps, max=np.pi-eps)
    _0_2pi = torch.clamp(angles[:,-1:], min=eps, max=2*np.pi-eps)
    angles_clamped = torch.cat((_0_pi, _0_2pi), dim=1)
    assert torch.all(angles_clamped[:, :-1] > 0)
    assert torch.all(angles_clamped[:, :-1] < np.pi)
    assert torch.all(angles_clamped[:, -1] > 0)
    assert torch.all(angles_clamped[:, -1] < 2 * np.pi)
    sin_prods = torch.cumprod(torch.sin(angles), dim=1)
    x1 = r * torch.cos(angles[:, :1])
    xs = r * sin_prods[:, :-1] * torch.cos(angles[:, 1:])
    xn = r * sin_prods[:, -1:]

    return torch.cat((x1, xs, xn), dim=1)

def logabsdet_sph_to_car(arr):
    # meant for batches of vectors, i.e. arr.shape = (mb, n)
    eps = 1e-10
    n = arr.shape[1]
    r = arr[:, -1]
    angles = arr[:, :-2]
    sin_angles = torch.sin(angles)
    sin_exp = torch.arange(n - 2, 0, -1).to(arr.device)

    logabsdet_r = (n - 1) * torch.log(r + eps)
    logabsdet_sin = torch.sum(sin_exp * torch.log(torch.abs(sin_angles) + eps), dim=1)

    return logabsdet_r + logabsdet_sin


# def sherman_morrison_inverse(M):
#     N = M.shape[-1]
#     lower_indices = np.tril_indices(n=N, k=-1)
#     mask = torch.ones(*M.shape)
#     mask[:, lower_indices[0], lower_indices[1]] = 0.
#     U = M * mask
#     assert torch.all(U.triu() == U)
#
#     v = torch.zeros(M.shape[:2])
#     v[:, :-1] = M[:, -1, :-1]
#     u = torch.zeros_like(v)
#     u[:, -1] = 1.
#
#     uv = torch.einsum("bi,bj->bij", u, v)
#     assert torch.all(M == U + torch.einsum("bi,bj->bij", u, v))
#
#     eye = torch.eye(N, N).repeat(M.shape[0], 1, 1)
#     U_inv = torch.linalg.solve_triangular(U, eye, upper=True)
#     num = U_inv @ uv @ U_inv
#
#     den = 1 + v.unsqueeze(1) @ U_inv @ u.unsqueeze(2)
#
#     return U_inv - num / den


def sherman_morrison_inverse(A):
    # print(A)
    A_triu = A.triu()
    assert not torch.any(torch.isnan(A_triu))
    eye = torch.eye(*A.shape[1:]).repeat(A.shape[0], 1, 1).to(A.device)
    # A_triu_inv = torch.triangular_solve(eye, A_triu)[0]
    eps = 1e-7
    A_triu_eps = A_triu + eye * eps
    A_triu_inv = torch.linalg.solve_triangular(A_triu_eps, eye, upper=True)
    # A_triu_inv = torch.linalg.inv(A_triu)

    assert not torch.any(torch.isnan(A_triu_inv))

    u = torch.zeros_like(A[:, :, -1:])
    u[:, -1, :] = 1.
    v = A[:, -1:, :-1]
    v = torch.cat((v, torch.zeros_like(v[:, :, :1])), 2)

    assert torch.all(A == A_triu + u @ v)

    num = ((A_triu_inv @ u) @ v) @ A_triu_inv
    den = 1 + (v @ A_triu_inv) @ u

    assert not torch.any(den==0)

    return A_triu_inv - num / den

def gradient_r(inputs, norm, q):
    r = r_given_norm(inputs, norm, q)
    grad_r_theta = torch.autograd.grad(r, inputs, grad_outputs=torch.ones_like(r))[0]
    grad_r_theta_aug = torch.cat([- grad_r_theta, torch.ones_like(grad_r_theta[:, :1])], dim=1)

    assert not torch.any(torch.isnan(grad_r_theta))

    return grad_r_theta_aug.unsqueeze(-1)


def jacobian_det_spherical_cartesian(x):
    d = x.shape[1]
    assert d >= 2

    sign = (-1.) ** ((d - 1) % 2)
    r_n_1 = torch.pow(x[:, -1], torch.ones_like(x[:, -1]) * (d - 1))
    sines = torch.sin(x[:, :-2])
    sine_powers = torch.arange(d - 2, 0, -1).repeat(x.shape[0], 1)
    sines_k = torch.pow(sines, sine_powers)

    return sign * r_n_1 * sines_k.prod(1)


class FixedNorm(Transform):
    def __init__(self, norm, q):
        super().__init__()
        self.norm = norm
        self.q = q

        self.r_given_norm = r_given_norm
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def forward(self, inputs, context=None):

        if self.training and not self.initialized:
            self._initialize_jacobian(inputs)

        theta_r = inflate_radius(inputs, self.norm, self.q)
        outputs = spherical_to_cartesian_torch(theta_r)

        logabsdet = self.logabs_pseudodet(inputs, theta_r)

        return outputs, logabsdet


    def inverse(self, inputs, context=None):
        raise NotImplementedError


    def logabs_pseudodet(self, inputs, theta_r, context=None):
        #spherical_dict = {name: theta_r[:, i] for i, name in enumerate(self.spherical_names)}
        #jac = self.sph_to_cart_jac(**spherical_dict).reshape(-1, self.N, self.N)
        jac = jacobian(spherical_to_cartesian_torch, theta_r).sum(-2)
        assert not torch.any(torch.isnan(jac))
        # assert torch.allclose(jac, jac_)

        jac_inv = sherman_morrison_inverse(jac.mT)
        #jac_inv = torch.inverse(jac.mT)

        assert not torch.any(torch.isnan(jac_inv))

        grad_r = gradient_r(inputs, self.norm, self.q)
        #grad_r = torch.clamp(grad_r, min=-100, max=100)
        # grad_r_np = grad_r.detach().cpu().numpy().reshape(-1,inputs.shape[-1]+1)[:,:inputs.shape[-1]]
        # inputs_np = inputs.detach().cpu().numpy().reshape(-1,inputs.shape[-1])
        # print("grad_r_np", grad_r_np)
        # print('contains nans', np.any(np.isnan(grad_r_np)))
        # grad_r_np[grad_r_np == 0] = 1e-7
        # log_min = np.log10(np.min(np.abs(grad_r_np.ravel())))
        # log_max = np.log10(np.max(np.abs(grad_r_np.ravel())))
        #
        # print(log_min, log_max)
        # plt.hist(np.abs(grad_r_np).ravel(), bins=np.logspace(log_min, log_max, 100))
        # plt.xscale('log')
        # # plt.scatter(np.linalg.norm(inputs_np, axis=-1), grad_r_np, marker='.')
        # plt.show()

        jac_inv_grad = jac_inv @ grad_r
        fro_norm = torch.norm(jac_inv_grad.squeeze(), p='fro', dim=1)

        logabsdet_fro_norm = torch.log(torch.abs(fro_norm))
        logabsdet_s_to_c = logabsdet_sph_to_car(theta_r)

        assert not torch.any(torch.isnan(logabsdet_fro_norm))
        assert not torch.any(torch.isnan(logabsdet_s_to_c))

        logabsdet = logabsdet_s_to_c + logabsdet_fro_norm


        return logabsdet

    def _initialize_jacobian(self, inputs):
        spherical_names, jac = sph_to_cart_jacobian_sympy(inputs.shape[1])
        self.spherical_names = spherical_names
        self.sph_to_cart_jac = sympytorch.SymPyModule(expressions=jac)

class ConditionalFixedNorm(Transform):

    def __init__(self,q):
        super().__init__()
        self.q = q

    def forward(self, inputs, context):
        # the first element of context is assumed to be the norm value
        transformer = FixedNorm(norm=context[:, 0], q=self.q)

        output, logabsdet = transformer.forward(inputs, context)

        return output, logabsdet

    def inverse(self, inputs, context):
        raise NotImplementedError
# class ConditionalFixedNorm(ConditionalTransform):
#
#     def __init__(
#             self,
#             features,
#             hidden_features,
#             q,
#             context_features=None,
#             num_blocks=2,
#             use_residual_blocks=True,
#             activation=F.relu,
#             dropout_probability=0.0,
#             use_batch_norm=False,
#     ):
#         self.q = q
#         super().__init__(
#             features=features,
#             hidden_features=hidden_features,
#             context_features=context_features,
#             num_blocks=num_blocks,
#             use_residual_blocks=use_residual_blocks,
#             activation=activation,
#             dropout_probability=dropout_probability,
#             use_batch_norm=use_batch_norm
#         )
#
#     def _forward_given_params(self, inputs, context):
#         # the first element of context is assumed to be the norm value
#         transformer = FixedNorm(norm=context[:,:1], q=self.q)
#
#         output, logabsdet = transformer.forward(inputs)
#
#     def _inverse_given_params(self, inputs, autoregressive_params):
#         NotImplementedError


class ResidualBlock(nn.Module):
    """A general-purpose residual block. Works only with 1-dim inputs."""

    def __init__(
        self,
        features,
        context_features,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        zero_initialization=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        if context_features is not None:
            self.context_layer = nn.Linear(context_features, features)
        self.linear_layers = nn.ModuleList(
            [nn.Linear(features, features) for _ in range(2)]
        )
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs, context=None):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.linear_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.linear_layers[1](temps)
        if context is not None:
            temps = F.glu(torch.cat((temps, self.context_layer(context)), dim=1), dim=1)
        return inputs + temps

class ResidualNetInput(nn.Module):
    """A residual network that . Works only with 1-dim inputs."""

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        context_features=None,
        num_blocks=2,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.context_features = context_features
        if context_features is not None:
            self.initial_layer = nn.Linear(
                in_features + context_features, hidden_features
            )
        else:
            self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_features,
                    context_features=context_features,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features -1)

    def forward(self, inputs, context=None):
        if context is None:
            temps = self.initial_layer(inputs)
        else:
            temps = self.initial_layer(torch.cat((inputs, context), dim=1))
        for block in self.blocks:
            temps = block(temps, context=context)
        outputs = self.final_layer(temps)

        return torch.cat((inputs[:,:1], outputs), dim=1)


class ConstrainedAngles(Transform):
    def __init__(self, elemwise_transform: Transform = Sigmoid()):
        super().__init__()
        self.elemwise_transform = elemwise_transform


    def forward(self, inputs, context=None):
        mask = torch.zeros_like(inputs)
        mask[...,-1] = torch.ones_like(inputs[..., -1])
        transformed_inputs, logabsdet_elemwise = self.elemwise_transform(inputs)
        outputs = mask * transformed_inputs + transformed_inputs
        logabsdet_last_elem = inputs.new_ones(inputs.shape[0]) * torch.log(torch.tensor(2.))

        return outputs, logabsdet_elemwise + logabsdet_last_elem

    def inverse(self, inputs, context=None):
        mask = torch.zeros_like(inputs)
        mask[..., -1] = torch.ones_like(inputs[..., -1])
        transformed_inputs, logabsdet_elemwise = self.elemwise_transform.inverse(inputs)
        outputs = mask * transformed_inputs + transformed_inputs
        logabsdet_last_elem = inputs.new_ones(inputs.shape[0]) * torch.log(torch.tensor(0.5))

        return outputs, logabsdet_elemwise + logabsdet_last_elem


class ConstrainedAnglesSigmoid(ConstrainedAngles):
    def __init__(self,temperature=1, learn_temperature=False):
        super().__init__(elemwise_transform=CompositeTransform([Sigmoid(temperature=temperature,
                                                                        learn_temperature=learn_temperature),
                                                                ScalarScale(scale=np.pi, trainable=False)]))

class ClampedAngles(Transform):
    _05PI = 0.5 * np.pi
    _10PI = 1.0 * np.pi
    _15PI = 1.5 * np.pi
    _20PI = 2.0 * np.pi

    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, context):
        self.dtype = inputs.dtype
        thetas = inputs[...,:-1]
        last_theta = inputs[...,-1:]
        _0pi_05pi_mask, _0pi_05pi_clamp = self.compute_mask(arr=thetas, vmin=0., vmax=self._05PI)
        _05pi_10pi_mask, _05pi_10pi_clamp = self.compute_mask(arr=thetas, vmin=self._05PI, vmax=self._10PI, right_included=True)
        clamped_thetas = _0pi_05pi_mask * _0pi_05pi_clamp + _05pi_10pi_mask * _05pi_10pi_clamp

        _0pi_05pi_mask, _0pi_05pi_clamp = self.compute_mask(arr=last_theta, vmin=0., vmax=self._05PI)
        _05pi_10pi_mask, _05pi_10pi_clamp = self.compute_mask(arr=last_theta, vmin=self._05PI, vmax=self._10PI)
        _10pi_15pi_mask, _10pi_15pi_clamp = self.compute_mask(arr=last_theta, vmin=self._10PI, vmax=self._15PI)
        _15pi_20pi_mask, _15pi_20pi_clamp = self.compute_mask(arr=last_theta, vmin=self._15PI, vmax=self._20PI, right_included=True)
        clamped_last_theta = _0pi_05pi_mask * _0pi_05pi_clamp + _05pi_10pi_mask * _05pi_10pi_clamp + \
                         _10pi_15pi_mask * _10pi_15pi_clamp + _15pi_20pi_mask * _15pi_20pi_clamp

        output = torch.cat((clamped_thetas, clamped_last_theta), dim = -1)
        logabsdet = output.new_zeros(inputs.shape[:-1])

        return output, logabsdet

    def compute_mask(self, arr, vmin, vmax, right_included=False):
        if right_included:
            condition = (arr >= vmin) * (arr < vmax)
        else:
            condition = (arr >= vmin) * (arr <= vmax)

        mask = condition.to(self.dtype)
        arr_clamped = torch.clamp(arr, min=vmin + self.eps, max=vmax - self.eps)

        return mask, arr_clamped


    def inverse(self, inputs, context):
        raise NotImplementedError