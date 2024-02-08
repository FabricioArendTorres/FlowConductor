import torch

from torch.nn import functional as F
from torch import nn
from torch.nn import init
from torch.func import jacfwd, jacrev
from functools import partial

import numpy as np
import sympytorch
from gpytorch.utils import linear_cg

from enflows.transforms import Transform, ConditionalTransform, Sigmoid, ScalarScale, CompositeTransform, ScalarShift
from enflows.transforms.injective.utils import sph_to_cart_jacobian_sympy, spherical_to_cartesian_torch, cartesian_to_spherical_torch, logabsdet_sph_to_car
from enflows.transforms.injective.utils import check_tensor, sherman_morrison_inverse, SimpleNN, jacobian_sph_to_car

import time
from datetime import timedelta
from torch.utils.benchmark import Timer

class ManifoldFlow(Transform):
    def __init__(self, logabs_jacobian):
        super().__init__()
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        assert logabs_jacobian in ["cholesky", "analytical"]
        self.logabs_jacobian = logabs_jacobian

    def r_given_theta(self, theta, context=None):
        raise NotImplementedError()

    def gradient_r_given_theta(self, theta, context=None):
        raise NotImplementedError()

    def inverse(self, theta, context=None):
        r = self.r_given_theta(theta, context=context)
        theta_r = torch.cat([theta, r], dim=1)

        outputs = spherical_to_cartesian_torch(theta_r)

        # torch.cuda.synchronize()
        # start_time = time.monotonic()
        if self.logabs_jacobian == "analytical":
            logabsdet = self.logabs_jacobian_analytical(theta, theta_r)
        elif self.logabs_jacobian == "cholesky":
            logabsdet = self.logabs_jacobian_cholesky(theta, theta_r)
        else:
            raise ValueError(f"logdet_jacobian {self.logdet_jacobian} is not a valid choice")
        # torch.cuda.synchronize()
        # end_time = time.monotonic()
        # total_time = end_time - start_time
        # print("Total time: ", total_time)

        # print("inverse")

        return outputs, logabsdet

    def forward(self, inputs, context=None):

        # print("forward")

        outputs = cartesian_to_spherical_torch(inputs)
        if self.logabs_jacobian == "analytical":
            logabsdet = self.logabs_jacobian_analytical(outputs[:,:-1], outputs)
        elif self.logabs_jacobian == "cholesky":
            logabsdet = self.logabs_jacobian_cholesky(outputs[:,:-1], outputs)
        else:
            raise ValueError(f"logdet_jacobian {self.logdet_jacobian} is not a valid choice")

        return outputs[..., :-1], -logabsdet

    def compute_jacobian_row(self, output, input):
        assert output.shape[0] == input.shape[0]
        output = output.view(output.shape[0], -1)

        # Compute Jacobian row by row.
        jac = []
        for j in range(output.shape[1]):
            dy_j_dx = torch.autograd.grad(output[:, j], input, torch.ones_like(output[:, j]), retain_graph=True,
                                          create_graph=True)[0].view(input.shape[0], -1)
            jac.append(torch.unsqueeze(dy_j_dx, 1))
        jac = torch.cat(jac, 1)
        return jac

    def logabs_jacobian_analytical(self, theta, theta_r, context=None):
        eps = 1e-8
        # spherical_dict = {name: theta_r[:, i] for i, name in enumerate(self.spherical_names)}
        # jac = self.sph_to_cart_jac(**spherical_dict).reshape(-1, theta_r.shape[-1], theta_r.shape[-1])#.to(theta.device)
        # jac = torch.autograd.functional.jacobian(spherical_to_cartesian_torch, theta_r).sum(-2)
        # jac = jacrev(spherical_to_cartesian_torch)(theta_r).sum(-2)
        # jac = jacfwd(spherical_to_cartesian_torch)(theta_r).sum(-2)
        # torch.cuda.synchronize()
        start_time = time.monotonic()
        # cartesian = spherical_to_cartesian_torch(theta_r)
        # jac = self.compute_jacobian_row(cartesian, theta_r)
        # jac = jacrev(spherical_to_cartesian_torch)(theta_r).sum(-2)
        cartesian = spherical_to_cartesian_torch(theta_r)
        jac = jacobian_sph_to_car(theta_r, cartesian)
        # print(jac.element_size() * jac.nelement())
        # breakpoint()
        # jac = vmap(functorch.jacrev(spherical_to_cartesian_torch)(theta_r).sum(-2))
        # torch.cuda.synchronize()
        end_time = time.monotonic()
        total_time = end_time - start_time
        # print("Jacobian: ", total_time)

        # torch.cuda.synchronize()
        start_time = time.monotonic()
        jac_inv = sherman_morrison_inverse(jac.mT)
        # torch.cuda.synchronize()
        # end_time = time.monotonic()
        # total_time = end_time - start_time
        # print("Sherman morrison inv: ", total_time)
        # jac_inv = torch.inverse(jac.mT + eps)
        # check_tensor(jac_inv)

        # torch.cuda.synchronize()
        # start_time = time.monotonic()

        grad_r = self.gradient_r_given_theta(theta, context=context)
        # check_tensor(grad_r)

        # check_tensor(jac_inv)
        jac_inv_grad = jac_inv @ grad_r
        # check_tensor(jac_inv_grad)

        # fro_norm = torch.norm(jac_inv_grad.squeeze(-1), p='fro', dim=1)
        fro_norm = torch.norm(jac_inv_grad.squeeze(-1), p='fro', dim=1)
        # check_tensor(fro_norm)

        logabsdet_fro_norm = torch.log(fro_norm + eps)
        logabsdet_s_to_c = logabsdet_sph_to_car(theta_r)
        # check_tensor(logabsdet_fro_norm)
        # check_tensor(logabsdet_s_to_c)

        logabsdet = logabsdet_s_to_c + logabsdet_fro_norm

        # breakpoint()

        # torch.cuda.synchronize()
        end_time = time.monotonic()
        total_time = end_time - start_time
        # f = open(f"results_analytical_manifold.csv", "a")
        # f.write(f"{total_time}\n")
        # f.close()
        # print("All the rest: ", total_time)

        return logabsdet

    def logabs_jacobian_cholesky(self, theta, theta_r, context=None):
        eps = 1e-8
        # spherical_dict = {name: theta_r[:, i].cpu() for i, name in enumerate(self.spherical_names)}
        # jac_sph_cart = self.sph_to_cart_jac(**spherical_dict).reshape(-1, theta_r.shape[-1], theta_r.shape[-1]).to(theta.device)
        # jac_sph_cart = torch.autograd.functional.jacobian(spherical_to_cartesian_torch, theta_r).sum(-2)
        # jac_sph_cart = jacrev(spherical_to_cartesian_torch)(theta_r).sum(-2)
        start_time = time.monotonic()
        cartesian = spherical_to_cartesian_torch(theta_r)
        jac_sph_cart = jacobian_sph_to_car(theta_r, cartesian)
        end_time = time.monotonic()
        total_time = end_time - start_time
        print("Jacobian: ", total_time)
        # jac_sph_cart_ = jacfwd(spherical_to_cartesian_torch)(theta_r).sum(-2)
        # breakpoint()
        # check_tensor(jac_sph_cart)

        start_time = time.monotonic()
        eye = torch.eye(theta.shape[-1]).repeat(theta.shape[0],1,1).to(theta.device)
        r_given_theta = partial(self.r_given_theta, context=context)
        # grad_r = torch.autograd.functional.jacobian(r_given_theta, theta).sum(-2)
        grad_r = jacfwd(r_given_theta)(theta).sum(-2)
        jac_r_theta = torch.cat((eye, grad_r), dim=1)
        # check_tensor(jac_r_theta)
        jac_forward = jac_sph_cart @ jac_r_theta

        # we need to compute 0.5 * sqrt(ac_forward.T @ jac_forward)
        # naively we could compute it as
        # logabsdet2 = 0.5 * torch.logdet(jac_full)
        # instead, we use cholesky decomposition to compute the determinant in O(d^3)
        jac_full = jac_forward.mT @ jac_forward
        jac_full_eye = torch.diag_embed(jac_full.new_ones(jac_full.shape[-1]))
        jac_full = jac_full + jac_full_eye * eps
        jac_full_lower = torch.linalg.cholesky(jac_full)
        jac_full_lower_diag = torch.diagonal(jac_full_lower, dim1=1, dim2=2)
        logabsdet = torch.log(jac_full_lower_diag).sum(1) # should be 2* but there is also a 0.5 factor

        # logabsdet_ = self.logabs_pseudodet(theta, theta_r, context=context)
        # max_diff = torch.square(logabsdet-logabsdet_).max().item()
        # print("max diff: ", max_diff)

        end_time = time.monotonic()
        total_time = end_time - start_time
        print("All the rest: ", total_time)

        return logabsdet

    def _initialize_jacobian(self, inputs):
        spherical_names, jac = sph_to_cart_jacobian_sympy(inputs.shape[1] + 1)
        self.spherical_names = spherical_names
        self.sph_to_cart_jac = sympytorch.SymPyModule(expressions=jac).to(inputs.device)
        self.initialized.data = torch.tensor(True, dtype=torch.bool)


class LearnableManifoldFlow(ManifoldFlow):
    def __init__(self, n, logabs_jacobian, max_radius=2.):
        super().__init__(logabs_jacobian=logabs_jacobian)

        self.network = SimpleNN(n, hidden_size=500, output_size=1, max_radius=max_radius)

    def r_given_theta(self, theta, context=None):
        r = self.network(theta)

        return r

    def gradient_r_given_theta(self, theta, context=None):
        r = self.r_given_theta(theta, context=context)
        grad_r_theta = torch.autograd.grad(r,theta, grad_outputs=torch.ones_like(r))[0]
        grad_r_theta_aug = torch.cat([- grad_r_theta, torch.ones_like(grad_r_theta[:, :1])], dim=1)

        # check_tensor(grad_r_theta)

        return grad_r_theta_aug.unsqueeze(-1)


class SphereFlow(ManifoldFlow):
    def __init__(self, n, logabs_jacobian, r=1.):
        super().__init__(logabs_jacobian=logabs_jacobian)
        self.radius = r
        # self.network = SimpleNN(n, hidden_size=50, output_size=1, max_radius=max_radius)

    def r_given_theta(self, theta, context=None):
        r = theta.new_ones(theta.shape[0], 1)
        # r = self.network(theta)

        return r

    def gradient_r_given_theta(self, theta, context=None):
        # r = self.r_given_theta(theta, context=context)
        # grad_r_theta = torch.autograd.grad(r,theta, grad_outputs=torch.ones_like(r))[0]
        grad_r_theta = torch.zeros_like(theta)
        grad_r_theta_aug = torch.cat([- grad_r_theta, torch.ones_like(grad_r_theta[:, :1])], dim=1)

        check_tensor(grad_r_theta)

        return grad_r_theta_aug.unsqueeze(-1)

class LpManifoldFlow(ManifoldFlow):
    def __init__(self, norm, p, logabs_jacobian):
        super().__init__(logabs_jacobian=logabs_jacobian)
        self.norm = norm
        self.p = p
        self.r_given_norm = r_given_norm
        # self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))


    def r_given_theta(self, theta, context=None):
        assert theta.shape[1] >= 1
        eps = 1e-8
        n = theta.shape[-1]

        sin_thetas = torch.sin(theta)
        cos_thetas = torch.cos(theta)

        norm_1 = (torch.abs(cos_thetas[:, 0]) + eps) ** self.p
        check_tensor(norm_1)
        norm_3 = (torch.abs(torch.prod(sin_thetas, dim=-1)) + eps) ** self.p
        check_tensor(norm_3)

        if theta.shape[1] == 1:
            r = self.norm / ((norm_1 + norm_3 + eps) ** (1. / self.p))
            check_tensor(r)

            return r.unsqueeze(-1)
        else:
            norm_2_ = [(torch.abs(torch.prod(sin_thetas[..., :k - 1], dim=-1) * cos_thetas[..., k - 1]) + eps) ** self.p
                       for k in range(2, n + 1)]
            norm_2 = torch.stack(norm_2_, dim=1).sum(-1)
            check_tensor(norm_2)

            r = self.norm / ((norm_1 + norm_2 + norm_3 + eps) ** (1. / self.p))
            check_tensor(r)
            # breakpoint()

            return r.unsqueeze(-1)

    def gradient_r_given_theta(self, theta, context=None):
        r = self.r_given_theta(theta).squeeze()
        grad_r_theta = torch.autograd.grad(r, theta, grad_outputs=torch.ones_like(r))[0]
        grad_r_theta_aug = torch.cat([- grad_r_theta, torch.ones_like(grad_r_theta[:, :1])], dim=1)

        check_tensor(grad_r_theta)
        # print("gradient_shape", grad_r_theta.shape, grad_r_theta_aug.unsqueeze(-1).shape)
        return grad_r_theta_aug.unsqueeze(-1)

class PeriodicElementwiseTransform(Transform):
    def __init__(self, elemwise_transform=torch.sin, elemwise_inv_transform=torch.asin, scale=0.5*np.pi):
        super().__init__()
        self.elemwise_transform = elemwise_transform
        self.scale = scale
        self.elemwise_inv_transform = elemwise_inv_transform
        self.eps = 1e-8

    def forward(self, inputs, context=None):
        outputs = (self.elemwise_transform(inputs) + 1) * self.scale
        logabsdet_cos = torch.log(torch.cos(inputs) + self.eps).sum(-1)
        logabsdet_scale = inputs.shape[-1] * np.log(self.scale)

        return outputs, logabsdet_cos + logabsdet_scale

    def inverse(self, inputs, context=None):
        outputs = self.elemwise_inv_transform(inputs / self.scale - 1)
        logabsdet_cos = torch.log(torch.cos(outputs) + self.eps).sum(-1)
        logabsdet_scale = inputs.shape[-1] * np.log(self.scale)

        return outputs, -logabsdet_cos - logabsdet_scale


class ScaleLastDim(Transform):
    def __init__(self, scale=2.):
        super().__init__()
        self.scale = scale

    def forward(self, inputs, context=None):
        mask = torch.zeros_like(inputs)
        mask[...,-1] = torch.ones_like(inputs[..., -1])
        outputs = (1 - mask) * inputs + mask * inputs * self.scale
        logabsdet = inputs.new_ones(inputs.shape[0]) * np.log(self.scale)

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        mask = torch.zeros_like(inputs)
        mask[..., -1] = torch.ones_like(inputs[..., -1])
        outputs = (1 - mask) * inputs + mask * inputs / self.scale
        logabsdet = -inputs.new_ones(inputs.shape[0]) * np.log(self.scale)

        return outputs, logabsdet


class ConstrainedAngles(Transform):
    def __init__(self, elemwise_transform: Transform = Sigmoid()):
        super().__init__()
        self.elemwise_transform = elemwise_transform

    def forward(self, inputs, context=None):
        mask = torch.zeros_like(inputs)
        mask[..., -1] = torch.ones_like(inputs[..., -1])
        transformed_inputs = inputs - 0.5 * mask * inputs
        outputs, logabsdet_elemwise = self.elemwise_transform.inverse(transformed_inputs)
        logabsdet_last_elem = inputs.new_ones(inputs.shape[0]) * torch.log(torch.tensor(0.5))
        # breakpoint()
        return outputs, logabsdet_elemwise + logabsdet_last_elem

    def inverse(self, inputs, context=None):
        mask = torch.zeros_like(inputs)
        mask[...,-1] = torch.ones_like(inputs[..., -1])
        transformed_inputs, logabsdet_elemwise = self.elemwise_transform(inputs)
        outputs = mask * transformed_inputs + transformed_inputs
        logabsdet_last_elem = inputs.new_ones(inputs.shape[0]) * torch.log(torch.tensor(2.))
        # breakpoint()
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
        return inputs, torch.zeros_like(inputs[...,0])

class ClampedTheta(Transform):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, context):
        self.dtype = inputs.dtype
        thetas = inputs[...,:-1]
        last_theta = inputs[...,-1:]
        _0_pi_mask, _0_pi_clamp = self.compute_mask(arr=thetas, vmin=0., vmax=np.pi, right_included=True)
        clamped_thetas = _0_pi_mask * _0_pi_clamp

        # _0_2pi_mask, _0_2pi_clamp = self.compute_mask(arr=last_theta, vmin=0., vmax=2*np.pi, right_included=True)
        # clamped_last_theta = _0_2pi_mask * _0_2pi_clamp

        output = torch.cat((clamped_thetas, last_theta), dim = -1)
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
        return inputs, torch.zeros_like(inputs[...,0])

###################################################CONDITIONAL LAYERS###################################################


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




########################################################################################################################
#######################################################DEPRECATED#######################################################
########################################################################################################################



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
        eps = 1e-8
        spherical_dict = {name: theta_r[:, i].cpu() for i, name in enumerate(self.spherical_names)}
        jac = self.sph_to_cart_jac(**spherical_dict).reshape(-1, theta_r.shape[-1], theta_r.shape[-1]).to(inputs.device)
        # jac = jacobian(spherical_to_cartesian_torch, theta_r).sum(-2)
        check_tensor(jac)

        # assert torch.allclose(jac, jac_)

        jac_inv = sherman_morrison_inverse(jac.mT)
        #jac_inv = torch.inverse(jac.mT)
        check_tensor(jac_inv)


        grad_r = gradient_r(inputs, self.norm, self.q)
        check_tensor(grad_r)
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
        check_tensor(jac_inv_grad)
        # print(f"jac inv max: {jac_inv.squeeze().max().item():.3e}, "
        #       f"min: {jac_inv.squeeze().min().item():.3e} ")
        # print(f"grad inv max: {grad_r.squeeze().max().item():.3e}, "
        #       f"min: {grad_r.squeeze().min().item():.3e} ")
        fro_norm = torch.norm(jac_inv_grad.squeeze(), p='fro', dim=1)

        check_tensor(fro_norm)
        logabsdet_fro_norm = torch.log(torch.abs(fro_norm) + eps)
        logabsdet_s_to_c = logabsdet_sph_to_car(theta_r)

        check_tensor(logabsdet_fro_norm)
        check_tensor(logabsdet_s_to_c)

        logabsdet = logabsdet_s_to_c + logabsdet_fro_norm


        return logabsdet

    def _initialize_jacobian(self, inputs):
        spherical_names, jac = sph_to_cart_jacobian_sympy(inputs.shape[1]+1)
        self.spherical_names = spherical_names
        self.sph_to_cart_jac = sympytorch.SymPyModule(expressions=jac).to(inputs.device)


def r_given_norm(thetas, norm, q):
    assert thetas.shape[1] >= 1
    eps = 1e-8
    n = thetas.shape[-1]

    check_tensor(thetas)

    sin_thetas = torch.sin(thetas)
    cos_thetas = torch.cos(thetas)

    norm_1 = (torch.abs(cos_thetas[:, 0]) + eps) ** q
    check_tensor(norm_1)
    norm_3 = (torch.abs(torch.prod(sin_thetas, dim=-1)) + eps) ** q
    check_tensor(norm_3)
    if thetas.shape[1] == 1:
        r = norm / ((norm_1 + norm_3) ** (1. / q))
        check_tensor(r)
        return norm / ((norm_1 + norm_3 + eps) ** (1. / q))
    else:
        norm_2_ = [(torch.abs(torch.prod(sin_thetas[..., :k - 1], dim=-1) * cos_thetas[..., k - 1]) + eps) ** q for k in
                   range(2, n + 1)]
        norm_2 = torch.stack(norm_2_, dim=1).sum(-1)
        check_tensor(norm_2)

        r = norm / ((norm_1 + norm_2 + norm_3 + eps) ** (1. / q))
        check_tensor(r)
        return r


def inflate_radius(inputs, norm, q):
    r = r_given_norm(inputs, norm, q)
    # if context is not None:
    #     theta_r = torch.cat([inputs, r], dim=1)
    # else:
    #     theta_r = torch.cat([inputs, r.unsqueeze(-1)], dim=1)
    theta_r = torch.cat([inputs, r.unsqueeze(-1)], dim=1)
    return theta_r

def gradient_r(inputs, norm, q):
    r = r_given_norm(inputs, norm, q)
    grad_r_theta = torch.autograd.grad(r, inputs, grad_outputs=torch.ones_like(r))[0]
    grad_r_theta_aug = torch.cat([- grad_r_theta, torch.ones_like(grad_r_theta[:, :1])], dim=1)

    check_tensor(grad_r_theta)

    return grad_r_theta_aug.unsqueeze(-1)

def sherman_morrison_inverse_old(M):
    N = M.shape[-1]
    lower_indices = np.tril_indices(n=N, k=-1)
    mask = torch.ones(*M.shape)
    mask[:, lower_indices[0], lower_indices[1]] = 0.
    U = M * mask
    assert torch.all(U.triu() == U)

    v = torch.zeros(M.shape[:2])
    v[:, :-1] = M[:, -1, :-1]
    u = torch.zeros_like(v)
    u[:, -1] = 1.

    uv = torch.einsum("bi,bj->bij", u, v)
    assert torch.all(M == U + torch.einsum("bi,bj->bij", u, v))

    eye = torch.eye(N, N).repeat(M.shape[0], 1, 1)
    U_inv = torch.linalg.solve_triangular(U, eye, upper=True)
    num = U_inv @ uv @ U_inv

    den = 1 + v.unsqueeze(1) @ U_inv @ u.unsqueeze(2)

    return U_inv - num / den