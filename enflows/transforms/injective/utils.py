import torch
from torch import nn
import sympy as sym
import numpy as np


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_radius=1.):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.max_radius = max_radius

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * self.max_radius
        return x


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
    check_tensor(arr)
    eps = 1e-5
    r = arr[:, -1:]
    angles = arr[:, :-1]

    # print(angles[:,:-1][angles[:, :-1] >= np.pi -eps])
    # print(angles[:,:-1][angles[:, :-1] <= eps])
    # print(angles[:, -1][angles[:, -1] >= 2 * np.pi - eps])
    # print(angles[:, -1][angles[:, -1] <= eps])

    # _0_pi = torch.clamp(angles[:,:-1], min=eps, max=np.pi-eps)
    # _0_2pi = torch.clamp(angles[:,-1:], min=eps, max=2*np.pi-eps)
    # angles_clamped = torch.cat((_0_pi, _0_2pi), dim=1)
    # assert torch.all(angles_clamped[:, :-1] > 0)
    # assert torch.all(angles_clamped[:, :-1] < np.pi)
    # assert torch.all(angles_clamped[:, -1] > 0)
    # assert torch.all(angles_clamped[:, -1] < 2 * np.pi)
    sin_prods = torch.cumprod(torch.sin(angles), dim=1)
    x1 = r * torch.cos(angles[:, :1])
    xs = r * sin_prods[:, :-1] * torch.cos(angles[:, 1:])
    xn = r * sin_prods[:, -1:]

    return torch.cat((x1, xs, xn), dim=1)


def cartesian_to_spherical_torch(arr):
    assert arr.shape[-1] >= 2
    check_tensor(arr)

    eps = 1e-5
    radius = torch.linalg.norm(arr, dim=-1)
    flipped_cumsum = torch.cumsum(torch.flip(arr ** 2, dims=(-1,)), dim=-1)
    sqrt_sums = torch.flip(torch.sqrt(flipped_cumsum + eps), dims=(-1,))[...,:-1]
    angles = torch.acos(arr[..., :-1] / (sqrt_sums + eps))
    last_angle = ((arr[...,-1] >= 0).float() * angles[..., -1] + (arr[...,-1] < 0).float() * (2 * np.pi - angles[..., -1]))

    return torch.cat((angles[..., :-1], last_angle.unsqueeze(-1), radius.unsqueeze(-1)), dim=-1)


def logabsdet_sph_to_car(arr):
    # meant for batches of vectors, i.e. arr.shape = (mb, n)
    eps = 1e-8
    n = arr.shape[-1]
    r = arr[..., -1]
    angles = arr[..., :-2]
    sin_angles = torch.sin(angles)
    sin_exp = torch.arange(n - 2, 0, -1).to(arr.device)

    logabsdet_r = (n - 1) * torch.log(r + eps)

    logabsdet_sin = torch.sum(sin_exp * torch.log(torch.abs(sin_angles) + eps), dim=-1)

    return logabsdet_r + logabsdet_sin


def sherman_morrison_inverse(A):
    # print(A)
    A_triu = A.triu()
    check_tensor(A_triu)
    eye = torch.eye(*A.shape[1:]).repeat(A.shape[0], 1, 1).to(A.device)
    # A_triu_inv = torch.triangular_solve(eye, A_triu)[0]
    eps = 1e-3
    A_triu_eps = A_triu + eye * eps
    A_triu_inv = torch.linalg.solve_triangular(A_triu_eps, eye, upper=True)
    # A_triu_inv = torch.linalg.inv(A_triu)
    check_tensor(A_triu_inv)

    u = torch.zeros_like(A[:, :, -1:])
    u[:, -1, :] = 1.
    v = A[:, -1:, :-1]
    v = torch.cat((v, torch.zeros_like(v[:, :, :1])), 2)

    assert torch.all(A == A_triu + u @ v)

    num = ((A_triu_inv @ u) @ v) @ A_triu_inv
    den = 1 + (v @ A_triu_inv) @ u

    assert not torch.any(den==0)

    return A_triu_inv - num / den


def jacobian_det_spherical_cartesian(x):
    d = x.shape[1]
    assert d >= 2

    sign = (-1.) ** ((d - 1) % 2)
    r_n_1 = torch.pow(x[:, -1], torch.ones_like(x[:, -1]) * (d - 1))
    sines = torch.sin(x[:, :-2])
    sine_powers = torch.arange(d - 2, 0, -1).repeat(x.shape[0], 1)
    sines_k = torch.pow(sines, sine_powers)

    return sign * r_n_1 * sines_k.prod(1)


def check_tensor(tensor):
    assert not torch.any(torch.isnan(tensor))
    assert not torch.any(torch.isinf(tensor))