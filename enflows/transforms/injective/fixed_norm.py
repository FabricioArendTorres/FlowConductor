import torch
from torch.autograd.functional import jacobian
import sympy as sym
import numpy as np
import sympytorch
from enflows.transforms import Transform, ConditionalTransform, Sigmoid, ScalarScale, CompositeTransform, ScalarShift
import matplotlib.pyplot as plt



def r_given_norm(thetas, norm, q):
    eps = 1e-5
    assert thetas.shape[1] >= 1
    n = thetas.shape[-1]
    sin_thetas = torch.sin(thetas)
    cos_thetas = torch.cos(thetas)
    norm_1 = torch.abs(cos_thetas[:, 0]) ** q
    norm_3 = torch.abs(torch.prod(sin_thetas, dim=-1)) ** q
    if thetas.shape[1] == 1:
        return norm / ((norm_1 + norm_3) ** (1. / q))
    else:
        norm_2_ = [torch.abs(torch.prod(sin_thetas[..., :k - 1], dim=-1) * cos_thetas[..., k - 1]) ** q for k in
                   range(2, n + 1)]
        norm_2 = torch.stack(norm_2_, dim=-1).sum(-1)
        return norm / ((norm_1 + norm_2 + norm_3) ** (1. / q))
    #return thetas.mean(1)

def inflate_radius(inputs, norm, q):
    r = r_given_norm(inputs, norm, q)
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
    r = arr[:, -1:]
    angles = arr[:, :-1]
    sin_prods = torch.cumprod(torch.sin(angles), dim=1)
    x1 = r * torch.cos(angles[:, :1])
    xs = r * sin_prods[:, :-1] * torch.cos(angles[:, 1:])
    xn = r * sin_prods[:, -1:]

    return torch.cat((x1, xs, xn), dim=1)

def logabsdet_sph_to_car(arr):
    # meant for batches of vectors, i.e. arr.shape = (mb, n)
    eps = 1e-8
    n = arr.shape[1]
    r = arr[:, -1]
    angles = arr[:, :-2]
    sin_angles = torch.sin(angles)
    sin_exp = torch.arange(n - 2, 0, -1)

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
    eye = torch.eye(*A.shape[1:]).repeat(A.shape[0], 1, 1)
    # A_triu_inv = torch.triangular_solve(eye, A_triu)[0]
    A_triu_inv = torch.linalg.solve_triangular(A_triu, eye, upper=True)
    # A_triu_inv = torch.linalg.inv(A_triu)

    u = torch.zeros_like(A[:, :, -1:])
    u[:, -1, :] = 1.
    v = A[:, -1:, :-1]
    v = torch.cat((v, torch.zeros_like(v[:, :, :1])), 2)

    assert torch.all(A == A_triu + u @ v)

    num = ((A_triu_inv @ u) @ v) @ A_triu_inv
    den = 1 + (v @ A_triu_inv) @ u

    return A_triu_inv - num / den

def gradient_r(inputs, norm, q):
    r = r_given_norm(inputs, norm, q)
    grad_r_theta = torch.autograd.grad(r, inputs, grad_outputs=torch.ones_like(r))[0]
    grad_r_theta_aug = torch.cat([- grad_r_theta, torch.ones_like(grad_r_theta[:, :1])], dim=1)

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
    def __init__(self, N, norm, q):
        super().__init__()
        self.N = N
        self.norm = norm
        self.q = q
        self.r_given_norm = r_given_norm

        spherical_names, jac = sph_to_cart_jacobian_sympy(self.N)
        self.spherical_names = spherical_names
        self.sph_to_cart_jac = sympytorch.SymPyModule(expressions=jac)

    def forward(self, inputs, context=None):
        theta_r = inflate_radius(inputs, self.norm, self.q)
        outputs = spherical_to_cartesian_torch(theta_r)

        logabsdet = self.logabs_pseudodet(inputs, theta_r)

        return outputs, logabsdet

    def just_forward(self, inputs, context=None):
        theta_r = inflate_radius(inputs, self.norm, self.q)
        outputs = spherical_to_cartesian_torch(theta_r)

        return outputs

    def inverse(self, inputs, context=None):
        raise NotImplementedError


    def logabs_pseudodet(self, inputs, theta_r):
        # spherical_dict = {name: theta_r[:, i] for i, name in enumerate(self.spherical_names)}
        # jac = self.sph_to_cart_jac(**spherical_dict).reshape(-1, self.N, self.N)
        jac = jacobian(spherical_to_cartesian_torch, theta_r).sum(-2)
        # assert torch.allclose(jac, jac_)
        jac_inv = sherman_morrison_inverse(jac.mT)
        #jac_inv = torch.inverse(jac.mT)


        grad_r = gradient_r(inputs, self.norm, self.q)
        #grad_r = torch.clamp(grad_r, min=-100, max=100)
        #grad_r_np = grad_r.detach().numpy().ravel()
        #theta_r_np = theta_r.detach().numpy().ravel()
        #plt.hist(grad_r_np, bins=100)
        #plt.show()
        #print("-" * 100)
        #idx = np.argsort(grad_r_np)
        #print(grad_r_np[idx])
        #print(theta_r_np[idx])
        #print(grad_r_np.min(), grad_r_np.max())
        jac_inv_grad = jac_inv @ grad_r
        fro_norm = torch.norm(jac_inv_grad.squeeze(), p='fro', dim=1)

        logabsdet_fro_norm = torch.log(torch.abs(fro_norm))
        logabsdet_s_to_c = logabsdet_sph_to_car(theta_r)

        logabsdet = logabsdet_s_to_c + logabsdet_fro_norm


        inputs_np = inputs.detach().numpy().ravel()
        fro_norm = logabsdet_fro_norm.detach().numpy().ravel()
        idx = np.argsort(inputs_np)
        #print(inputs_np[idx])
        #print(fro_norm[idx])
        #print("+" * 100)
        #print(fro_norm.min(), fro_norm.max())

        return logabsdet


class ConstrainedAngles(Transform):
    def __init__(self, elemwise_transform: Transform = Sigmoid()):
        super().__init__()
        self.elemwise_transform = elemwise_transform


    def forward(self, inputs, context=None):
        mask = torch.zeros_like(inputs)
        mask[:,-1] = torch.ones_like(inputs[:, -1])
        transformed_inputs, logabsdet_elemwise = self.elemwise_transform(inputs)
        outputs = mask * transformed_inputs + transformed_inputs
        logabsdet_last_elem = inputs.new_ones(inputs.shape[0]) * torch.log(torch.tensor(2.))

        return outputs, logabsdet_elemwise + logabsdet_last_elem

    def inverse(self, inputs, context=None):
        mask = torch.zeros_like(inputs)
        mask[:, -1] = torch.ones_like(inputs[:, -1])
        transformed_inputs, logabsdet_elemwise = self.elemwise_transform.inverse(inputs)
        outputs = mask * transformed_inputs + transformed_inputs
        logabsdet_last_elem = inputs.new_ones(inputs.shape[0]) * torch.log(torch.tensor(0.5))

        return outputs, logabsdet_elemwise + logabsdet_last_elem


class ConstrainedAnglesSigmoid(ConstrainedAngles):
    def __init__(self):
        super().__init__(elemwise_transform=CompositeTransform([Sigmoid(),
                                                                ScalarScale(scale=np.pi, trainable=False)]))

