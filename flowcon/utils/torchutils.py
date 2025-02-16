"""Various PyTorch utility functions."""

import random
from typing import Iterable, Optional

import numpy as np
import torch
from numpy.typing import ArrayLike
import torch.types

from flowcon.utils import typechecks as check


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def tile(x: torch.Tensor, n: int) -> torch.Tensor:
    if not check.is_positive_int(n):
        raise TypeError("Argument 'n' must be a positive integer.")
    x_ = x.reshape(-1)
    x_ = x_.repeat(n)
    x_ = x_.reshape(n, -1)
    x_ = x_.transpose(1, 0)
    x_ = x_.reshape(-1)
    return x_


def sum_except_batch(x: torch.Tensor, num_batch_dims: int = 1) -> torch.Tensor:
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    if not check.is_nonnegative_int(num_batch_dims):
        raise TypeError("Number of batch dimensions must be a non-negative integer.")
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)


def split_leading_dim(x: torch.Tensor, shape: Iterable[int]) -> torch.Tensor:
    """Reshapes the leading dim of `x` to have the given shape."""
    new_shape = torch.Size(shape) + x.shape[1:]
    return torch.reshape(x, new_shape)


def merge_leading_dims(x: torch.Tensor, num_dims: int) -> torch.Tensor:
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to one."""
    if not check.is_positive_int(num_dims):
        raise TypeError("Number of leading dims must be a positive integer.")
    if num_dims > x.dim():
        raise ValueError(
            "Number of leading dims can't be greater than total number of dims."
        )
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x: torch.Tensor, num_reps: int) -> torch.Tensor:
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    if not check.is_positive_int(num_reps):
        raise TypeError("Number of repetitions must be a positive integer.")
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)


def logabsdet(matrix: torch.Tensor) -> torch.Tensor:
    """
    Returns the log absolute determinant of a square matrix.

    Parameters
    ----------
    matrix : torch.Tensor
        Square matrix in a tensor of shape [dim, dim]

    Returns
    -------
    torch.Tensor
        Scalar valued logabsdet of the input matrix.
    """
    # Note: torch.logdet() only works for positive determinant.
    _, logabsdet = torch.slogdet(matrix)
    return logabsdet


def batch_JTJ_logabsdet(inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    jacs = batch_jacobian(outputs, inputs)
    logabsdet = 0.5 * torch.slogdet(torch.bmm(torch.transpose(jacs, -2, -1), jacs))[1]
    return logabsdet


def random_orthogonal(dim: int) -> torch.Tensor:
    """
    Returns a random orthogonal matrix as a tensor of shape [dim, dim]
    using the QR decomposition of a normaly distributed dim x dim matrix.

    Parameters
    ----------
    size : int
        Dimension of the matrix.

    Returns
    -------
    torch.Tensor
        Q component of the QR c, shape [dim, dim]
    """

    # Use the QR decomposition of a random Gaussian matrix.
    x = torch.randn(dim, dim)
    q, _ = torch.linalg.qr(x)
    return q


def get_num_parameters(model: torch.nn.Module) -> int:
    """
    Returns the number of trainable parameters in a model of type nets.Module
    :param model: nets.Module containing trainable parameters
    :return: number of trainable parameters in model
    """
    num_parameters = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            num_parameters += torch.numel(parameter)
    return num_parameters


def create_alternating_binary_mask(features: int, even: bool = True) -> torch.Tensor:
    """
    Creates a binary mask of a given dimension which alternates its masking.

    :param features: Dimension of mask.
    :param even: If True, even values are assigned 1s, odd 0s. If False, vice versa.
    :return: Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


def create_mid_split_binary_mask(features: int) -> torch.Tensor:
    """
    Creates a binary mask of a given dimension which splits its masking at the midpoint.

    :param features: Dimension of mask.
    :return: Binary mask split at midpoint of type torch.Tensor
    """
    mask = torch.zeros(features).byte()
    midpoint = features // 2 if features % 2 == 0 else features // 2 + 1
    mask[:midpoint] += 1
    return mask


def create_random_binary_mask(features: int) -> torch.Tensor:
    """
    Creates a random binary mask of a given dimension with half of its entries
    randomly set to 1s.

    :param features: Dimension of mask.
    :return: Binary mask with half of its entries set to 1s, of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    weights = torch.ones(features).float()
    num_samples = features // 2 if features % 2 == 0 else features // 2 + 1
    indices = torch.multinomial(
        input=weights, num_samples=num_samples, replacement=False
    )
    mask[indices] += 1
    return mask


def searchsorted(
    bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def cbrt(x: torch.Tensor) -> torch.Tensor:
    """Cube root. Equivalent to torch.pow(x, 1/3), but numerically stable."""
    return torch.sign(x) * torch.exp(torch.log(torch.abs(x)) / 3.0)


def gradient(
    y: torch.Tensor, x: torch.Tensor, grad_outputs: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def batchwise_dot_prod(bvector1: torch.Tensor, bvector2: torch.Tensor) -> torch.Tensor:
    return (bvector1 * bvector2).sum(-1)


def batch_jacobian(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    jac = []
    for d in range(g.shape[1]):
        jac.append(
            torch.autograd.grad(torch.sum(g[:, d]), x, create_graph=True)[0].view(
                x.shape[0], 1, x.shape[1]
            )
        )
    return torch.cat(jac, 1)


def batch_trace(M: torch.Tensor) -> torch.Tensor:
    return M.view(M.shape[0], -1)[:, :: M.shape[1] + 1].sum(1)


def sech2(x: torch.Tensor) -> torch.Tensor:
    return 1 / torch.cosh(x) ** 2


def np_to_tensor(
    array: ArrayLike, dtype: Optional[torch.dtype] = None, device: str = "cpu"
) -> torch.Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    if isinstance(array, np.ndarray):
        return torch.tensor(array, dtype=dtype, device=device)
    elif isinstance(array, torch.Tensor):
        return array.to(dtype).to(device)
    else:
        raise ValueError("Unknown Type: " + str(type(array)))


def tensor_to_np(tensor: torch.Tensor) -> ArrayLike:
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise ValueError("Unknown Type: " + str(type(tensor)))


def sample_rademacher_like(y: torch.Tensor) -> torch.Tensor:
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def safe_detach(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().requires_grad_(tensor.requires_grad)
