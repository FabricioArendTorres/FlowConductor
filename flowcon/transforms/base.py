"""Basic definitions for the transforms module."""

from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import nn

import flowcon.utils.typechecks as check


class InverseNotAvailable(Exception):
    """Exception to be thrown when a transform does not have an inverse."""

    pass


class InputOutsideDomain(Exception):
    """Exception to be thrown when the input to a transform is not within its domain."""

    pass


class Transform(nn.Module):
    """Base class for all transform objects."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inverted = False

    def forward(
        self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A function that bijectively transforms the inputs,
        and returns the transformed input as well as the logabsdet of the transformation.
        The forward direction is always the efficient direction of the bijection.

        Parameters
        ----------
        inputs : torch.Tensor
            Input values that should be transformed.
        context : torch.Tensor, optional
            Context for conditioning the transforms applied to the inputs, if applicable, by default None
        """
        raise NotImplementedError()

    def inverse(
        self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A function that bijectively transforms the inputs,
        and returns the transformed input as well as the logabsdet of the transformation.
        The inverse direction is always the less efficient direction of the bijection.
        It might also be just an approximation, rather than the exact inverse.
        Avoid differentiating through this function if possible.

        Parameters
        ----------
        inputs : torch.Tensor
            Input values that should be transformed.
        context : torch.Tensor, optional
            Context for conditioning the transforms applied to the inputs, if applicable, by default None

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (outputs, total_logabsdet)

        Raises
        ------
        InverseNotAvailable
            Inverses always have to exist, but are not necessarily known.

        """
        raise InverseNotAvailable()

    def is_inverted(self) -> bool:
        """
        Method to help keep track of this class was inverted or not.
        If not, the forward direction is still the more efficient one.
        Two Inverted should cancel each other out.

        Returns
        -------
        bool
            Whether this layer was inverted.
        """
        return self._inverted


class Sequential(Transform):
    """
    A transform that composes multiple transforms sequentially,
    similar in spirit to torch.nn.Sequential.

    During the forward pass, the transforms are applied in order.
    During the inverse pass, they are applied in reverse order
    """

    def __init__(self, transforms: Iterable[Transform]):
        """
        Initialize the composite transform.

        Parameters
        ----------
        transforms : Iterable[Transform]
            An iterable of `Transform` objects to be applied sequentially.
        """
        super().__init__()
        self._transforms = nn.ModuleList(transforms)

    @staticmethod
    def _cascade(
        inputs: torch.Tensor,
        funcs: Iterable[
            Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ],
        context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies a sequence of functions to the inputs, accumulating log determinants.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor to transform.
        funcs : Iterable[Callable]
            Sequence of transformation functions.
        context : torch.Tensor
            Optional context tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Transformed output and accumulated log determinant.
        """
        batch_size = inputs.shape[0]
        outputs = inputs
        total_logabsdet = inputs.new_zeros(batch_size)
        for func in funcs:
            outputs, logabsdet = func(outputs, context)
            total_logabsdet += logabsdet
        return outputs, total_logabsdet

    def forward(
        self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the sequence of transforms in forward order.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor.
        context : Optional[torch.Tensor], default=None
            Optional conditioning tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Transformed output and accumulated log absolute determinant ln|det(JT(x))|.
        """
        funcs = self._transforms
        return self._cascade(inputs, funcs, context)

    def inverse(
        self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the sequence of transforms in reverse order.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor.
        context : Optional[torch.Tensor], default=None
            Optional conditioning tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Inverted output and accumulated log determinant ln|det(JT^{-1}(x))|
        """
        inverted_funcs = (transform.inverse for transform in reversed(self._transforms))
        return self._cascade(inputs, inverted_funcs, context)


class MultiscaleSequential(Transform):
    """
    A multiscale composite transform as described in the RealNVP paper.

    Splits the outputs along the given dimension after every transform, outputs one half, and
    passes the other half to further transforms. No splitting is done before the last transform.

    Note: Inputs could be of arbitrary shape, but outputs will always be flattened.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(self, num_transforms: int, split_dim: int = 1):
        """
        Constructor.

        Parameters
        ----------
        num_transforms : int
            Total number of transforms to be added.
        split_dim : int, optional
            dimension along which to split, by default 1
        """
        if not check.is_positive_int(split_dim):
            raise TypeError("Split dimension must be a positive integer.")

        super().__init__()
        self._transforms = nn.ModuleList()
        self._output_shapes = []
        self._num_transforms = num_transforms
        self._split_dim = split_dim

    def add_transform(
        self, transform: Transform, transform_output_shape: Tuple[int, ...]
    ) -> Optional[Tuple[int, ...]]:
        """
        Add a transform. Must be called exactly `num_transforms` times.

        Parameters
        ----------
        transform : Transform
            The `Transform` object to be added.
        transform_output_shape : Tuple[int, ...]
            shape of transform's outputs, excl. the first batch dimension.

        Returns
        -------
        Optional[Tuple[int, ...]]
            Input shape for the next transform, or None if adding the last transform.

        """
        assert len(self._transforms) <= self._num_transforms

        if len(self._transforms) == self._num_transforms:
            raise RuntimeError(
                f"Adding more than {self._num_transforms} transforms is not allowed."
            )

        if (self._split_dim - 1) >= len(transform_output_shape):
            raise ValueError("No split_dim in output shape")

        if transform_output_shape[self._split_dim - 1] < 2:
            raise ValueError(f"Size of dimension {self._split_dim} must be at least 2.")

        self._transforms.append(transform)

        if len(self._transforms) != self._num_transforms:  # Unless last transform.
            output_shape = list(transform_output_shape)
            output_shape[self._split_dim - 1] = (
                output_shape[self._split_dim - 1] + 1
            ) // 2
            output_shape = tuple(output_shape)

            hidden_shape = list(transform_output_shape)
            hidden_shape[self._split_dim - 1] = hidden_shape[self._split_dim - 1] // 2
            hidden_shape = tuple(hidden_shape)
        else:
            # No splitting for last transform.
            output_shape = transform_output_shape
            hidden_shape = None

        self._output_shapes.append(output_shape)
        return hidden_shape

    def forward(
        self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._split_dim >= inputs.dim():
            raise ValueError("No split_dim in inputs.")
        if self._num_transforms != len(self._transforms):
            raise RuntimeError(
                f"Expecting exactly {self._num_transforms} transform(s) to be added."
            )

        batch_size = inputs.shape[0]

        def cascade():
            hiddens = inputs

            for i, transform in enumerate(self._transforms[:-1]):
                transform_outputs, logabsdet = transform(hiddens, context)
                outputs, hiddens = torch.chunk(
                    transform_outputs, chunks=2, dim=self._split_dim
                )
                assert outputs.shape[1:] == self._output_shapes[i]
                yield outputs, logabsdet

            # Don't do the splitting for the last transform.
            outputs, logabsdet = self._transforms[-1](hiddens, context)
            yield outputs, logabsdet

        all_outputs = []
        total_logabsdet = inputs.new_zeros(batch_size)

        for outputs, logabsdet in cascade():
            all_outputs.append(outputs.reshape(batch_size, -1))
            total_logabsdet += logabsdet

        all_outputs = torch.cat(all_outputs, dim=-1)
        return all_outputs, total_logabsdet

    def inverse(self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None):
        if inputs.dim() != 2:
            raise ValueError("Expecting NxD inputs")
        if self._num_transforms != len(self._transforms):
            raise RuntimeError(
                f"Expecting exactly {self._num_transforms} transform(s) to be added."
            )

        batch_size = inputs.shape[0]

        rev_inv_transforms = [transform.inverse for transform in self._transforms[::-1]]

        split_indices = np.cumsum([np.prod(shape) for shape in self._output_shapes])
        split_indices = np.insert(split_indices, 0, 0)

        split_inputs = []
        for i in range(len(self._output_shapes)):
            flat_input = inputs[:, split_indices[i] : split_indices[i + 1]]
            split_inputs.append(flat_input.view(-1, *self._output_shapes[i]))
        rev_split_inputs = split_inputs[::-1]

        total_logabsdet = inputs.new_zeros(batch_size)

        # We don't do the splitting for the last (here first) transform.
        hiddens, logabsdet = rev_inv_transforms[0](rev_split_inputs[0], context)
        total_logabsdet += logabsdet

        for inv_transform, input_chunk in zip(
            rev_inv_transforms[1:], rev_split_inputs[1:]
        ):
            tmp_concat_inputs = torch.cat([input_chunk, hiddens], dim=self._split_dim)
            hiddens, logabsdet = inv_transform(tmp_concat_inputs, context)
            total_logabsdet += logabsdet

        outputs = hiddens

        return outputs, total_logabsdet


class Inverse(Transform):
    """
    Wraps a transform to create its inverse.

    This class inverts a given transform, effectively swapping its forward
    and inverse operations. The inversion status can be checked using
    the `Transform.is_inverted()` method.
    """

    def __init__(self, transform: Transform):
        """
        Initialize the inverse transform.

        Parameters
        ----------
        transform : Transform
            The transform to be inverted.
        """
        super().__init__()
        self._transform = transform
        self._inverted = not transform._inverted

    def forward(self, inputs, context=None):
        return self._transform.inverse(inputs, context)

    def inverse(self, inputs, context=None):
        return self._transform(inputs, context)


class RemoveContext(Transform):
    """
    A wrapper for transforms that ensures they do not receive external context.

    This is useful for autoregressive or coupling transforms that internally use conditioning.
    Wrapping a transform with this class forces it to condition only on itself,
    preventing issues with architectures like neural spline flows,
    which can be overly flexible when conditioned on additional variables.
    """

    def __init__(self, transform: Transform):
        """
        Constructor.

        Parameters
        ----------
        transform : Transform
            The transform that shall be inverted.
        """
        super().__init__()
        self._transform = transform
        self._inverted = not transform._inverted

    def forward(self, inputs, context=None):
        return self._transform.inverse(inputs, None)

    def inverse(self, inputs, context=None):
        return self._transform(inputs, None)
