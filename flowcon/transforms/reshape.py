from typing import Optional

import torch

import flowcon.utils.typechecks as check
from flowcon.transforms.base import Transform


class FlattenTransform(Transform):
    """
    Flattens an input tensor from shape ``[n, dim1, dim2, dim3, ...]``
    to ``[n, prod(dim1, dim2, dim3, ...)]``, and restores it in the inverse transform.

    This module caches the input shape during the first forward pass and prevents
    modifications after that. The cached shape is stored in a buffer to allow
    serialization.

    Attributes
    ----------
    _cached_shape : torch.Tensor
        Stores the original shape (excluding batch dimension).
    _locked : torch.Tensor
        A flag indicating whether `_cached_shape` is immutable.

    Methods
    -------
    forward(inputs, context=None)
        Flattens the input tensor while storing its original shape.
    inverse(inputs, context=None)
        Reshapes the flattened tensor back to its original shape.

    Examples
    --------
    >>> transform = Flatten()
    >>> x = torch.randn(4, 3, 32, 32)
    >>> y, log_det = transform.forward(x)
    >>> y.shape
    torch.Size([4, 3072])
    >>> x_recovered, log_det = transform.inverse(y)
    >>> x_recovered.shape
    torch.Size([4, 3, 32, 32])
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("_cached_shape", torch.empty(0, dtype=torch.long))
        self.register_buffer(
            "_locked", torch.tensor([0], dtype=torch.uint8)
        )  # Stored as tensor for compilation

    def forward(self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None):
        """
        Flattens the input tensor and caches its original shape.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor with shape ``(n, dim1, dim2, ...)``.
        context : Optional[torch.Tensor], default=None
            Optional conditioning tensor (unused in this transform).

        Returns
        -------
        outputs : torch.Tensor
            The flattened tensor with shape ``(n, prod(dim1, dim2, ...))``.
        logabsdet : torch.Tensor
            A tensor of zeros with shape ``(n,)``.
        """
        assert len(inputs.shape) >= 2, (
            f"Invalid {inputs.shape=}. Must have at least 2 axes."
        )
        if self._locked == 0:  # Avoid .item() for torch.compile compatibility
            shape_tensor = torch.tensor(
                inputs.shape[1:], dtype=torch.long, device=inputs.device
            )
            self._cached_shape = shape_tensor  # Store shape as tensor
            self._locked.fill_(1)  # Lock further modifications

        try:
            outputs = inputs.reshape(inputs.shape[0], torch.prod(self._cached_shape))
        except RuntimeError:
            # reraise, otherwise we get a rather cryptic error
            raise RuntimeError(
                f"Input with {inputs[1:].shape=} is incompatible with cached shape {self._cached_shape=}."
                + "This layer is immutable, you have to instantiate a new layer if you want to change shapes."
            ) from RuntimeError
        logabsdet = inputs.new_zeros(inputs.shape[0])
        return outputs, logabsdet

    def inverse(self, inputs: torch.Tensor, context: Optional[torch.Tensor] = None):
        """
        Restores the original shape of a flattened tensor.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensor with shape ``(n, prod(dim1, dim2, ...))``.
        context : Optional[torch.Tensor], default=None
            Optional conditioning tensor (unused in this transform).

        Returns
        -------
        outputs : torch.Tensor
            The reshaped tensor with its original dimensions.
        logabsdet : torch.Tensor
            A tensor of zeros with shape ``(n,)``.

        Raises
        ------
        RuntimeError
            If the original shape is not available.
        """
        if self._cached_shape.numel() == 0:
            raise RuntimeError("Original shape is not available. Call forward first.")
        outputs = inputs.reshape(inputs.shape[0], *self._cached_shape.tolist())
        logabsdet = inputs.new_zeros(inputs.shape[0])
        return outputs, logabsdet

    def load_state_dict(self, state_dict, strict=True):
        """
        Sidesteps the issue of having a dynamic inference of the shape
        for storing and restoring from / to a state_dict.
        """
        if "_cached_shape" in state_dict and state_dict["_cached_shape"].numel() > 0:
            self._cached_shape = state_dict["_cached_shape"]
        super().load_state_dict(state_dict, strict)


class SqueezeTransform(Transform):
    """A transformation defined for image data that trades spatial dimensions for channel
    dimensions, i.e. "squeezes" the inputs along the channel dimensions.

    Implementation adapted from https://github.com/pclucas14/pytorch-glow and
    https://github.com/chaiyujin/glow-pytorch.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(self, factor=2):
        super(SqueezeTransform, self).__init__()

        if not check.is_int(factor) or factor <= 1:
            raise ValueError("Factor must be an integer > 1.")

        self.factor = factor

    def get_output_shape(self, c, h, w):
        return (c * self.factor * self.factor, h // self.factor, w // self.factor)

    def forward(self, inputs, context=None):
        if inputs.dim() != 4:
            raise ValueError("Expecting inputs with 4 dimensions")

        batch_size, c, h, w = inputs.size()

        if h % self.factor != 0 or w % self.factor != 0:
            raise ValueError("Input image size not compatible with the factor.")

        inputs = inputs.view(
            batch_size, c, h // self.factor, self.factor, w // self.factor, self.factor
        )
        inputs = inputs.permute(0, 1, 3, 5, 2, 4).contiguous()
        inputs = inputs.view(
            batch_size,
            c * self.factor * self.factor,
            h // self.factor,
            w // self.factor,
        )

        return inputs, inputs.new_zeros(batch_size)

    def inverse(self, inputs, context=None):
        if inputs.dim() != 4:
            raise ValueError("Expecting inputs with 4 dimensions")

        batch_size, c, h, w = inputs.size()

        if c < 4 or c % 4 != 0:
            raise ValueError("Invalid number of channel dimensions.")

        inputs = inputs.view(
            batch_size, c // self.factor**2, self.factor, self.factor, h, w
        )
        inputs = inputs.permute(0, 1, 4, 2, 5, 3).contiguous()
        inputs = inputs.view(
            batch_size, c // self.factor**2, h * self.factor, w * self.factor
        )

        return inputs, inputs.new_zeros(batch_size)
