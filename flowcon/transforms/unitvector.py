from __future__ import annotations

import torch

from flowcon.transforms.base import InputOutsideDomain, Transform


class UnitVector(Transform):
    """
    Bijective map from R^d to the surface of a sphere in R^(d+1) with an inverse
    stereographic projection.

    Inspired by
    https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/UnitVector

    """

    def __init__(self, features: int):
        super().__init__()
        self.dim_Rd = features
        self.dim_sphere = torch.nn.Parameter(torch.tensor(features + 1, dtype=torch.float32))

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        R^d to surface of sphere
        """
        assert inputs.shape[-1] == self.dim_Rd

        norm_sq = self.norm_sq(inputs)

        output = torch.concatenate([2 * inputs, norm_sq - 1], dim=-1) / (norm_sq + 1)
        return output, self.forward_logabsdet(inputs).view(-1)

    def inverse(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Surface of sphere to R^d
        """
        assert inputs.shape[-1] == self.dim_sphere
        self.check_inverse_range(inputs)

        outputs = (inputs / (1 - inputs[..., -1:]))[..., :-1]
        return outputs, -self.forward_logabsdet(outputs).view(-1)

    def check_inverse_range(self, inverse_inputs: torch.Tensor) -> None:
        if torch.abs(torch.max(self.norm_sq(inverse_inputs)) - 1.0) > 1e-4:
            raise InputOutsideDomain()

    def forward_logabsdet(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        sqrt{det{J^T J}} = (2 / (sum^2 + 1))^n.


        Parameters
        ----------
        inputs : torch.Tensor
            _description_

        Returns
        -------
        torch.Tensor
            _description_
        """
        n = self.dim_Rd
        log_2 = torch.log(torch.tensor(2).to(inputs))

        return n * (log_2 - torch.log1p(self.norm_sq(inputs).squeeze()))

    @staticmethod
    def norm_sq(x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the squared norm over the last axis of the provided tensor.

        Parameters
        ----------
        x : torch.Tensor
            input tensor, of shape [..., d], with d being the dimension summed over.

        Returns
        -------
        torch.Tensor
            Squared norm of shape [..., d], same as the input shape.
        """
        return torch.sum(x**2, dim=-1, keepdim=True)
