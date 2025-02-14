import torch

from flowcon.distributions import BaseDistribution


class Uniform(BaseDistribution):
    """
    A multivariate uniform distribution with independent components over the box [low, high).
    """

    def __init__(
        self,
        dim: int,
        low: torch.Tensor = None,
        high: torch.Tensor = None,
    ):
        """
        Constructor.

        Parameters
        ----------
        dim : int
            Dimension of Distribution.
        low : torch.Tensor, optional
            Optional lower bounds of the uniform distribution, by default None.
            Initialized as -1 if None.
            Needs to be squeezable to shape [dim].
        high : torch.Tensor, optional
            Optional upper bounds of the uniform distribution, by default None.
            Initialized as 1 if None.
            Needs to be squeezable to shape [dim].
        """
        super().__init__(dim=dim)

        low = -torch.ones(self.dim) if low is None else torch.squeeze(low)
        high = torch.ones(self.dim) if high is None else torch.squeeze(high)

        assert len(low.shape) == 1 and low.shape[0] == self.dim
        assert len(high.shape) == 1 and high.shape[0] == self.dim
        assert torch.all(low < high), (
            "All elements in 'low' must be smaller than 'high'"
        )

        self.register_buffer("_low", low.reshape(1, self.dim))
        self.register_buffer("_high", high.reshape(1, self.dim))
        self.register_buffer(
            "_log_norm_const", -torch.log(high - low).sum(dim=0).reshape(1)
        )

    def log_prob(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[1] == self.dim

        lb = self._low <= inputs
        ub = self._high > inputs
        in_bounds = torch.all(lb & ub, dim=1)

        log_prob = torch.full_like(
            in_bounds, float("-inf"), dtype=torch.get_default_dtype()
        )
        log_prob[in_bounds] = self._log_norm_const
        return log_prob

    def _sample(self, num_samples: int) -> torch.Tensor:
        rand = torch.rand(
            (num_samples, self.dim), dtype=self._low.dtype, device=self._low.device
        )
        return self._low + rand * (self._high - self._low)


class Uniform(Uniform):
    pass
