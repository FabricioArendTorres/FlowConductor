import torch
import torch.nn.functional as F
from typing import Optional
from torch.nn.utils.parametrize import register_parametrization


def scaled_spectral_norm(module: torch.nn.modules.Module,
                         domain,
                         codomain,
                         coeff=0.97,
                         name: str = 'weight',
                         n_power_iterations: int = 1,
                         eps: float = 1e-12,
                         dim: Optional[int] = None) -> torch.nn.modules.Module:
    r"""Applies spectral normalization to a parameter in the given module.
    Calls custom normalization modules instead of default pytorch ones.
    """
    weight = getattr(module, name, None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )

    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    register_parametrization(module, name,
                             _InducedSpectralNorm(weight, domain, codomain, n_power_iterations, dim, eps, coeff=coeff))
    return module


# noinspection PyProtectedMember
class _ScaledSpectralNorm(torch.nn.utils.parametrizations._SpectralNorm):
    def __init__(self,
                 weight: torch.Tensor,
                 domain,
                 codomain,
                 n_power_iterations: int = 2000,
                 dim: int = 0,
                 eps: float = 1e-12,
                 coeff=0.97
                 ) -> None:

        self.coeff = coeff
        self.domain = domain
        self.codomain = codomain
        super().__init__(weight, n_power_iterations, dim, eps)

    def compute_domain_codomain(self):
        if torch.is_tensor(self.domain):
            domain = asym_squash(self.domain)
            codomain = asym_squash(self.codomain)
        else:
            domain, codomain = self.domain, self.codomain
        return domain, codomain

    def estimate_max_singular_val(self, weight: torch.Tensor) -> torch.Tensor:
        weight_mat = self._reshape_weight_to_matrix(weight)
        if self.training:
            self._power_method(weight_mat, self.n_power_iterations)
        # See above on why we need to clone
        u = self._u.clone(memory_format=torch.contiguous_format)
        v = self._v.clone(memory_format=torch.contiguous_format)
        # The proper way of computing this should be through F.bilinear, but
        # it seems to have some efficiency issues:
        # https://github.com/pytorch/pytorch/issues/58093
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        return sigma

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim == 1:
            # Faster and more exact path, no need to approximate anything
            return F.normalize(weight, dim=0, eps=self.eps)
        else:
            sigma = self.estimate_max_singular_val(weight)
            # soft normalization: only when sigma larger than coeff
            normalization_term = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
            return weight / normalization_term


class _InducedSpectralNorm(_ScaledSpectralNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.autograd.no_grad()
    def _power_method(self, weight_mat: torch.Tensor, n_power_iterations: int) -> None:

        max_itrs = n_power_iterations

        domain, codomain = self.compute_domain_codomain()
        for _ in range(max_itrs):
            # Algorithm from http://www.qetlab.com/InducedMatrixNorm.
            self._u = self.normalize_u(torch.mv(weight_mat, self._v), codomain, out=self._u)
            self._v = self.normalize_v(torch.mv(weight_mat.t(), self._u), domain, out=self._v)

    @staticmethod
    def normalize_v(v, domain, out=None):
        if not torch.is_tensor(domain) and domain == 2:
            v = F.normalize(v, p=2, dim=0, out=out)
        elif domain == 1:
            v = projmax_(v)
        else:
            vabs = torch.abs(v)
            vph = v / vabs
            vph[torch.isnan(vph)] = 1
            vabs = vabs / torch.max(vabs)
            vabs = vabs ** (1 / (domain - 1))
            v = vph * vabs / vector_norm(vabs, domain)
        return v

    @staticmethod
    def normalize_u(u, codomain, out=None):
        if not torch.is_tensor(codomain) and codomain == 2:
            u = F.normalize(u, p=2, dim=0, out=out)
        elif codomain == float('inf'):
            u = projmax_(u)
        else:
            uabs = torch.abs(u)
            uph = u / uabs
            uph[torch.isnan(uph)] = 1
            uabs = uabs / torch.max(uabs)
            uabs = uabs ** (codomain - 1)
            if codomain == 1:
                u = uph * uabs / vector_norm(uabs, float('inf'))
            else:
                u = uph * uabs / vector_norm(uabs, codomain / (codomain - 1))
        return u


# Utility stuff
def leaky_elu(x, a=0.3):
    return a * x + (1 - a) * F.elu(x)


def asym_squash(x):
    return torch.tanh(-leaky_elu(-x + 0.5493061829986572)) * 2 + 3


def projmax_(v):
    """Inplace argmax on absolute value."""
    ind = torch.argmax(torch.abs(v))
    v.zero_()
    v[ind] = 1
    return v


def vector_norm(x, p):
    x = x.view(-1)
    return torch.sum(x ** p) ** (1 / p)
