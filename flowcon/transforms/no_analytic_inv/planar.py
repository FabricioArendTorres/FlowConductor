import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from flowcon.transforms.base import Transform
from flowcon.utils import torchutils
import flowcon.utils.typechecks as check
from flowcon.transforms.orthogonal import HouseholderSequence


class PlanarTransform(Transform):
    """Implementation of the invertible transformation used in planar flow:
        f(z) = z + u * h(dot(w.T, z) + b)
    See Section 4.1 in https://arxiv.org/pdf/1505.05770.pdf.
    """

    def __init__(self, features: int = 2, num_iterations=25, lim = 50):
        """Initialise weights and bias.

        Args:
            features: Dimensionality of the distribution to be estimated.
        """
        super().__init__()
        self.w = nn.Parameter(torch.randn(1, features).normal_(0, 0.1))
        self.b = nn.Parameter(torch.randn(1).normal_(0, 0.1))
        self.u = nn.Parameter(torch.randn(1, features).normal_(0, 0.1))

    def forward(self, inputs, context=None):
        # if torch.mm(self.u, self.w.T) < -1:
        #     self.enforce_u_condition()
        u = self.get_constrained_u()
        _w = self.w
        _b = self.b
        _a = torch.mm(inputs, _w.T) + _b
        outputs = inputs + u * torch.tanh(_a)
        return outputs, self.forward_logabsdet(inputs).squeeze()

    def forward_logabsdet(self, inputs: torch.Tensor, context=None) -> torch.Tensor:
        # if torch.mm(self.u, self.w.T) < -1:
        #     self.enforce_u_condition()
        u = self.get_constrained_u()
        a = torch.mm(inputs, self.w.T) + self.b

        psi = (1 - torch.tanh(a) ** 2) * self.w
        abs_det = (1 + torch.mm(u, psi.T)).abs()
        log_det = torch.log(1e-7 + abs_det)
        return log_det

    #
    # def enforce_u_condition(self) -> None:
    #     """Enforce w^T u >= -1. When using h(.) = tanh(.), this is a sufficient condition
    #     for invertibility of the transformation f(z). See Appendix A.1.
    #     """
    #     wtu = torch.mm(self.u, self.w.T)
    #     m_wtu = -1 + torch.nn.functional.softplus(wtu)
    #     self.u.data = (
    #             self.u + (m_wtu - wtu) * (self.w / (torch.norm(self.w, p=2, dim=1) ** 2))
    #     )

    def get_constrained_u(self) -> torch.Tensor:
        """Enforce w^T u >= -1. When using h(.) = tanh(.), this is a sufficient condition
        for invertibility of the transformation f(z). See Appendix A.1.
        """
        wtu = torch.mm(self.u, self.w.T)
        m_wtu = -1 + torch.nn.functional.softplus(wtu)
        w_direction = (self.w / (torch.norm(self.w, p=2, dim=1) ** 2))
        return self.u + (m_wtu - wtu) * w_direction


class SylvesterTransform(Transform):
    """Implementation of the invertible transformation used in planar flow:
        f(z) = z + u * h(dot(w.T, z) + b)
    See Section 4.1 in https://arxiv.org/pdf/1505.05770.pdf.
    """

    def __init__(self, features: int = 2, num_householder=None, device="cuda"):
        """Initialise weights and bias.

        Args:
            features: Dimensionality of the distribution to be estimated.
        """
        super().__init__()

        self.n_diag_entries = features
        self.n_triangular_entries = ((features - 1) * features) // 2

        self.features = features
        if num_householder is None:
            num_householder = self.features
        self.num_householder = num_householder

        # Parameterization of matrices
        self.upper_indices = np.triu_indices(features, k=1)
        self.diag_indices = np.diag_indices(features)

        # R1
        self.upper_entries1 = nn.Parameter(torch.zeros(self.n_triangular_entries))
        self.log_upper_diag1 = nn.Parameter(torch.zeros(features))

        # R2
        self.upper_entries2 = nn.Parameter(torch.zeros(self.n_triangular_entries))
        self.log_upper_diag2 = nn.Parameter(torch.zeros(features))

        # Q
        self.Q_orth = HouseholderSequence(features=features, num_transforms=self.num_householder).to(device=device)

        # bias
        self.bias = nn.Parameter(torch.zeros(features))

        self._initialize()

    def _initialize(self):
        stdv = 1.0 / np.sqrt(self.features)
        init.uniform_(self.upper_entries1, -stdv, stdv)
        init.uniform_(self.upper_entries2, -stdv, stdv)
        init.uniform_(self.log_upper_diag1, -stdv, stdv)
        init.uniform_(self.log_upper_diag2, -stdv, stdv)
        init.constant_(self.bias, 0.0)

    def _create_R1(self):
        upper = self.upper_entries1.new_zeros(self.features, self.features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries1
        upper[self.diag_indices[0], self.diag_indices[1]] = torch.tanh(
            self.log_upper_diag1
        )
        return upper

    def _create_R2(self):
        upper = self.upper_entries2.new_zeros(self.features, self.features)
        upper[self.upper_indices[0], self.upper_indices[1]] = self.upper_entries2
        upper[self.diag_indices[0], self.diag_indices[1]] = torch.tanh(
            self.log_upper_diag2
        )
        return upper

    def dh_dx(self, x):
        return 1 - torch.tanh(x) ** 2

    def h(self, x):
        return torch.tanh(x)

    def forward(self, inputs, context=None):
        # Q = self.Q_orth.matrix().unsqueeze(0)
        # Qt =torch.transpose(Q, dim0=-2, dim1=-1)

        R1 = self._create_R1()  # self.Rtilde_unconstrained * self.triu_mask.squeeze()
        R2 = self._create_R2()  # self.R_unconstrained * self.triu_mask.squeeze()

        Qtz, _ = self.Q_orth.inverse(inputs)
        # Qtz = (Qt@inputs.unsqueeze(-1)).squeeze() # (n, d)
        RQtz = R1.unsqueeze(0) @ Qtz.unsqueeze(-1)  # (1,d,d) x (n, d, 1)
        preact = RQtz.squeeze() + self.bias.unsqueeze(0)
        act = self.h(preact)
        Ract = R2.unsqueeze(0) @ act.unsqueeze(-1)
        # QRact = Q@Ract
        QRact, _ = self.Q_orth.forward(Ract.squeeze())
        outputs = inputs + QRact.squeeze()

        deriv_act = self.dh_dx(preact)
        R_sq = torch.diag(R1) * torch.diag(R2)
        diag = R_sq.new_ones(self.features).unsqueeze(0) + deriv_act * R_sq.unsqueeze(0)
        logdet = torch.log(diag).sum(-1)
        # RQtz = torch.bmm(Qtz.)
        return outputs, logdet

    def inverse(self, inputs, context=None):
        raise NotImplementedError("ups")


class RadialTransform(Transform):
    """Implementation of the invertible transformation used in radial flow:
         f(z) = z + beta * h(alpha, r) * (z - z_0)
    See https://arxiv.org/pdf/1505.05770.pdf.
    """

    def __init__(self, features: int = 2, z_0=None):
        """Initialise weights and bias.

        Args:
            features: Dimensionality of the distribution to be estimated.
        """
        super().__init__()
        self.features = features
        self.d_cpu = torch.tensor(self.features)
        self.register_buffer("d", self.d_cpu)
        self.beta = nn.Parameter(torch.empty(1))
        lim = 1.0 / self.features
        nn.init.uniform_(self.beta, -lim - 1.0, lim - 1.0)
        self.alpha = nn.Parameter(torch.empty(1))
        nn.init.uniform_(self.alpha, -lim, lim)

        if z_0 is not None:
            self.z_0 = nn.Parameter(z_0)
        else:
            self.z_0 = nn.Parameter(torch.randn(self.features)[None])

    def forward(self, inputs, context=None):

        beta = torch.log(1 + torch.exp(self.beta)) - torch.abs(self.alpha)
        dz = inputs - self.z_0
        r = torch.linalg.vector_norm(dz, dim=list(range(1, self.z_0.dim())), keepdim=True)
        h_arr = beta / (torch.abs(self.alpha) + r)
        h_arr_ = -beta * r / (torch.abs(self.alpha) + r) ** 2
        z_ = inputs + h_arr * dz
        log_det = (self.d - 1) * torch.log(1 + h_arr) + torch.log(1 + h_arr + h_arr_)

        log_det = log_det.reshape(-1)

        return z_, log_det

    def inverse(self, inputs, context=None):
        raise NotImplementedError("ups")
