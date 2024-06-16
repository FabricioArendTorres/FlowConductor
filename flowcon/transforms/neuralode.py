import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from typing import *
from flowcon.nn.neural_odes import RegularizedODEfunc
from flowcon.transforms import Transform

__all__ = ["NeuralODE"]


class NeuralODE(Transform):
    """
    Transformation given by a neural ode.
    """

    def __init__(self, odefunc, T=1.0, train_T=False, regularization_fns=None,
                 solver=Literal['dopri5', 'dopri8', 'bosh3'], atol=1e-5, rtol=1e-5):
        super(NeuralODE, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)
        self.odefunc = odefunc
        self.nreg = nreg
        self.regularization_states = None
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}

    def forward(self, inputs, context=None):
        return self._ode_transform(inputs, context=context, logpz=None, integration_times=None)

    def inverse(self, outputs, context=None):
        return self._ode_transform(outputs, context=context, logpz=None, integration_times=None,
                                   reverse=True)

    def _ode_transform(self, z, context=None, logpz=None, integration_times=None, reverse=False):
        _logpz = torch.zeros(z.shape[0], 1).to(z)
        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # Add regularization states.

        if self.training:
            reg_states = tuple(torch.tensor(0).to(z) for _ in range(self.nreg))
            odeint_params = self.odeint_train_params(_logpz, reg_states, z)
        else:
            odeint_params = self.odeint_test_params(_logpz, z)
        reg_states = tuple(torch.tensor(0).to(z) for _ in range(self.nreg))
        odeint_params = self.odeint_train_params(_logpz, reg_states, z)

        state_t = odeint(
            func=self.odefunc,
            t=integration_times.to(z),
            **odeint_params,
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        self.regularization_states = state_t[2:]

        return z_t, -logpz_t

    def odeint_train_params(self, _logpz, reg_states, z):
        atol, rtol, method, options = self.atol, self.rtol, self.solver, self.solver_options
        y0 = (z, _logpz) + reg_states
        adjoint_options = {"norm": "seminorm"}

        return {"atol": atol, "method": method, "options": options, "rtol": rtol, "y0": y0,
                "adjoint_options": adjoint_options}

    def odeint_test_params(self, _logpz, z):
        y0 = (z, _logpz)
        atol = self.test_atol
        rtol = self.test_rtol
        method = self.test_solver
        adjoint_options = {"norm": "seminorm"}
        options = self.solver_options

        return {"atol": atol, "method": method, "options": options, "rtol": rtol, "y0": y0,
                "adjoint_options": adjoint_options}

    def get_regularization_states(self):
        reg_states = self.regularization_states
        self.regularization_states = None
        return reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()


class _ODENetWrapper(torch.nn.Module):
    """
    A wrapper around the `torch.nn.Module` that provides the dynamics of the neural ode for the continuous
    normalizing flow.

    You should not have to create objects of this class yourself,
    since it is being called within the NeuralODE transforms.

    Essentially it just provides a function that outputs the network value combined with an estimate
    of the trace of the Jacobian of the network (i.e. the divergence).
    This has to be a separate `torch.nn.Module` due to `torchdiffeq`.
    """

    def __init__(self,
                 dynamics_network,
                 divergence_fn_train: Literal["approximate", "brute_force"] = "approximate",
                 divergence_fn_test: Literal["approximate", "brute_force"] = "brute_force",
                 sampler: Literal["rademacher", "gaussian"] = "rademacher",
                 ):
        super().__init__()

        nreg = 0

        self.diffeq = dynamics_network
        self.nreg = nreg
        self.solver_options = {}
        self.rademacher = True

        divergences = dict(approximate=divergence_approx,
                           brute_force=divergence_bf)
        self.sample_like = dict(rademacher=sample_rademacher_like,
                                gaussian=sample_gaussian_like)[sampler]

        self.divergence_fn_train = divergences[divergence_fn_train]
        self.divergence_fn_test = divergences[divergence_fn_test]

        self.register_buffer("_num_evals", torch.tensor(0.))
        self.before_odeint()

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()

    def forward(self, t: Union[torch.Tensor, float], states: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        t       time
        states  current state (a tuple of current position and integrated divergence, i.e. the intermediate logabsdet)

        Returns Dynamics of the states, as to be used in odeint.
        -------

        """
        assert len(states) >= 2
        y = states[0]

        # increment num evals
        self._num_evals += 1

        # convert to tensor
        if not torch.is_tensor(t):
            t = torch.tensor(t).type_as(y)
        else:
            t = t.type_as(y)
        batchsize = y.shape[0]

        # Sample and fix the noise.
        if self._e is None:
            self._e = self.sample_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)

            dy = self.diffeq(t, y)
            if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
                divergence = divergence_bf(dy, y).view(batchsize, 1)
            else:
                if self.training:
                    divergence = self.divergence_fn_train(dy, y, e=self._e).view(batchsize, 1)
                else:
                    divergence = self.divergence_fn_train(dy, y, e=self._e).view(batchsize, 1)
        d_states_dt = dy, divergence.squeeze()
        return d_states_dt


class SimpleCNF(Transform):
    def __init__(self, dynamics_network, train_T=True, T=1.0,
                 solver:Literal['dopri5', 'dopri8', 'bosh3']='dopri5', atol=1e-5, rtol=1e-5,
                 divergence_fn:Literal["approximate", "brute_force"]="approximate",
                 eval_mode_divergence_fn:Literal["approximate", "brute_force"]="approximate"):
        super(SimpleCNF, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        nreg = 0

        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        self.odefunc = _ODENetWrapper(dynamics_network)
        self.nreg = nreg
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}
        self.rademacher = True

        self.odefunc.before_odeint()

    def num_evals(self):
        return self.odefunc.num_evals()

    def forward(self, inputs, context=None):
        return self.integrate(inputs, context=context, logpz=None, integration_times=None)

    def inverse(self, inputs, context=None):
        return self.integrate(inputs, context=context, logpz=None, integration_times=None, reverse=True)

    def integrate(self, z, context=None, logpz=None, integration_times=None, reverse=False):
        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        if self.training:
            state_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                options=self.solver_options,
                adjoint_options={"norm": "seminorm"}
                # step_size = self.solver_options["step_size"]
            )
        else:
            state_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
                adjoint_options={"norm": "seminorm"}
                # step_size=self.solver_options["step_size"]
            )

        z_t, logpz_t = tuple(s[1] for s in state_t)
        return z_t, logpz_t.squeeze()


class CompactTimeVariableCNF(nn.Module):
    start_time = 0.0
    end_time = 1.0

    def __init__(self, dynamics_network, solver:Literal['dopri5', 'dopri8', 'bosh3']='dopri5', atol=1e-5, rtol=1e-5,
                 divergence_fn:Literal["approximate", "brute_force"]="approximate"):
        super(CompactTimeVariableCNF, self).__init__()
        assert divergence_fn in ("brute_force", "approximate")

        nreg = 0

        self.diffeq = dynamics_network
        self.nreg = nreg
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}
        self.rademacher = True

        if divergence_fn == "brute_force":
            self.divergence_fn = divergence_bf
        elif divergence_fn == "approximate":
            self.divergence_fn = divergence_approx



        self.register_buffer("_num_evals", torch.tensor(0.))
        self.before_odeint()

        self.odeint_kwargs = dict(
            train=dict(
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
                options=self.solver_options,
                adjoint_options={"norm": "seminorm"}),
            test=dict(
                atol=self.test_atol,
                rtol=self.test_rtol,
                method=self.test_solver,
                adjoint_options={"norm": "seminorm"})
        )

    def integrate(self, t0, t1, z, logpz=None):
        _logpz = torch.zeros(z.shape[0], 1).to(z) if logpz is None else logpz
        initial_state = (t0, t1, z, _logpz)

        integration_times = torch.tensor([self.start_time, self.end_time]).to(t0)

        # Refresh the odefunc statistics.
        self.before_odeint(e=self.sample_e_like(z))

        self.get_odeint_kwargs()
        state_t = odeint(
            func=self,
            y0=initial_state,
            t=integration_times,
            **self.get_odeint_kwargs()
        )
        _, _, z_t, logpz_t = tuple(s[-1] for s in state_t)

        return z_t, logpz_t

    def forward(self, s, states):
        assert len(states) >= 2
        t0, t1, y, _ = states
        ratio = (t1 - t0) / (self.end_time - self.start_time)

        # increment num evals
        self._num_evals += 1

        # Sample and fix the noise.
        if self._e is None:
            self._e = self.sample_e_like(y)

        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t = (s - self.start_time) * ratio + t0
            dy = self.diffeq(t, y)
            dy = dy * ratio.reshape(-1, *([1] * (y.ndim - 1)))

            divergence = self.calculate_divergence(y, dy)

        return tuple([torch.zeros_like(t0), torch.zeros_like(t1), dy, -divergence])

    def sample_e_like(self, y):
        if self.rademacher:
            return sample_rademacher_like(y)
        else:
            return sample_gaussian_like(y)

    def calculate_divergence(self, y, dy):
        # Hack for 2D data to use brute force divergence computation.
        if not self.training and dy.view(dy.shape[0], -1).shape[1] == 2:
            divergence = divergence_bf(dy, y).view(-1, 1)
        else:
            if self.training:
                divergence = self.divergence_fn(dy, y, e=self._e).view(-1, 1)
            else:
                divergence = divergence_bf(dy, y, e=self._e).view(-1, 1)
        return divergence

    def get_odeint_kwargs(self):
        if self.training:
            return self.odeint_kwargs["train"]
        else:
            return self.odeint_kwargs["test"]

    def before_odeint(self, e=None):
        self._e = e
        self._num_evals.fill_(0)

    def num_evals(self):
        return self._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)


def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()


def divergence_approx(f, y, e=None):
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    e_dzdx_e = e_dzdx * e
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    return approx_tr_dzdx
