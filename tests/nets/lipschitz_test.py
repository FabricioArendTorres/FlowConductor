import torch
import torchtestcase
import unittest
from parameterized import parameterized_class
import numpy as np

from enflows.nn.nets import mlp, lipschitz, activations, lipschitz_dense
from torch.nn.utils.parametrize import is_parametrized
from types import SimpleNamespace
from enflows.utils.torchutils import tensor_to_np, batch_jacobian, logabsdet


def spectral_norm(model):
    list_singular_vals = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) or is_parametrized(m):
            U, S, Vt = np.linalg.svd(tensor_to_np(m.weight))
            list_singular_vals.append(np.max(S))
    return SimpleNamespace(mean=np.mean(list_singular_vals),
                           max=np.max(list_singular_vals),
                           min=np.min(list_singular_vals))


@parameterized_class(('input_dim', 'output_dim'),
                     [(10, 5),
                      (2, 20),
                      (50, 50)])
class TestLipschitzLayer(torchtestcase.TorchTestCase):
    def setUp(self) -> None:
        self.coef = 0.97
        self.eps = 1e-3

    def test_spectral_norms(self):
        for spectral_norm_param in (lipschitz.scaled_spectral_norm_powerits,
                                    lipschitz.scaled_spectral_norm_induced
                                    ):
            wrapper = lambda net: spectral_norm_param(net,
                                                      coeff=self.coef,
                                                      n_power_iterations=1)
            self._test_single_layer(wrapper)
            self._test_Lipschitz_dense(wrapper)

    def _test_single_layer(self, wrapper):
        net = torch.nn.Linear(self.input_dim, self.output_dim)
        net.weight.data.fill_(10)
        self.assertGreater(spectral_norm(net).mean, 1)

        wrapped_net = wrapper(net)
        self.assertAlmostEqual(spectral_norm(wrapped_net).mean, self.coef, delta=1e-4)

if __name__ == "__main__":
    unittest.main()
