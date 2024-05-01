"""
https://github.com/yperugachidiaz/invertible_densenets/blob/master/lib/layers/dense_layer.py

MIT License

Copyright (c) 2019 Ricky Tian Qi Chen
Copyright (c) 2021 Yura Perugachi-Diaz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import torch.nn.functional as F


class LipschitzDenseLayer(torch.nn.Module):
    def __init__(self, network, learnable_concat=False, lip_coeff=0.98):
        super(LipschitzDenseLayer, self).__init__()
        self.network = network
        self.lip_coeff = lip_coeff

        if learnable_concat:
            self.K1_unnormalized = torch.nn.Parameter(torch.tensor([1.]))
            self.K2_unnormalized = torch.nn.Parameter(torch.tensor([1.]))
        else:
            self.register_buffer("K1_unnormalized", torch.tensor([1.]))
            self.register_buffer("K2_unnormalized", torch.tensor([1.]))

    def get_eta1_eta2(self, beta=0.1):
        eta1 = F.softplus(self.K1_unnormalized) + beta
        eta2 = F.softplus(self.K2_unnormalized) + beta
        divider = torch.sqrt(eta1 ** 2 + eta2 ** 2)

        eta1_normalized = (eta1 / divider) * self.lip_coeff
        eta2_normalized = (eta2 / divider) * self.lip_coeff
        return eta1_normalized, eta2_normalized

    def forward(self, x):
        out = self.network(x)
        eta1_normalized, eta2_normalized = self.get_eta1_eta2()
        return torch.cat([x * eta1_normalized, out * eta2_normalized], dim=1)

    def build_clone(self):
        class LipschitzDenseLayerClone(torch.nn.Module):
            def __init__(self, network, eta1, eta2):
                super(LipschitzDenseLayerClone, self).__init__()
                self.network = network
                self.eta1_normalized = eta1_normalized
                self.eta2_normalized = eta2_normalized

            def forward(self, x, concat=True):
                out = self.network(x)
                if concat:
                    return torch.cat([x * self.eta1_normalized, out * self.eta2_normalized], dim=1)
                else:
                    return x * self.eta1_normalized, out * self.eta2_normalized

        with torch.no_grad():
            eta1_normalized, eta2_normalized = self.get_eta1_eta2()
            return LipschitzDenseLayerClone(self.network.build_clone(), eta1_normalized, eta2_normalized)

    def build_jvp_net(self, x, concat=True):
        class LipschitzDenseLayerJVP(torch.nn.Module):
            def __init__(self, network, eta1_normalized, eta2_normalized):
                super(LipschitzDenseLayerJVP, self).__init__()
                self.network = network
                self.eta1_normalized = eta1_normalized
                self.eta2_normalized = eta2_normalized

            def forward(self, v):
                out = self.network(v)
                return torch.cat([v * self.eta1_normalized, out * self.eta2_normalized], dim=1)

        with torch.no_grad():
            eta1_normalized, eta2_normalized = self.get_eta1_eta2()
            network, out = self.network.build_jvp_net(x)
            if concat:
                y = torch.cat([x * eta1_normalized, out * eta2_normalized], dim=1)
                return LipschitzDenseLayerJVP(network, eta1_normalized, eta2_normalized), y
            else:
                return LipschitzDenseLayerJVP(network, eta1_normalized,
                                              eta2_normalized), x * eta1_normalized, out * eta2_normalized