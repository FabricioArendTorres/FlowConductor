"""
Base, including ClipSwish https://github.com/yperugachidiaz/invertible_densenets/blob/master/lib/layers/base/activations.py
CPila: https://github.com/mlvlab/MonotoneFlows


MIT License

Copyright (c) 2019 Ricky Tian Qi Chen
Copyright (c) 2020 Cheng Lu
Copyright (c) 2021 Yura Perugachi-Diaz
Copyright (c) 2022 Byeongkeun Ahn

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
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.cpp_extension as cpp_extension
import copy
import os
from pathlib import Path

dir_path = Path(__file__).resolve().parent


class FullSort(nn.Module):

    def forward(self, x):
        return torch.sort(x, 1)[0]


class MaxMin(nn.Module):

    def forward(self, x):
        b, d = x.shape
        max_vals = torch.max(x.view(b, d // 2, 2), 2)[0]
        min_vals = torch.min(x.view(b, d // 2, 2), 2)[0]
        return torch.cat([max_vals, min_vals], 1)


class LipschitzCube(nn.Module):

    def forward(self, x):
        return (x >= 1).to(x) * (x - 2 / 3) + (x <= -1).to(x) * (x + 2 / 3) + ((x > -1) * (x < 1)).to(x) * x ** 3 / 3


class SwishFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, beta):
        beta_sigm = torch.sigmoid(beta * x)
        output = x * beta_sigm
        ctx.save_for_backward(x, output, beta)
        return output / 1.1

    @staticmethod
    def backward(ctx, grad_output):
        x, output, beta = ctx.saved_tensors
        beta_sigm = output / x
        grad_x = grad_output * (beta * output + beta_sigm * (1 - beta * output))
        grad_beta = torch.sum(grad_output * (x * output - output * output)).expand_as(beta)
        return grad_x / 1.1, grad_beta / 1.1


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


class Sin(nn.Module):
    def __init__(self, w0=1):
        super(Sin, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(x * self.w0) / self.w0

    def build_clone(self):
        return copy.deepcopy(self)


class CSin(nn.Module):
    def __init__(self, w0=1):
        super(CSin, self).__init__()
        self.w0 = w0

    def forward(self, x):
        x = torch.cat((x, -x), 1)
        return torch.sin(x * self.w0) / self.w0

    def build_clone(self):
        return copy.deepcopy(self)


class LeakyLSwish(nn.Module):

    def __init__(self):
        super(LeakyLSwish, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([-3.]))
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha)
        return alpha * x + (1 - alpha) * (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


class CLipSwish(nn.Module):

    def __init__(self):
        super(CLipSwish, self).__init__()
        self.swish = Swish()

    def forward(self, x):
        x = torch.cat((x, -x), 1)
        return self.swish(x).div_(1.004)


'''
By default, forward() and backward() of torch.autograd.Function
   is supplied with torch.no_grad() context.
   Hence, no need to worry about that inefficiency.
'''

#
# pila_cpp = cpp_extension.load(name='pila_cpp', sources=[dir_path / 'pila.cpp', dir_path / 'pila.cu'],
#                               extra_cuda_cflags=['-allow-unsupported-compiler'], verbose=True)
# class PilaFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, kabcdmn):
#         y = pila_cpp.forward(x, kabcdmn)
#         ctx.save_for_backward(x, kabcdmn)
#         return y
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         x, kabcdmn = ctx.saved_tensors
#         k, a, b, c, d, m, n = kabcdmn
#         kabcdmn_new = torch.stack([k, k * a, k * b + 3 * a, k * c + 2 * b, k * d + c, 0 * k, m])
#         return grad_output * PilaFunction.apply(x, kabcdmn_new), None
#
#
# class Pila(nn.Module):
#     def __init__(self, k=5, device=None):
#         super(Pila, self).__init__()
#         assert k > 0
#         self.k = k
#
#     def forward(self, x):
#         k = self.k
#         kabcdmn = torch.tensor((k, k ** 2 / 2, -k, 1, 0, 1, 0), dtype=x.dtype, device=x.device)
#         return PilaFunction.apply(x, kabcdmn)
#
#
# class CPila(nn.Module):
#     def __init__(self, k=5, device=None):
#         super(CPila, self).__init__()
#         assert k > 0
#         self.k = k
#         self.pila = Pila(k=k, device=device)
#
#     def forward(self, x):
#         x = torch.cat((x - 0.2, -x - 0.2), 1)
#         return self.pila(x) / 1.06
#
#     def build_clone(self):
#         return copy.deepcopy(self)
#
#     def build_jvp_net(self, x):
#         class CPilaJVP(nn.Module):
#             def __init__(self, grad):
#                 super(CPilaJVP, self).__init__()
#                 self.register_buffer('grad', grad)
#
#             def forward(self, x):
#                 return torch.cat((x, x), dim=1) * self.grad
#
#         with torch.no_grad():
#             y = self.forward(x)
#
#             k = self.k
#             a, b, c, d, m, n = k ** 2 / 2, -k, 1, 0, 1, 0
#             kabcdmn = torch.tensor((k, k * a, k * b + 3 * a, k * c + 2 * b, k * d + c, 0 * k, m), dtype=x.dtype,
#                                    device=x.device)
#             grad = torch.cat((pila_cpp.forward(x - 0.2, kabcdmn), -pila_cpp.forward(-x - 0.2, kabcdmn)), dim=1)
#             grad.div_(1.06)
#
#             return CPilaJVP(grad), y
