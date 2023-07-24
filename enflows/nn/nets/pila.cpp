/*
Source: https://github.com/mlvlab/MonotoneFlows/

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
*/
#include <torch/extension.h>

#define CONST(c, device_tensor) (torch::tensor(c, torch::dtype(torch::kFloat32).device(device_tensor.device())))


torch::Tensor pila_cpu_forward(torch::Tensor x, torch::Tensor kabcdmn) {
	const auto k = kabcdmn[0];
	const auto a = kabcdmn[1];
	const auto b = kabcdmn[2];
	const auto c = kabcdmn[3];
	const auto d = kabcdmn[4];
	const auto m = kabcdmn[5];
	const auto n = kabcdmn[6];

    auto x2 = x*x;
    auto x3 = x2*x;

    auto p = torch::min(k*x, CONST(0.01, x));
    auto q = torch::exp(p);
    auto r = a*x3 + b*x2 + c*x + d;

	return torch::where(x > 0, m*x+n, r*q);
}

torch::Tensor pila_cuda_forward(
    torch::Tensor x,
	torch::Tensor kabcdmn);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor pila_forward(
	torch::Tensor x,
	torch::Tensor kabcdmn) {
	TORCH_CHECK(kabcdmn.dim() == 1 && kabcdmn.size(0) == 7,
		"kabcdmn has wrong dim/size; it must be 1-dimensional 7-element tensor, but got dim size(0)",
		kabcdmn.dim(), kabcdmn.size(0))
	switch (x.device().type()) {
	case c10::kCUDA:
		CHECK_INPUT(x);
		CHECK_INPUT(kabcdmn);
		return pila_cuda_forward(x, kabcdmn);
	case c10::kCPU:
		return pila_cpu_forward(x, kabcdmn);
    default:
		TORCH_CHECK(false, "Unsupported device type, should be CPU or CUDA but got ", x.device().type());
	}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &pila_forward, "Pila forward");
}