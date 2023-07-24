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

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


namespace kernel {

template <typename scalar_t>
__device__ __forceinline__ scalar_t pila(scalar_t z, scalar_t k, scalar_t a, scalar_t b, scalar_t c, scalar_t d, scalar_t m, scalar_t n) {
const auto q = exp(k*z);
const auto r = a*z*z*z + b*z*z + c*z + d;

return (z>0) ? (m*z+n) : (r*q);
}

template <typename scalar_t>
__global__ void pila_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> kabcdmn,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> output
    ) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int numel = x.size(0);
    for (int i = index; i < numel; i += stride) {
        output[i] = pila(x[i], kabcdmn[0], kabcdmn[1], kabcdmn[2], kabcdmn[3], kabcdmn[4], kabcdmn[5], kabcdmn[6]);
    }
}

} // namespace kernel



torch::Tensor pila_cuda_forward(
    torch::Tensor x,
    torch::Tensor kabcdmn) {

    auto x_1d = x.view(-1);
    auto kabcdmn_1d = kabcdmn.view(-1);

    const int numel = x.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    auto output_1d = torch::zeros_like(x_1d);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pila_cuda_forward", ([&] {
        kernel::pila_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            x_1d.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            kabcdmn_1d.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            output_1d.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>()
        );
    }));

    return output_1d.view_as(x);
}
