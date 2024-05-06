Each submodule in `flowcon` reflects a different component in Normalizing Flows.
The following table provides a rough description.

| <div style="width:150px">Submodule</div>   | Description                                                                                                                                                                                                                                                                                   |
|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`flows`](`flowcon.flows`)                 | Contains the core logic of the Normalizing Flow for evaluating densities, sampling, and conditioning of these.                                                                                                                                                                                |                                                                                                                                                             
| [`distributions`](`flowcon.distributions`) | Contains different base distributions that can be used. The base flow that you will mostly need is [`flows.Flow`](`flowcon.flows.Flow`).                                                                                                                                                                      |
| [`transforms`](`flowcon.transforms`)       | Contains the invertible layers to be used within a [`flows.Flow`](`flowcon.flows.Flow`). Transforms implement at the very least a forward pass with corresponding log-absolute Jacobian. Most transforms also provide an inverse transform, which might however be more expensive to compute. |
| [`nn`](`flowcon.nn`)                       | Contains general (non-invertible) neural network layers and architectures. These might be used either for transforms or conditioning.                                                                                                                                                         |



Install
----
FlowConductor is installable via `pip`.
We recommend using a virtual environment, where you set up your pytorch version beforehand.
You can check out in `./docker` which pytorch versions we test for, but in general there shouldn't be any complications
for any version after 1.13.

You may either install the latest release from pipy:
```
$  pip install flowcon
```

or install it directly from github via pip 
```
$  pip install git+https://github.com/FabricioArendTorres/FlowConductor.git
```

Of course, you may also just download the repo and install it locally
```
$ git clone https://github.com/FabricioArendTorres/FlowConductor
$ cd FlowConductor
$ pip install . 
```


Getting Started
----------------------------

In general, you need to follow these step to build a Normalizing Flow with `flowcon`:

1. Decide on a Base Distribution. Usually this is a Gaussian.
2. Build the invertible transformations, i.e. the bijective layers that maps between your Base distribution and the target distribution.
3. Generate a Flow object, with the previous two components.

A more detailed explanation for different settings will follow at some point.
For now, take a look at the examples.


Examples
----------------------------
You can find some basic examples for the usage of this library in `examples/toy_2d.py` and `examples/conditional_toy_2d.py`.


Some Flow Architectures that work well
-----------------------------------------
There are many papers on Normalizing Flows and thus many possible combination of layers.
Some work well togethers - other don't. Although you might want to try a range of combinations for your project, we provide you a list of 
basic combinations that usually worked well for us.

### ActNorm + i-DenseNet + SVD

This architecture is based on the invertible DenseNet paper, which is an
extension of invertible ResNets.
We extended it by providing a more flexible activation function, a rescaled sine similar to SIREN networks,
in `flowcon.nn.CSIN`. 
Compared to the CLipSwish activation in the paper, the CSIN activation is much more flexible
in lower dimensions.
We used this architecture in [1]

```python
from flowcon import transforms, nn

def build_transform(n_features, num_layers=10) -> transforms.Transform:
    transform_list = []
    densenet_factory = (transforms.iResBlock.Factory()
                        .set_logabsdet_estimator(brute_force=True)
                        .set_densenet(dimension=2,
                                      densenet_depth=3,
                                      densenet_growth=16,
                                      activation_function=nn.CSin(10))
                        )
    for _ in range(num_layers):
        transform_list.append(transforms.ActNorm(features=2))
        transform_list.append(transforms.SVDLinear(features=n_features, num_householder=n_features))
        transform_list.append(densenet_factory.build())

    transform = transforms.CompositeTransform(transform_list)
    return transform
```

#### Caveats and things to consider
- If you data has more than 3 dimensions, turn of the brute-force estimation of the logabsdet.
- The inverse of iResBlocks is not available in closed-form and computed via a fix-point iteration. While it converges quickly, backpropagating through it is slow and not exact.
- Play around with the number of layers. The range from 5 to 30 is often reasonable.

[1] Torres, Fabricio Arend, et al. "Lagrangian Flow Networks for Conservation Laws." The Twelfth International Conference on Learning Representations. 2023.

### ActNorm + MaskedSumOfSigmoids

This is essentially based on our work in [2].
The SumOfSigmoids layers are really flexible element-wise transformations,
which have the nice property of getting linear for large / small inputs, and are only non-linear within some region around the origin.

Putting them into a masked autoregressive flow, where the parameters of each element-wise transformation are conditioned
on previous parameters, makes them powerful density estimators.

```python
from flowcon import transforms

def build_transform(n_features=2, num_layers=5) -> transforms.Transform:
    transform_list = []

    for _ in range(num_layers):
        transform_list.append(transforms.ActNorm(features=n_features))
        transform_list.append(transforms.ReversePermutation(features=n_features))
        transform_list.append(transforms.MaskedSumOfSigmoidsTransform(features=n_features,
                                                                      hidden_features=32))

    transform = transforms.CompositeTransform(transform_list)
    return transform
```

#### Caveats and things to consider
- You don't need a large `n_sigmoids` for the autoregressive version of the SumOfSigmoidTransform.
- Similarly, you do not need many layers.
- For higher dimensions you might want to try random permutations.
- Be careful with the inverse of SumOfSigmoids:
  - The inverse is only numerically approximated and based on a bisection search, and may in some cases be inexact.
  - This is an autoregressive model. The inverse is always painfully slow, as it can not be computed in parallel.

[2] Negri, Marcello Massimo, Fabricio Arend Torres, and Volker Roth. "Conditional Matrix Flows for Gaussian Graphical Models." Advances in Neural Information Processing Systems 36 (2023).


About the Package
-----------------------
During our research with Normalizing Flows (NFs) we noticed a lack of support for conditional NF
libraries in PyTorch, even though Normalizing Flows are by now a well-established and well-studied field.

We decided to work with the PyTorch package  [nflows](https://github.com/bayesiains/nflows) for Normalizing Flows,
as its core logic and design were very straight-forward to work with and extend.
While the core logic and code design is still used, we expanded the support for conditional transformations,
extended on the unit tests, added some new Normalizing Flow layers, and overall wish to develop this into a more mature library.

It should be noted that we mainly focus on conditional density estimation in structured data, i.e. we do not (yet?) provide current architectures for image generation.
If anyone wants to contribute, we would be open to that.

Backward-compatibility, Issues, and Contributing
-----------------------
This package is very much in an alpha phase.
That is, code-breaking changes at some points can not be avoided, and backward-compatibility
is not guaranteed when pulling a new version.

If you notice a bug, implementation error, or would like to request some additional feature,
please just open an issue on GitHub.

If you want to contribute yourself, feel free to send a pull-request!


License
-------

`flowcon` is licensed under the [MIT License](https://opensource.org/license/MIT), 
which it inherited from the [nflows](https://github.com/bayesiains/nflows) package it is based on.

Copyright (c) 2020 Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios

Copyright (c) 2023 Fabricio Arend Torres, Marcello Massimo Negri, Jonathan Aellen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

