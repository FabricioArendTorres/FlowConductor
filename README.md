# FlowConductor: (Conditional) Normalizing Flows and </br> bijective Layers for Pytorch

<a href="https://github.com/FabricioArendTorres/enflows/actions/workflows/build_lint_test.yml"><img src="https://github.com/FabricioArendTorres/enflows/actions/workflows/build_lint_test.yml/badge.svg" alt="Build status"></a>
<a href="https://codecov.io/gh/FabricioArendTorres/enflows" >
<img src="https://codecov.io/gh/FabricioArendTorres/enflows/graph/badge.svg?token=UPQ2ZNQ6G4"/>
</a>

FlowConductor provides a collection of [normalizing flows](https://arxiv.org/abs/1912.02762) in  [PyTorch](https://pytorch.org).
It's core logic and transformations were originally based on the [nflows package](https://github.com/bayesiains/nflows).
The main focus lies in implementing more flow layers from the literature in one consistent framework, and adding support for conditional normalizing flows.
In the original nflows package, conditional networks were restricted to using a conditional base distribution. 
In `FlowConductor`, nearly every layer can be conditional :).
In particular, we support conditional transformations based on hypernetworks.


While using the package, we extensively expanded it, implementing layers such as invertible residual networks, and focusing on conditional normalizing flows.
In addition, we improved the testing, so that the Jacobian determinants are actually reliably compared to a reference based on brute-force autograd.

The bijective layers we additionally provide includes but are not limited to Planar Flows, (conditional) invertible ResNets/DenseNets, a variant of neural autoregressive flows, and a basic support of continuous normalizing flows (and FFJORD) based on the `torchdiffeq` package.


## Install

Install PyTorch first (CPU or CUDA), then install FlowConductor into that environment.
Dependencies are unpinned so the library stays usable across PyTorch builds.
CI currently tests the versions listed under Docker below.

### uv (recommended)

[uv](https://docs.astral.sh/uv/) is the project package manager.

```
$ git clone https://github.com/FabricioArendTorres/FlowConductor
$ cd FlowConductor
$ uv pip install -e ".[examples]"
```

From GitHub without cloning:

```
$ uv pip install git+https://github.com/FabricioArendTorres/FlowConductor.git
```

### pip

```
$ pip install flowcon
```

or from GitHub:

```
$ pip install git+https://github.com/FabricioArendTorres/FlowConductor.git
```

or from a local checkout:

```
$ git clone https://github.com/FabricioArendTorres/FlowConductor
$ cd FlowConductor
$ pip install .
```

### Docker

A single parameterized image lives at `docker/Dockerfile`. Pass a `pytorch/pytorch` tag as `PYTORCH_TAG`:

```
$ git clone https://github.com/FabricioArendTorres/FlowConductor
$ cd FlowConductor

$ docker build -f docker/Dockerfile --build-arg PYTORCH_TAG=latest -t flowc .

$ docker run --rm flowc pytest tests
```

Tags we test in CI: `1.13.1-cuda11.6-cudnn8-runtime`, `2.1.2-cuda12.1-cudnn8-runtime`, `2.2.2-cuda12.1-cudnn8-runtime`, and `latest`.

To work interactively with the repo mounted:

```
$ docker run --rm -it -v .:/flowc flowc python examples/toy_2d.py
```

### Development

```
$ uv pip install -e ".[dev,examples]"
$ pre-commit install
$ ruff check
$ pyrefly check
$ pytest
```

The devcontainer is a normal Python 3.10 image with uv. It is not the PyTorch CI image: uv owns the editor venv (CPU torch). Official `pytorch/pytorch` tags are used in CI via `docker/Dockerfile`.

## Package Usage

As the core is based on `nflows`, its usage is similar. To define a flow:

```python
from flowcon import transforms, distributions, flows

# Define an invertible transformation.
transform = transforms.CompositeTransform([
  transforms.MaskedAffineAutoregressiveTransform(features=2, hidden_features=4),
  transforms.RandomPermutation(features=2)
])

# Define a base distribution.
base_distribution = distributions.StandardNormal(shape=[2])

# Combine into a flow.
flow = flows.Flow(transform=transform, distribution=base_distribution)
```

To evaluate log probabilities of inputs:
```python
log_prob = flow.log_prob(inputs)
```

To sample from the flow:
```python
samples = flow.sample(num_samples)
```

Additional examples of the workflow are provided in [examples folder](examples/).
# Changes and added features compared to nflows
The core logic of the code for LFlows (i.e. the `nflows/` directory) is based on the [nflows package](https://github.com/bayesiains/nflows).
Aside from added features, our extension provides tests for the calculation of the Jacobian log-determinant, which is at the heart of Normalizing Flow.

Added Layers / Flwos:

- [(Conditional) Sum-of-Sigmoid Layers](https://arxiv.org/abs/2306.07255)
- [Cholesky Outer Product for flows on symmetric positive definite matrices](https://arxiv.org/abs/2306.07255)
- [Lipschitz Constrained invertible DenseNets](https://arxiv.org/abs/2010.02125)
  In particular, we provide three ways to condition these of these transformations without affecting the invertibility.
- Transformations for which the inverse is only known to exist, but not available: 
  - [(Conditional) Planar Flow](https://arxiv.org/abs/1912.02762) 
  - [(Conditional) Sylvester Flow](https://arxiv.org/abs/1803.05649)
- Conditional Versions of existing non-conditional transformations from nflows. Can be found for imports at `nflows.transforms.conditional.*`:
    - LU Transform
    - Orthogonal Transforms based on parameterized Householder projections
    - SVD based on the Orthogonal transforms
    - Shift Transform
- Conditional Versions of existing auto-regressive Variations, i.e. getting rid of the autoregressive parts.
    - [ConditionalPiecewiseRationalQuadraticTransform](https://proceedings.neurips.cc/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html)
    - [ConditionalUMNNTransform](https://arxiv.org/abs/1908.05164)
