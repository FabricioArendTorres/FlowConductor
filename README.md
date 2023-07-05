# nflows

<a href="https://github.com/FabricioArendTorres/enflows/actions/workflows/build_lint_test.yml"><img src="https://github.com/FabricioArendTorres/enflows/actions/workflows/build_lint_test.yml/badge.svg" alt="Build status"></a>

`nflows` is a comprehensive collection of [normalizing flows](https://arxiv.org/abs/1912.02762) using [PyTorch](https://pytorch.org).
nflows-extended, or `enflows`, is as the name suggests an extension of this package. 
The main focus lies in implementing more flow layers from the literature in one consistent framework.


## Setting up the Environment.
The environment set-up was tested with [mamba](https://github.com/mamba-org/mamba).
We assume cuda is available, and did not explicitely test a CPU setup.
In principle, it should also work with [conda](https://docs.conda.io/en/latest/) (just a lot slower), but we did not explicitely test that.
For using conda, exchange `mamba` with `conda` in the following commands.

The `.yml` file for the environment is given in `env/conda_env.yml`, and can be created from the base directory via:

```
(base) $  mamba env create --file env/conda_env.yml
(base) $  conda activate lflows_neurips
# This code should then work:
(enflows) $  python examples/lflows_conditional_moons.py
```

If you do not create it from the base directory, the pip install of the local package will not work.
In that case, you can try fixing it by running afterwards:

`(enflows) /lagrangian_flow_net$ pip install -e .`

Ideally, after a successfull install you should  be able to run and pass the unit tests with:

` 
(enflows) /lagrangian_flow_net$  pytest
`

## Usage

To define a flow:

```python
from enflows import transforms, distributions, flows

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

Main changes compared to the nflows repository include:

1. The addition of a few transformations.

- Sum-of-Sigmoid Layers
- [DeepSigmoid Layers / Neural Autoregressive Flows] (https://proceedings.mlr.press/v80/huang18d.html)
- Transformations for which the inverse is only known to exist, but not available: 
  - [Planar Flow](https://arxiv.org/abs/1912.02762) 
  - [Sylvester Flow](https://arxiv.org/abs/1803.05649)
- Conditional Versions of existing non-conditional transformations from nflows. Can be found for imports at `nflows.transforms.conditional.*`:
    - Planar Flow, Sylvester Flow
    - LU Transform
    - Orthogonal Transforms based on parameterized Householder projections
    - SVD based on the Orthogonal transforms
    - Shift Transform
- Conditional Versions of existing auto-regressive Variations, i.e. getting rid of the autoregressive parts.
    - [ConditionalPiecewiseRationalQuadraticTransform](https://proceedings.neurips.cc/paper/2019/hash/7ac71d433f282034e088473244df8c02-Abstract.html)
    - [ConditionalUMNNTransform](https://arxiv.org/abs/1908.05164)

2. Some extensions and fixes of the unit tests provided in `tests`.
