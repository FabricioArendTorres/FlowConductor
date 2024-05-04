"""
The Python package FlowConductor, or short `flowcon`, provides a collection of Normalizing Flows
 architectures and utilities in PyTorch.
We specifically focus on conditional Normalizing Flows, which may find use in tasks such as variational inference
of conditional density estimation.

.. include:: ./documentation.md
"""
from flowcon.flows import Flow, MaskedAutoregressiveFlow

__all__ = ['Flow']