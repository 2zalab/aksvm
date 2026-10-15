"""Associated-kernel support vector machines.

Positive semi-definite kernels built from the associated kernels of
nonparametric smoothing (Beta, Gamma, Poisson, binomial, discrete triangular,
Conway-Maxwell-Poisson) through probability-product inner products.
"""

from .families import (FAMILIES, AssociatedFamily, BetaFamily, BinomialFamily, CMPFamily,
                       DiscreteTriangularFamily, GammaFamily, GaussianFamily, PoissonFamily,
                       get_family)
from .kernel import AssociatedKernel, infer_family
from .selection import alignf_weights, centered_alignment, select_feature_rho
from .svm import AssociatedKernelSVC

__version__ = "0.1.0"

__all__ = [
    "FAMILIES", "AssociatedFamily", "BetaFamily", "BinomialFamily", "CMPFamily",
    "DiscreteTriangularFamily", "GammaFamily", "GaussianFamily", "PoissonFamily",
    "get_family", "AssociatedKernel", "infer_family", "alignf_weights",
    "centered_alignment", "select_feature_rho", "AssociatedKernelSVC",
]
