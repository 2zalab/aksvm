# aksvm — Associated-Kernel Support Vector Machines

[![tests](https://github.com/2zalab/aksvm/actions/workflows/tests.yml/badge.svg)](https://github.com/2zalab/aksvm/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/aksvm.svg)](https://pypi.org/project/aksvm/)
[![Python](https://img.shields.io/pypi/pyversions/aksvm.svg)](https://pypi.org/project/aksvm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**aksvm** turns the *associated kernels* of nonparametric smoothing
(Gamma, Beta, Poisson, binomial, discrete triangular, Conway–Maxwell–Poisson)
into valid **positive semi-definite (Mercer) kernels**. You can use them in
support vector machines on data whose variables live on different supports:
positive measurements, proportions in `[0, 1]` and counts.

It works with scikit-learn: `AssociatedKernelSVC` can be used in
`GridSearchCV`, pipelines and cross-validation like any other classifier.

## Why

An associated kernel `K_{x,h}` is a probability distribution indexed by a
target point `x`, whose support matches the data (`[0, ∞)`, `[0, 1]`, `ℕ`, …).
It removes the boundary bias of classical kernel smoothing. Seen as a function
of two points, however, `(x, z) ↦ K_{x,h}(z)` is **neither symmetric nor
positive semi-definite**. Using it directly in an SVM gives a non-convex
problem with no RKHS interpretation.

aksvm uses the **associated probability-product kernel (AK-PPK)** instead:

```
k(x, z) = ∫ K_{x,h}(u)^ρ · K_{z,h}(u)^ρ dμ(u),      k̃(x, z) = k(x, z) / sqrt(k(x, x) · k(z, z))
```

- **Always valid.** It is an inner product of the feature map `x ↦ K_{x,h}^ρ`, so it is symmetric and PSD by construction.
- **Exact and fast.** Gamma, Beta and Gaussian families have closed forms (log-gamma functions). Discrete families are exact finite sums, and a truncated sum stays exactly PSD.
- **Support-aware.** Locally, the kernel is a Gaussian kernel in the Fisher–Rao geometry of the family (≈ `√x` for Gamma and Poisson, ≈ `arcsin √x` for Beta). Unlike "transform + RBF", it keeps a finite length scale at the boundary of the support.
- **Mixed data.** Products of one-dimensional kernels handle vectors that mix several supports. The product kernel is itself the AK-PPK of the multivariate associated kernel.

## Installation

```bash
pip install aksvm
```

Requires Python ≥ 3.9, NumPy, SciPy and scikit-learn.

## Quick start

```python
import numpy as np
from sklearn.model_selection import train_test_split
from aksvm import AssociatedKernelSVC

rng = np.random.default_rng(0)
n = 400
X = np.column_stack([
    rng.gamma(2.0, 1.0, n),   # positive  -> gamma kernel
    rng.beta(0.5, 0.5, n),    # in [0, 1] -> beta kernel
    rng.poisson(4.0, n),      # counts    -> poisson kernel
])
y = (np.log(X[:, 0]) + 4 * (X[:, 1] - 0.5) ** 2 - 0.3 * np.sqrt(X[:, 2]) > 0).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

clf = AssociatedKernelSVC(families="auto", rho=1.0, C=10.0)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

With `families="auto"`, the family of each feature is inferred from its
training values: non-negative integers → `poisson`, values in `[0, 1]` →
`beta`, positive values → `gamma`, anything else → `gaussian`.

### Tuning

`rho` is a dimensionless global sharpness. After the default per-feature
calibration it plays the role of `γ` in an RBF kernel: on standardised
Gaussian features it corresponds to `γ = rho / (2d)`. Tune it together with
`C`:

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    AssociatedKernelSVC(),
    {"rho": [0.25, 0.5, 1, 2, 4], "C": [0.1, 1, 10, 100]},
    cv=3,
)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.score(X_test, y_test))
```

### Explicit families and adaptive smoothing

```python
# one family per feature; per-feature sharpness chosen by kernel-target alignment
clf = AssociatedKernelSVC(families=["gamma", "beta", "cmp"], adapt="kta")
clf.fit(X_train, y_train)
print(clf.feature_multipliers_)   # 0 means the feature was switched off
```

- `adapt="kta"` selects a sharpness multiplier for each feature by coordinate ascent on the centred kernel–target alignment. It acts as automatic relevance determination.
- `adapt="alignf"` learns the weights of an additive (sum) kernel, using the `alignf` algorithm of Cortes, Mohri and Rostamizadeh (2012).

### Using the kernel on its own

```python
from sklearn.svm import SVC
from aksvm import AssociatedKernel

kernel = AssociatedKernel(families=["gamma", "beta", "poisson"], rho=0.5).fit(X_train)
K_train = kernel(X_train)            # (n_train, n_train), PSD, unit diagonal
K_test = kernel(X_test, X_train)     # (n_test, n_train)

svc = SVC(kernel="precomputed", C=10).fit(K_train, y_train)
print(svc.score(K_test, y_test))
```

The Gram matrix can be passed to any method that accepts a precomputed
kernel: kernel ridge, kernel PCA, spectral clustering, Gaussian processes, etc.

## Available families

| Name | Support | Associated kernel `K_{x,h}` | Default `h` |
|------|---------|-----------------------------|-------------|
| `gaussian` | ℝ | `N(x, h²)` (gives the RBF kernel) | 1 |
| `gamma` | `[0, ∞)` | `Gamma(shape = x/h + 1, scale = h)` (Chen, 2000) | 1 |
| `beta` | `[0, 1]` | `Beta(x/h + 1, (1 − x)/h + 1)` (Chen, 1999) | 1 |
| `poisson` | ℕ | `Poisson(x + h)` | 1 |
| `binomial` | ℕ | `Bin(x + 1, (x + h)/(x + 1))`, `0 < h ≤ 1` (Kokonendji & Senga Kiessé, 2011) | 0.5 |
| `triangular` | ℤ | discrete triangular, arm `a` (Kokonendji, Senga Kiessé & Zocchi, 2007) | 1 |
| `cmp` | ℕ | Conway–Maxwell–Poisson with mode `x` (`h < 1` under-, `h > 1` over-dispersed) | 1 |

For the Gamma, Beta and Gaussian families, the normalised kernel depends on
`(h, ρ)` only through `ρ/h` (or `ρ/h²`). You therefore only need to tune `rho`.

## Main parameters of `AssociatedKernelSVC`

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `families` | `"auto"` | family name, list of names (one per feature) or `"auto"` |
| `h` | `None` | bandwidth(s); `None` uses each family's default |
| `rho` | `0.5` | global sharpness (probability-product exponent) |
| `combination` | `"product"` | `"product"` (models interactions) or `"sum"` (additive model) |
| `calibrate` | `True` | per-feature calibration, the analogue of standardisation for RBF |
| `adapt` | `None` | `None`, `"kta"` or `"alignf"` |
| `C` | `1.0` | soft-margin penalty |
| `kernel_type` | `"ppk"` | `"naive"` uses the raw, non-PSD associated kernel (for comparison only) |

Multi-class problems are handled one-versus-one, as in LIBSVM.

## Running the tests

```bash
pip install -e ".[test]"
pytest -q tests
```

The tests check:
- the closed forms against numerical integration;
- the PSD property for every family, with product and sum combinations;
- that the raw associated kernel is indeed *not* PSD;
- the end-to-end classifiers.

## Citation

If you use aksvm in your research, please cite:

> I. Touza and W. E. Wansouwé. *Associated-Kernel Support Vector Machines for
> Nonlinear Classification of Heterogeneous Data: Construction, Theory, and
> Empirical Evaluation.* In preparation, 2026.

GitHub's *Cite this repository* button gives the same reference in BibTeX
and APA format (from `CITATION.cff`).

## Authors

- **Isaac Touza**: University of Maroua; LARI, University of Maroua · ORCID [0009-0003-6288-9458](https://orcid.org/0009-0003-6288-9458)
- **Wanbitching Eric Wansouwé**: ISSEA, Yaoundé; LARI, University of Maroua · ORCID [0009-0005-4055-5689](https://orcid.org/0009-0005-4055-5689)

## License

MIT, see [LICENSE](LICENSE).