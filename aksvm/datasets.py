"""Synthetic and bundled benchmark data sets with heterogeneous supports."""

from __future__ import annotations

import numpy as np
from sklearn import datasets as skd

__all__ = ["make_positive_ring", "make_unit_band", "make_count_shells", "make_mixed",
           "load_benchmark", "BENCHMARKS"]


def _flip(y, noise, rng):
    flip = rng.random(len(y)) < noise
    return np.where(flip, 1 - y, y)


def make_positive_ring(n=400, noise=0.05, random_state=0):
    """Two skewed positive features; class = inside a ring in log-scale."""
    rng = np.random.default_rng(random_state)
    X = rng.lognormal(mean=0.0, sigma=1.0, size=(n, 2))
    r = np.sqrt(np.log(X[:, 0]) ** 2 + np.log(X[:, 1]) ** 2)
    y = (r < 1.0).astype(int)
    return X, _flip(y, noise, rng), ["gamma", "gamma"]


def make_unit_band(n=400, noise=0.05, random_state=0):
    """Proportions concentrated near 0 and 1; class = diagonal band in logit scale."""
    rng = np.random.default_rng(random_state)
    X = rng.beta(0.35, 0.35, size=(n, 2))
    X = np.clip(X, 1e-6, 1 - 1e-6)
    logit = np.log(X / (1 - X))
    y = (np.abs(logit[:, 0] - logit[:, 1]) < 2.0).astype(int)
    return X, _flip(y, noise, rng), ["beta", "beta"]


def make_count_shells(n=400, noise=0.0, random_state=0):
    """Poisson counts whose square-root rates lie on two concentric arcs."""
    rng = np.random.default_rng(random_state)
    y = rng.integers(0, 2, n)
    radius = np.where(y == 1, 5.0, 3.5)
    t = rng.uniform(0.05, np.pi / 2 - 0.05, n)
    rate = np.column_stack([(radius * np.cos(t)) ** 2, (radius * np.sin(t)) ** 2])
    X = rng.poisson(rate + 0.2)
    return X.astype(float), _flip(y, noise, rng), ["poisson", "poisson"]


def make_mixed(n=500, noise=0.05, random_state=0, n_irrelevant=2):
    """Mixed positive / proportion / count features plus irrelevant ones."""
    rng = np.random.default_rng(random_state)
    x1 = rng.gamma(1.2, 1.0, n)
    x2 = rng.beta(0.5, 0.5, n)
    x3 = rng.poisson(rng.gamma(2.0, 3.0, n)).astype(float)
    score = np.sin(1.5 * np.log(x1 + 1e-3)) + 8 * (x2 - 0.5) ** 2 - 0.6 * np.sqrt(x3) + 1.0
    y = (score > np.median(score)).astype(int)
    cols, fams = [x1, x2, x3], ["gamma", "beta", "poisson"]
    for k in range(n_irrelevant):
        if k % 2 == 0:
            cols.append(rng.gamma(1.0, 2.0, n)); fams.append("gamma")
        else:
            cols.append(rng.poisson(5.0, n).astype(float)); fams.append("poisson")
    return np.column_stack(cols), _flip(y, noise, rng), fams


def _sk(loader):
    def f(random_state=0):
        d = loader()
        X = d.data.astype(float)
        fams = ["auto"] * X.shape[1]
        return X, d.target, fams
    return f


def _digits(random_state=0):
    d = skd.load_digits()
    keep = np.isin(d.target, [3, 5, 8, 9])  # the mutually confusable digits
    X = d.data[keep].astype(float)
    return X, d.target[keep], ["poisson"] * X.shape[1]


BENCHMARKS = {
    "positive-ring": make_positive_ring,
    "unit-band": make_unit_band,
    "count-shells": make_count_shells,
    "mixed": make_mixed,
    "breast-cancer": _sk(skd.load_breast_cancer),
    "wine": _sk(skd.load_wine),
    "iris": _sk(skd.load_iris),
    "digits-3589": _digits,
}


def load_benchmark(name, random_state=0):
    return BENCHMARKS[name](random_state=random_state)
