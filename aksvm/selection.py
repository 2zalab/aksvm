"""Data-driven tuning of associated kernels by centred kernel-target alignment."""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

__all__ = ["target_matrix", "centered_alignment", "select_feature_rho", "alignf_weights"]


def _center(K):
    return K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()


def target_matrix(y):
    """Ideal kernel ``Y Y^T`` from one-hot (+1/-1 for two classes) labels."""
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) == 2:
        v = np.where(y == classes[1], 1.0, -1.0)
        return np.outer(v, v)
    Y = (y[:, None] == classes[None, :]).astype(float)
    return Y @ Y.T


def centered_alignment(K, y=None, T=None):
    """Centred alignment ``<K_c, T_c>_F / (||K_c||_F ||T_c||_F)`` (Cortes et al., 2012)."""
    T = target_matrix(y) if T is None else T
    Kc, Tc = _center(K), _center(T)
    den = np.linalg.norm(Kc) * np.linalg.norm(Tc)
    return float(np.sum(Kc * Tc) / den) if den > 0 else 0.0


def select_feature_rho(kernel, X, y, multipliers=None, n_sweeps=2, max_samples=400,
                       random_state=0):
    """Per-feature sharpness ``rho_j = rho * m_j`` by coordinate ascent on alignment.

    For the continuous families only ``rho_j / h_j`` enters the normalised
    kernel, so this is equivalent to a per-feature bandwidth selection
    (automatic relevance determination). ``kernel`` must already be fitted;
    the selected exponents are written into a refitted copy and returned.
    """
    X = np.asarray(X, float)
    y = np.asarray(y)
    if multipliers is None:
        multipliers = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    rng = np.random.default_rng(random_state)
    if len(y) > max_samples:
        idx = rng.choice(len(y), max_samples, replace=False)
        X, y = X[idx], y[idx]
    T = target_matrix(y)
    logs = kernel.feature_log_kernels(X)  # computed at the base exponents
    base = np.asarray(kernel.rho_, float)
    # For the normalised continuous families log k is (to first order) linear
    # in rho; we recompute exactly for every candidate to stay correct for
    # discrete families too.
    d = len(logs)
    cache = {}

    def logk(j, m):
        if m == 0.0:
            return np.zeros_like(logs[j])
        key = (j, m)
        if key not in cache:
            f = kernel.families_[j]
            Xt = kernel._transform(X)[:, j]
            cache[key] = f.log_ppk(Xt, Xt, kernel.h_[j], base[j] * m, kernel.normalize)
        return cache[key]

    m = np.ones(d)
    current = [logk(j, 1.0) for j in range(d)]
    total = np.sum(current, axis=0)
    best = centered_alignment(np.exp(total), T=T)
    for _ in range(n_sweeps):
        improved = False
        for j in range(d):
            rest = total - current[j]
            for cand in multipliers:
                if cand == m[j]:
                    continue
                Lj = logk(j, float(cand))
                a = centered_alignment(np.exp(rest + Lj), T=T)
                if a > best + 1e-6:
                    best, m[j], current[j] = a, float(cand), Lj
                    total = rest + Lj
                    rest = total - current[j]
                    improved = True
        if not improved:
            break
    return m, best


def alignf_weights(log_kernels, y):
    """Non-negative combination weights maximising centred alignment.

    Implements the ``alignf`` algorithm of Cortes, Mohri & Rostamizadeh (2012):
    ``min_v v^T M v - 2 v^T a`` subject to ``v >= 0``, with
    ``M_kl = <K_k^c, K_l^c>_F`` and ``a_k = <K_k^c, Y Y^T>_F``.
    """
    T = target_matrix(y)
    Kc = [_center(np.exp(L)) for L in log_kernels]
    a = np.array([np.sum(K * T) for K in Kc])
    M = np.array([[np.sum(Ki * Kj) for Kj in Kc] for Ki in Kc])
    M = M + 1e-10 * np.trace(M) * np.eye(len(Kc))
    R = np.linalg.cholesky(M).T  # M = R^T R
    v, _ = nnls(R, np.linalg.solve(R.T, a))
    if v.sum() <= 0:
        v = np.ones(len(Kc))
    return v / v.sum()
