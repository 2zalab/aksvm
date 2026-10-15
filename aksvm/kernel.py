"""Multivariate associated probability-product kernels for mixed data."""

from __future__ import annotations

import numpy as np

from .families import AssociatedFamily, get_family

__all__ = ["infer_family", "AssociatedKernel"]


def infer_family(column) -> str:
    """Guess an associated-kernel family from the support of a feature."""
    x = np.asarray(column, dtype=float)
    if np.all(np.abs(x - np.round(x)) < 1e-9) and np.all(x >= 0):
        return "poisson"
    if np.all((x >= 0) & (x <= 1)):
        return "beta"
    if np.all(x >= 0):
        return "gamma"
    return "gaussian"


def _per_feature(value, d, name):
    if value is None or np.isscalar(value):
        return [value] * d
    value = list(value)
    if len(value) != d:
        raise ValueError(f"{name} has length {len(value)}, expected {d}")
    return value


class AssociatedKernel:
    """Product (or weighted sum) of one-dimensional associated PPKs.

    Parameters
    ----------
    families : "auto", str, or sequence
        Family of each feature (``"gamma"``, ``"beta"``, ``"poisson"``,
        ``"binomial"``, ``"triangular"``, ``"cmp"``, ``"gaussian"`` or an
        :class:`AssociatedFamily` instance). ``"auto"`` infers it from the
        training support.
    h : float or sequence, optional
        Bandwidths ``h_j``; ``None`` uses the family default.
    rho : float or sequence
        Probability-product exponents ``rho_j`` (global sharpness).
    combination : {"product", "sum"}
        ``K = prod_j k_j`` or ``K = sum_j w_j k_j / sum_j w_j``.
    weights : sequence, optional
        Non-negative weights for ``combination="sum"``.
    normalize : bool
        Use the cosine-normalised PPK (``k(x, x) = 1``).
    scale_positive : bool
        Divide gamma features by their training mean and standardise gaussian
        features, so that one bandwidth suits all features. This is a scale
        change only and preserves each support.
    calibrate : bool
        Multiply ``rho_j`` by ``c_j = 1 / (d * mean_{i != i'} -log k_j(x_i, x_i'; rho=1))``
        so that every feature contributes on average ``rho / d`` to
        ``-log K`` on the training sample (the analogue of standardising the
        inputs of an RBF kernel). For ``combination="sum"`` the factor ``d`` is
        dropped. Only used with ``normalize=True``.
    max_calibration_samples : int
        Size of the random sub-sample used for calibration.
    """

    def __init__(self, families="auto", h=None, rho=0.5, combination="product",
                 weights=None, normalize=True, scale_positive=True, calibrate=False,
                 max_calibration_samples=300):
        self.families = families
        self.h = h
        self.rho = rho
        self.combination = combination
        self.weights = weights
        self.normalize = normalize
        self.scale_positive = scale_positive
        self.calibrate = calibrate
        self.max_calibration_samples = max_calibration_samples

    # --------------------------------------------------------------- fitting
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        d = X.shape[1]
        if isinstance(self.families, str) and self.families == "auto":
            specs = [infer_family(X[:, j]) for j in range(d)]
        elif isinstance(self.families, (str, AssociatedFamily)):
            specs = [self.families] * d
        else:
            specs = _per_feature(self.families, d, "families")
        specs = [infer_family(X[:, j]) if s == "auto" else s for j, s in enumerate(specs)]
        self.families_ = [get_family(s) for s in specs]
        hs = _per_feature(self.h, d, "h")
        self.h_ = np.array([f.default_h if hj is None else float(hj)
                            for f, hj in zip(self.families_, hs)])
        self.rho_ = np.array(_per_feature(self.rho, d, "rho"), dtype=float)
        if np.any(self.rho_ <= 0):
            raise ValueError("rho must be > 0")
        self.shift_ = np.zeros(d)
        self.scale_ = np.ones(d)
        for j, f in enumerate(self.families_):
            f.check(X[:, j])
            f.check_h(self.h_[j])
            if self.scale_positive and f.name == "gamma":
                m = X[:, j].mean()
                self.scale_[j] = m if m > 0 else 1.0
            elif self.scale_positive and f.name == "gaussian":
                s = X[:, j].std()
                self.shift_[j] = X[:, j].mean()
                self.scale_[j] = s if s > 0 else 1.0
        if self.combination not in ("product", "sum"):
            raise ValueError("combination must be 'product' or 'sum'")
        w = np.ones(d) if self.weights is None else np.asarray(self.weights, float)
        if w.shape != (d,) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be d non-negative numbers with positive sum")
        self.weights_ = w / w.sum()
        self.n_features_in_ = d
        self.calibration_ = np.ones(d)
        if self.calibrate and self.normalize:
            self.calibration_ = self._calibration(X)
            self.rho_ = self.rho_ * self.calibration_
        return self

    def _calibration(self, X):
        rng = np.random.default_rng(0)
        if len(X) > self.max_calibration_samples:
            X = X[rng.choice(len(X), self.max_calibration_samples, replace=False)]
        Xt = self._transform(X)
        d = X.shape[1] if self.combination == "product" else 1
        off = ~np.eye(len(X), dtype=bool)
        c = np.empty(X.shape[1])
        for j, f in enumerate(self.families_):
            m = -f.log_ppk(Xt[:, j], Xt[:, j], self.h_[j], 1.0, True)[off].mean()
            c[j] = 1.0 / (d * m) if m > 1e-12 else 1.0 / d
        return c

    def _transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong shape")
        Xt = (X - self.shift_) / self.scale_
        for j, f in enumerate(self.families_):
            if f.name == "gamma":  # out-of-support test values are clipped
                Xt[:, j] = np.clip(Xt[:, j], 0, None)
            elif f.name == "beta":
                Xt[:, j] = np.clip(Xt[:, j], 0, 1)
            elif f.discrete:
                Xt[:, j] = np.round(Xt[:, j])
                if f.support == "count":
                    Xt[:, j] = np.clip(Xt[:, j], 0, None)
        return Xt

    # ------------------------------------------------------------ evaluation
    def feature_log_kernels(self, X, Z=None):
        """List of per-feature log-kernel matrices ``log k_j(X_j, Z_j)``."""
        Xt = self._transform(X)
        Zt = Xt if Z is None else self._transform(Z)
        return [f.log_ppk(Xt[:, j], Zt[:, j], self.h_[j], self.rho_[j], self.normalize)
                for j, f in enumerate(self.families_)]

    def __call__(self, X, Z=None):
        logs = self.feature_log_kernels(X, Z)
        if self.combination == "product":
            return np.exp(np.sum(logs, axis=0))
        return np.sum([w * np.exp(L) for w, L in zip(self.weights_, logs)], axis=0)

    def naive_log_gram(self, X, Z=None):
        """Log of the product of raw associated kernels ``prod_j K_{x_j,h_j}(z_j)``.

        This is the "plug-in" kernel one would get by reusing the density
        estimation kernel directly; it is in general neither symmetric nor PSD.
        """
        Xt = self._transform(X)
        Zt = Xt if Z is None else self._transform(Z)
        return np.sum([f.log_density(Xt[:, j], Zt[:, j], self.h_[j])
                       for j, f in enumerate(self.families_)], axis=0)

    def naive_gram(self, X, Z=None):
        return np.exp(self.naive_log_gram(X, Z))

    def naive_self_log(self, X):
        """``log prod_j K_{x_j,h_j}(x_j)`` for every row of ``X``."""
        Xt = self._transform(X)
        return np.sum([np.diag(f.log_density(Xt[:, j], Xt[:, j], self.h_[j]))
                       for j, f in enumerate(self.families_)], axis=0)

    def naive_symmetric_gram(self, X, Z=None):
        """Symmetrised, cosine-normalised naive kernel
        ``(A(x,z) + A(z,x)) / (2 sqrt(A(x,x) A(z,z)))`` (not PSD in general)."""
        Z = X if Z is None else Z
        dx, dz = self.naive_self_log(X), self.naive_self_log(Z)
        off = 0.5 * dx[:, None] + 0.5 * dz[None, :]
        A = np.exp(self.naive_log_gram(X, Z) - off)
        B = np.exp(self.naive_log_gram(Z, X).T - off)
        return 0.5 * (A + B)
