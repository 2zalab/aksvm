"""Associated-kernel families and their probability-product (PSD) kernels.

An *associated kernel* is a family of probability densities (or mass
functions) ``u -> K_{x,h}(u)`` indexed by a target point ``x`` and a smoothing
parameter ``h``; its support matches the support of the data (``[0, 1]``,
``[0, inf)``, ``N``, ...). Used as ``(x, z) -> K_{x,h}(z)`` such a function is
neither symmetric nor positive semi-definite, so it cannot be plugged into a
support vector machine as such.

Every family below therefore exposes the *associated probability-product
kernel*

    k_{h,rho}(x, z) = < K_{x,h}^rho , K_{z,h}^rho >_{L^2(mu)}
                    = int K_{x,h}(u)^rho K_{z,h}(u)^rho dmu(u),

which is symmetric and positive semi-definite by construction (it is a Gram
matrix of the feature map ``x -> K_{x,h}^rho``), together with its cosine
normalisation ``k(x,z) / sqrt(k(x,x) k(z,z))``. All quantities are computed in
log-space for numerical stability.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp

__all__ = [
    "AssociatedFamily",
    "GaussianFamily",
    "GammaFamily",
    "BetaFamily",
    "PoissonFamily",
    "BinomialFamily",
    "DiscreteTriangularFamily",
    "CMPFamily",
    "FAMILIES",
    "get_family",
]


def _as_1d(x):
    return np.asarray(x, dtype=float).ravel()


class AssociatedFamily:
    """Base class for an associated-kernel family on a one-dimensional support."""

    name = "base"
    support = "real"
    discrete = False
    default_h = 1.0

    # ------------------------------------------------------------------ API
    def check(self, x):
        """Raise ``ValueError`` if some value of ``x`` is outside the support."""

    def check_h(self, h):
        if not np.isfinite(h) or h <= 0:
            raise ValueError(f"{self.name}: bandwidth must be > 0, got {h}")

    def log_density(self, x, u, h):
        """Matrix ``log K_{x_i,h}(u_j)`` of the associated kernel."""
        raise NotImplementedError

    def log_ppk(self, x, z, h, rho=0.5, normalize=True):
        """Matrix ``log k_{h,rho}(x_i, z_j)`` of the probability-product kernel."""
        raise NotImplementedError

    def naive_kernel(self, x, z, h):
        """Non-symmetric matrix ``K_{x_i,h}(z_j)`` (the raw associated kernel)."""
        x, z = _as_1d(x), _as_1d(z)
        return np.exp(self.log_density(x, z, h))

    def __repr__(self):
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------- continuous
class GaussianFamily(AssociatedFamily):
    """Classical symmetric kernel ``K_{x,h} = N(x, h^2)`` on the real line.

    Its normalised probability-product kernel is the Gaussian RBF kernel
    ``exp(-rho (x - z)^2 / (4 h^2))``.
    """

    name = "gaussian"
    support = "real"

    def log_density(self, x, u, h):
        x, u = _as_1d(x)[:, None], _as_1d(u)[None, :]
        return -0.5 * np.log(2 * np.pi * h**2) - (u - x) ** 2 / (2 * h**2)

    def log_ppk(self, x, z, h, rho=0.5, normalize=True):
        self.check_h(h)
        x, z = _as_1d(x)[:, None], _as_1d(z)[None, :]
        out = -rho * (x - z) ** 2 / (4 * h**2)
        if not normalize:
            out = out + 0.5 * (1 - 2 * rho) * np.log(2 * np.pi * h**2) - 0.5 * np.log(2 * rho)
        return out


class GammaFamily(AssociatedFamily):
    """Gamma associated kernel of Chen (2000) on ``[0, inf)``.

    ``K_{x,h} = Gamma(shape = x/h + 1, scale = h)``. Normalised PPK:

        log k(x, z) = lnG(s_xz) - (lnG(s_xx) + lnG(s_zz)) / 2,
        s_xz = rho (x + z) / h + 1.
    """

    name = "gamma"
    support = "positive"

    def check(self, x):
        if np.any(_as_1d(x) < 0):
            raise ValueError("gamma family requires non-negative values")

    def log_density(self, x, u, h):
        x, u = _as_1d(x)[:, None], _as_1d(u)[None, :]
        a = x / h + 1.0
        with np.errstate(divide="ignore"):
            lu = np.where(u > 0, np.log(np.where(u > 0, u, 1.0)), -np.inf)
        out = (a - 1) * lu - u / h - gammaln(a) - a * np.log(h)
        # u = 0 with a = 1 (x = 0) has density 1/h
        out = np.where((u == 0) & (a == 1), -np.log(h), out)
        return out

    def log_ppk(self, x, z, h, rho=0.5, normalize=True):
        self.check_h(h)
        x, z = _as_1d(x), _as_1d(z)
        s = rho * (x[:, None] + z[None, :]) / h + 1.0
        if normalize:
            sx = gammaln(2 * rho * x / h + 1.0)
            sz = gammaln(2 * rho * z / h + 1.0)
            return gammaln(s) - 0.5 * sx[:, None] - 0.5 * sz[None, :]
        ax, az = x / h + 1.0, z / h + 1.0
        return (
            -rho * (gammaln(ax)[:, None] + gammaln(az)[None, :])
            - rho * (ax[:, None] + az[None, :]) * np.log(h)
            + gammaln(s)
            + s * np.log(h / (2 * rho))
        )


def _betaln(p, q):
    return gammaln(p) + gammaln(q) - gammaln(p + q)


class BetaFamily(AssociatedFamily):
    """Beta associated kernel of Chen (1999) on ``[0, 1]``.

    ``K_{x,h} = Beta(x/h + 1, (1 - x)/h + 1)``. Normalised PPK:

        log k(x, z) = [lnG(p_xz) + lnG(q_xz)] - half the same at (x,x), (z,z),
        p_xz = rho (x + z)/h + 1,  q_xz = rho (2 - x - z)/h + 1.
    """

    name = "beta"
    support = "unit"

    def check(self, x):
        x = _as_1d(x)
        if np.any((x < 0) | (x > 1)):
            raise ValueError("beta family requires values in [0, 1]")

    def log_density(self, x, u, h):
        x, u = _as_1d(x)[:, None], _as_1d(u)[None, :]
        a, b = x / h + 1.0, (1 - x) / h + 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            lu = np.log(np.clip(u, 1e-300, None))
            l1u = np.log(np.clip(1 - u, 1e-300, None))
        return (a - 1) * lu + (b - 1) * l1u - _betaln(a, b)

    def log_ppk(self, x, z, h, rho=0.5, normalize=True):
        self.check_h(h)
        x, z = _as_1d(x), _as_1d(z)
        p = rho * (x[:, None] + z[None, :]) / h + 1.0
        q = rho * (2.0 - x[:, None] - z[None, :]) / h + 1.0
        if normalize:
            # p + q = 2 rho / h + 2 is constant, so the lnG(p+q) terms cancel.
            def diag(v):
                return gammaln(2 * rho * v / h + 1.0) + gammaln(2 * rho * (1 - v) / h + 1.0)

            return gammaln(p) + gammaln(q) - 0.5 * diag(x)[:, None] - 0.5 * diag(z)[None, :]
        ax, bx = x / h + 1.0, (1 - x) / h + 1.0
        az, bz = z / h + 1.0, (1 - z) / h + 1.0
        return _betaln(p, q) - rho * (_betaln(ax, bx)[:, None] + _betaln(az, bz)[None, :])


# ------------------------------------------------------------------ discrete
class DiscreteFamily(AssociatedFamily):
    """Discrete associated kernel; the PPK is an exact finite sum on a grid.

    The infinite sum over the support is truncated to a grid that carries all
    but a negligible amount of the mass of every ``K_{x,h}`` involved. A
    truncated sum is itself a Gram matrix of truncated feature vectors, so
    positive semi-definiteness holds *exactly* whatever the truncation.
    """

    discrete = True
    support = "count"

    def check(self, x):
        x = _as_1d(x)
        if np.any(x < 0) or np.any(np.abs(x - np.round(x)) > 1e-9):
            raise ValueError(f"{self.name} family requires non-negative integers")

    def grid(self, xmax, h):
        raise NotImplementedError

    def log_pmf(self, x, u, h):
        raise NotImplementedError

    def log_density(self, x, u, h):
        x, u = _as_1d(x), _as_1d(u)
        return self.log_pmf(x, u, h)

    def log_ppk(self, x, z, h, rho=0.5, normalize=True):
        self.check_h(h)
        x, z = np.round(_as_1d(x)), np.round(_as_1d(z))
        ux, ix = np.unique(x, return_inverse=True)
        uz, iz = np.unique(z, return_inverse=True)
        allv = np.union1d(ux, uz)
        u = self.grid(allv, h)
        L = rho * self.log_pmf(allv, u, h)  # (n_unique, n_grid)
        # log of sum_u exp(L_a + L_b), computed by blocks to limit memory
        G = np.empty((len(allv), len(allv)))
        for a in range(len(allv)):
            G[a] = logsumexp(L[a][None, :] + L, axis=1)
        pos_x = np.searchsorted(allv, ux)[ix]
        pos_z = np.searchsorted(allv, uz)[iz]
        out = G[np.ix_(pos_x, pos_z)]
        if normalize:
            d = np.diag(G)
            out = out - 0.5 * d[pos_x][:, None] - 0.5 * d[pos_z][None, :]
        return out


def _poisson_logpmf(lam, u):
    lam = lam[:, None]
    u = u[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = u * np.log(lam) - lam - gammaln(u + 1)
    out = np.where(u < 0, -np.inf, out)
    return np.where((lam == 0) & (u == 0), 0.0, out)


class PoissonFamily(DiscreteFamily):
    """Poisson associated kernel ``K_{x,h} = Poisson(x + h)`` on ``N``.

    For ``rho = 1/2`` the normalised PPK has the closed form
    ``exp(-(sqrt(x + h) - sqrt(z + h))^2 / 2)``.
    """

    name = "poisson"
    default_h = 1.0

    def grid(self, values, h):
        lam = values.max() + h
        top = int(np.ceil(lam + 12 * np.sqrt(lam) + 25))
        return np.arange(0, top + 1, dtype=float)

    def log_pmf(self, x, u, h):
        return _poisson_logpmf(_as_1d(x) + h, _as_1d(u))


class BinomialFamily(DiscreteFamily):
    """Binomial associated kernel ``Bin(x + 1, (x + h)/(x + 1))``, ``0 < h <= 1``
    (Kokonendji & Senga Kiessé, 2011)."""

    name = "binomial"
    default_h = 0.5

    def check_h(self, h):
        if not (0 < h <= 1):
            raise ValueError(f"binomial family requires 0 < h <= 1, got {h}")

    def grid(self, values, h):
        return np.arange(0, int(values.max()) + 2, dtype=float)

    def log_pmf(self, x, u, h):
        x, u = _as_1d(x)[:, None], _as_1d(u)[None, :]
        n = x + 1
        p = (x + h) / (x + 1)
        inside = (u >= 0) & (u <= n)
        uu = np.where(inside, u, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.where(uu > 0, uu * np.log(p), 0.0)
            lq = np.where(n - uu > 0, (n - uu) * np.log(1 - p), 0.0)
        out = gammaln(n + 1) - gammaln(uu + 1) - gammaln(n - uu + 1) + lp + lq
        return np.where(inside, out, -np.inf)


class DiscreteTriangularFamily(DiscreteFamily):
    """General discrete triangular kernel of Kokonendji, Senga Kiessé & Zocchi
    (2007): ``P(u) proportional to (a+1)^h - |u - x|^h`` for ``|u - x| <= a``.

    The support ``{x - a, ..., x + a}`` is taken on ``Z``; this is harmless for
    the PPK, which only needs an inner product.
    """

    name = "triangular"
    default_h = 1.0

    def __init__(self, arm: int = 1):
        if int(arm) < 1:
            raise ValueError("arm must be a positive integer")
        self.arm = int(arm)

    def check(self, x):
        x = _as_1d(x)
        if np.any(np.abs(x - np.round(x)) > 1e-9):
            raise ValueError("triangular family requires integer values")

    def grid(self, values, h):
        return np.arange(values.min() - self.arm, values.max() + self.arm + 1, dtype=float)

    def log_pmf(self, x, u, h):
        a = self.arm
        x, u = _as_1d(x)[:, None], _as_1d(u)[None, :]
        k = np.arange(1, a + 1, dtype=float)
        norm = (2 * a + 1) * (a + 1) ** h - 2 * np.sum(k**h)
        d = np.abs(u - x)
        w = (a + 1) ** h - np.where(d <= a, d, 0.0) ** h
        with np.errstate(divide="ignore"):
            return np.where(d <= a, np.log(w) - np.log(norm), -np.inf)

    def __repr__(self):
        return f"DiscreteTriangularFamily(arm={self.arm})"


class CMPFamily(DiscreteFamily):
    """Conway-Maxwell-Poisson associated kernel on ``N``.

    ``K_{x,h} = CMP(lambda = (x + 1/2)^{1/h}, nu = 1/h)``, with
    ``P(u) proportional to lambda^u / (u!)^nu``. The ratio
    ``P(u)/P(u-1) = ((x + 1/2)/u)^nu`` shows that the unique mode is ``x``;
    ``h < 1`` gives under-dispersion, ``h > 1`` over-dispersion, and ``h = 1``
    is ``Poisson(x + 1/2)``.
    """

    name = "cmp"
    default_h = 1.0

    def grid(self, values, h):
        m = values.max() + 0.5
        top = int(np.ceil(m + 12 * np.sqrt(max(h, 1.0) * m) + 25 * max(h, 1.0)))
        return np.arange(0, top + 1, dtype=float)

    def log_pmf(self, x, u, h):
        x, u = _as_1d(x), _as_1d(u)
        nu = 1.0 / h
        grid = np.arange(0, max(u.max(), self.grid(x, h).max()) + 1, dtype=float)
        logw = nu * (grid[None, :] * np.log(x[:, None] + 0.5) - gammaln(grid + 1)[None, :])
        logZ = logsumexp(logw, axis=1)
        out = nu * (u[None, :] * np.log(x[:, None] + 0.5) - gammaln(np.clip(u, 0, None) + 1)[None, :])
        out = out - logZ[:, None]
        return np.where(u[None, :] < 0, -np.inf, out)


FAMILIES = {
    "gaussian": GaussianFamily,
    "gamma": GammaFamily,
    "beta": BetaFamily,
    "poisson": PoissonFamily,
    "binomial": BinomialFamily,
    "triangular": DiscreteTriangularFamily,
    "cmp": CMPFamily,
}


def get_family(spec) -> AssociatedFamily:
    """Return a family instance from a name or an instance."""
    if isinstance(spec, AssociatedFamily):
        return spec
    try:
        return FAMILIES[str(spec).lower()]()
    except KeyError as exc:
        raise ValueError(f"unknown family {spec!r}; choose from {sorted(FAMILIES)}") from exc
