import numpy as np
import pytest
from scipy import integrate, stats

from aksvm import AssociatedKernel, AssociatedKernelSVC, get_family
from aksvm.selection import centered_alignment

RNG = np.random.default_rng(0)


def numeric_ppk(fam, x, z, h, rho, lo, hi):
    f = lambda u: np.exp(rho * (fam.log_density([x], [u], h)[0, 0]
                                + fam.log_density([z], [u], h)[0, 0]))
    return integrate.quad(f, lo, hi, limit=200, points=[x, z])[0]


@pytest.mark.parametrize("rho", [0.5, 1.0, 1.7])
@pytest.mark.parametrize("x,z,h", [(0.3, 1.2, 0.4), (0.0, 2.0, 0.7), (2.5, 2.4, 0.2)])
def test_gamma_closed_form(x, z, h, rho):
    fam = get_family("gamma")
    num = numeric_ppk(fam, x, z, h, rho, 0, 60)
    assert np.isclose(np.exp(fam.log_ppk([x], [z], h, rho, normalize=False))[0, 0], num, rtol=1e-6)


@pytest.mark.parametrize("rho", [0.5, 1.0, 1.7])
@pytest.mark.parametrize("x,z,h", [(0.1, 0.8, 0.2), (0.0, 1.0, 0.5), (0.45, 0.5, 0.05)])
def test_beta_closed_form(x, z, h, rho):
    fam = get_family("beta")
    num = numeric_ppk(fam, x, z, h, rho, 0, 1)
    assert np.isclose(np.exp(fam.log_ppk([x], [z], h, rho, normalize=False))[0, 0], num, rtol=1e-6)


def test_gaussian_is_rbf():
    fam = get_family("gaussian")
    x, z = RNG.normal(size=5), RNG.normal(size=4)
    K = np.exp(fam.log_ppk(x, z, 0.7, 0.5))
    assert np.allclose(K, np.exp(-0.5 * (x[:, None] - z[None, :]) ** 2 / (4 * 0.49)))
    num = numeric_ppk(fam, 0.2, 1.0, 0.7, 1.3, -20, 20)
    assert np.isclose(np.exp(fam.log_ppk([0.2], [1.0], 0.7, 1.3, normalize=False))[0, 0], num)


def test_poisson_half_closed_form():
    fam = get_family("poisson")
    x = np.arange(0, 30)
    h = 0.6
    K = np.exp(fam.log_ppk(x, x, h, 0.5))
    lam = x + h
    ref = np.exp(-0.5 * (np.sqrt(lam)[:, None] - np.sqrt(lam)[None, :]) ** 2)
    assert np.allclose(K, ref, atol=1e-10)


@pytest.mark.parametrize("name", ["poisson", "binomial", "triangular", "cmp"])
def test_discrete_pmfs_sum_to_one(name):
    fam = get_family(name)
    x = np.array([0, 1, 4, 11])
    h = 0.4 if name == "binomial" else 1.3
    u = fam.grid(x.astype(float), h)
    P = np.exp(fam.log_pmf(x, u, h))
    assert np.allclose(P.sum(axis=1), 1, atol=1e-9)
    # pmf of the binomial kernel against scipy
    if name == "binomial":
        ref = stats.binom.pmf(u[None, :], x[:, None] + 1, (x[:, None] + h) / (x[:, None] + 1))
        assert np.allclose(P, ref)


def test_cmp_mode_is_target():
    fam = get_family("cmp")
    x = np.arange(0, 15)
    for h in (0.3, 1.0, 2.5):
        u = fam.grid(x.astype(float), h)
        P = fam.log_pmf(x, u, h)
        assert np.array_equal(u[P.argmax(axis=1)], x)


@pytest.mark.parametrize("fams", [
    ["gamma", "beta", "poisson"],
    ["binomial", "triangular", "cmp"],
    ["gaussian", "gamma", "cmp"],
])
@pytest.mark.parametrize("combination", ["product", "sum"])
def test_gram_is_psd(fams, combination):
    n = 120
    cols = []
    for f in fams:
        if f == "gamma":
            cols.append(RNG.gamma(1.5, 2.0, n))
        elif f == "beta":
            cols.append(RNG.beta(0.6, 0.6, n))
        elif f == "gaussian":
            cols.append(RNG.normal(size=n))
        else:
            cols.append(RNG.poisson(3.0, n))
    X = np.column_stack(cols)
    h = [0.5 if f == "binomial" else 0.3 for f in fams]
    for normalize in (True, False):
        k = AssociatedKernel(fams, h=h, rho=0.8, combination=combination,
                             normalize=normalize).fit(X)
        K = k(X)
        assert np.allclose(K, K.T)
        ev = np.linalg.eigvalsh(K)
        assert ev.min() >= -1e-8 * max(1.0, ev.max())
        if normalize:
            assert np.allclose(np.diag(K), 1.0)


def test_naive_kernel_is_not_psd():
    X = RNG.gamma(0.8, 1.0, size=(150, 2))
    k = AssociatedKernel("gamma", h=0.1).fit(X)
    A = k.naive_gram(X)
    assert not np.allclose(A, A.T)
    ev = np.linalg.eigvalsh(0.5 * (A + A.T))
    assert ev.min() < -1e-6 * ev.max()


def test_continuous_depends_on_rho_over_h():
    x = RNG.gamma(2.0, 1.0, 20)
    fam = get_family("gamma")
    assert np.allclose(fam.log_ppk(x, x, 0.4, 0.5), fam.log_ppk(x, x, 0.8, 1.0))
    xb = RNG.uniform(size=20)
    fam = get_family("beta")
    assert np.allclose(fam.log_ppk(xb, xb, 0.1, 0.5), fam.log_ppk(xb, xb, 0.2, 1.0))


def test_svc_end_to_end():
    n = 300
    X = np.column_stack([RNG.gamma(2, 1, n), RNG.beta(2, 2, n), RNG.poisson(4, n)])
    y = (np.log(X[:, 0]) + 3 * (X[:, 1] - 0.5) ** 2 * 4 - 0.2 * X[:, 2] > 0).astype(int)
    for adapt in (None, "kta", "alignf"):
        clf = AssociatedKernelSVC(rho=0.5, C=10, adapt=adapt).fit(X[:200], y[:200])
        acc = (clf.predict(X[200:]) == y[200:]).mean()
        assert acc > 0.75, (adapt, acc)
        assert clf.decision_function(X[200:]).shape == (100,)
    clf = AssociatedKernelSVC(rho=0.5, C=10, kernel_type="naive", h=0.3).fit(X[:200], y[:200])
    assert clf.predict(X[200:]).shape == (100,)


def test_alignment_bounds():
    X = RNG.normal(size=(50, 2))
    y = (X[:, 0] > 0).astype(int)
    K = AssociatedKernel("gaussian").fit(X)(X)
    a = centered_alignment(K, y)
    assert -1 <= a <= 1 and a > 0.2
