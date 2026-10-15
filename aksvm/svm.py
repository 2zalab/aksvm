"""Support vector classifier with an associated probability-product kernel."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted

from .kernel import AssociatedKernel
from .selection import alignf_weights, select_feature_rho

__all__ = ["AssociatedKernelSVC"]


class AssociatedKernelSVC(ClassifierMixin, BaseEstimator):
    """Soft-margin SVM with the associated PPK of :class:`AssociatedKernel`.

    Parameters
    ----------
    families, h, rho, combination, normalize : see :class:`AssociatedKernel`.
    C : float
        Soft-margin penalty.
    adapt : {None, "kta", "alignf"}
        ``"kta"`` selects per-feature exponents ``rho_j`` by centred
        kernel-target alignment (product kernel); ``"alignf"`` learns the
        weights of a sum combination (Cortes et al., 2012).
    kernel_type : {"ppk", "naive"}
        ``"naive"`` uses the symmetrised raw associated kernel
        ``(K_x(z) + K_z(x)) / 2``, which is not PSD in general (for study only).
    calibrate : bool
        Per-feature calibration of the exponents (see :class:`AssociatedKernel`);
        ``rho`` is then a dimensionless global sharpness.
    """

    def __init__(self, families="auto", h=None, rho=0.5, combination="product",
                 normalize=True, C=1.0, adapt=None, kernel_type="ppk",
                 calibrate=True, class_weight=None, random_state=0):
        self.families = families
        self.h = h
        self.rho = rho
        self.combination = combination
        self.normalize = normalize
        self.C = C
        self.adapt = adapt
        self.kernel_type = kernel_type
        self.calibrate = calibrate
        self.class_weight = class_weight
        self.random_state = random_state

    def _make_kernel(self, X, rho, weights=None, combination=None):
        return AssociatedKernel(self.families, self.h, rho,
                                combination or self.combination, weights,
                                self.normalize, calibrate=self.calibrate).fit(X)

    def _gram(self, X, Z=None):
        if self.kernel_type == "naive":
            return self.kernel_.naive_symmetric_gram(X, Z)
        return self.kernel_(X, Z)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        d = X.shape[1]
        rho = np.broadcast_to(np.asarray(self.rho, float), (d,)).copy()
        self.kernel_ = self._make_kernel(X, rho)
        if self.adapt == "kta":
            m, self.alignment_ = select_feature_rho(self.kernel_, X, y,
                                                    random_state=self.random_state)
            keep = m > 0
            self.feature_multipliers_ = m
            # a zero multiplier removes the feature (k_j == 1)
            rho_new = np.where(keep, rho * np.where(keep, m, 1.0), rho)
            self.kernel_ = self._make_kernel(X, rho_new)
        elif self.adapt == "alignf":
            logs = self.kernel_.feature_log_kernels(X)
            w = alignf_weights(logs, y)
            self.kernel_ = self._make_kernel(X, rho, w, "sum")
            self.feature_weights_ = w
        elif self.adapt is not None:
            raise ValueError("adapt must be None, 'kta' or 'alignf'")
        self.active_features_ = (self.feature_multipliers_ > 0 if self.adapt == "kta"
                                 else np.ones(d, bool))
        self.X_fit_ = X
        K = self._gram_active(X)
        self.svc_ = SVC(C=self.C, kernel="precomputed", class_weight=self.class_weight)
        self.svc_.fit(K, y)
        return self

    def _gram_active(self, X, Z=None):
        if self.kernel_type == "ppk" and not np.all(self.active_features_):
            logs = self.kernel_.feature_log_kernels(X, Z)
            logs = [L for L, a in zip(logs, self.active_features_) if a]
            if not logs:
                n = len(X)
                m = n if Z is None else len(Z)
                return np.ones((n, m))
            return np.exp(np.sum(logs, axis=0))
        return self._gram(X, Z)

    def decision_function(self, X):
        check_is_fitted(self, "svc_")
        return self.svc_.decision_function(self._gram_active(np.asarray(X, float), self.X_fit_))

    def predict(self, X):
        check_is_fitted(self, "svc_")
        return self.svc_.predict(self._gram_active(np.asarray(X, float), self.X_fit_))

    @property
    def n_support_(self):
        return self.svc_.n_support_
