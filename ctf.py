# include all packages, including those needed for the children classes
from abc import ABCMeta
from collections.abc import Mapping
from copy import copy

import numpy as np
from numpy.linalg import norm, lstsq
from tensorly.tenalg import multi_mode_dot
from tensorly.decomposition._cp import parafac

import numpy as np
from numpy.linalg import norm
from tensorly import fold
from tensorly.tenalg import khatri_rao
from functools import reduce


class ctPLS(Mapping, metaclass=ABCMeta):
    """Coupled 2-mode tensor PLS"""

    def __init__(self, n_components: int):
        """Store the number of latent components to fit."""
        super().__init__()
        # Parameters
        self.n_components = n_components

    def __getitem__(self, index):
        """Return factor matrices or coefficients by fixed integer index."""
        if index == 0:
            return self.Xs_factors
        elif index == 1:
            return self.Y_factors
        elif index == 2:
            return self.coef_
        else:
            raise IndexError

    def __iter__(self):
        """Iterate through X factors, Y factors, and regression coefficients."""
        yield self.Xs_factors
        yield self.Y_factors
        yield self.coef_

    def __len__(self):
        """Report mapping length used by the Mapping interface."""
        return 3

    def copy(self):
        """Return a shallow copy of the fitted model object."""
        return copy(self)

    """
    Coupled tensor PLS with 2 shared modes (mode 0 and mode 1).
    All tensors in Xs must have the same size along both mode 0 and mode 1,
    and must be at least 3-way tensors.
    """
 
    def preprocess(self, Xs, Y):
        """Validate inputs, center tensors/targets, and initialize model state."""
        # check input integrity
        assert isinstance(Xs, list)
 
        for X in Xs:
            assert X.shape[0] == len(Y)
            assert X.ndim >= 3, "Tensors must be at least 3-way for 2-mode coupling"
            assert X.shape[1] == Xs[0].shape[1], "All tensors must share mode-1 size"
            assert X.shape[0] == Xs[0].shape[0], "All tensors must share mode-0 size"
        assert Y.ndim <= 2, "Only a matrix (2-mode tensor) Y is acceptable."
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
 
        # mean center the data; set up factors
        self.Xs_len = len(Xs)
        self.Xs_dim = [X.ndim for X in Xs]
        self.Xs_shape = [X.shape for X in Xs]
        self.Y_shape = Y.shape
 
        # Shared factor matrices — same objects placed in every tensor's factor list
        self.factor_T = np.zeros((self.Y_shape[0], self.n_components))   # shared mode 0
        self.factor_S = np.zeros((Xs[0].shape[1], self.n_components))    # shared mode 1
 
        self.Xs_factors = [
            [self.factor_T, self.factor_S]     # shared modes 0 & 1
            + [np.zeros((lf, self.n_components)) for lf in X.shape[2:]] # tensor-specific modes 2+
            for X in Xs
        ]
        self.Y_factors = [np.zeros((lf, self.n_components)) for lf in Y.shape]
        self.coef_ = np.zeros((self.n_components, self.n_components))
 
        self.R2Xs = [np.zeros((self.n_components)) for _ in range(self.Xs_len)]
        self.R2Y = np.zeros((self.n_components))
 
        self.Xs_mean = [np.nanmean(X, axis=0) for X in Xs]
        self.Y_mean = np.nanmean(Y, axis=0)
 
        self.Xs_hasMiss = [np.any(np.isnan(X)) for X in Xs]
        if any(self.Xs_hasMiss):
            print("At least one X has missing values")
        self.Xs_miss = [np.isnan(X) for X in Xs]
 
        return [X - self.Xs_mean[i] for (i, X) in enumerate(Xs)], Y - self.Y_mean

    def fit(self, Xs, Y, tol=1e-10, max_iter=1000, verbose=0):
        """Fit coupled tPLS factors and regression coefficients."""
        oXs, oY = [X.copy() for X in Xs], Y
        Xs, Y = self.preprocess(Xs, Y)
        for a in range(self.n_components):
            oldU = np.ones_like(self.Y_factors[0][:, a]) * np.inf
            self.Y_factors[0][:, a] = Y[:, 0]
            for iter in range(max_iter):
                for ti, X in enumerate(Xs):
                    if not self.Xs_hasMiss[ti]:
                        Z = np.einsum("i...,i...->...", X, self.Y_factors[0][:, a])
                    else:
                        Z = miss_tensordot(X, self.Y_factors[0][:, a], self.Xs_miss[ti])

                    Z_comp = [Z / norm(Z)]
                    if Z.ndim >= 2:
                        Z_comp = parafac(
                            Z, 1, tol=tol, init="svd", normalize_factors=True
                        )[1]
                    for ii in range(Z.ndim):
                        self.Xs_factors[ti][ii + 1][:, a] = Z_comp[ii].flatten()

                Ts = [
                    multi_mode_dot(
                        Xs[ti],
                        [ff[:, a] for ff in self.Xs_factors[ti][1:]],
                        range(1, self.Xs_dim[ti]),
                    )
                    if not self.Xs_hasMiss[ti]
                    else miss_mmodedot(
                        Xs[ti],
                        [ff[:, a] for ff in self.Xs_factors[ti][1:]],
                        self.Xs_miss[ti],
                    )
                    for ti in range(self.Xs_len)
                ]
                self.factor_T[:, a] = np.average(Ts, axis=0)
                self.Y_factors[1][:, a] = Y.T @ self.factor_T[:, a]
                self.Y_factors[1][:, a] /= norm(self.Y_factors[1][:, a])
                self.Y_factors[0][:, a] = Y @ self.Y_factors[1][:, a]
                if norm(oldU - self.Y_factors[0][:, a]) < tol:
                    if verbose:
                        print("Comp {}: converged after {} iterations".format(a, iter))
                    break
                oldU = self.Y_factors[0][:, a].copy()

            for ti, X in enumerate(Xs):
                X -= factors_to_tensor([ff[:, [a]] for ff in self.Xs_factors[ti]])
                self.R2Xs[ti][a] = calcR2X(
                    oXs[ti] - self.Xs_mean[ti], factors_to_tensor(self.Xs_factors[ti])
                )
            self.coef_[:, a] = lstsq(self.factor_T, self.Y_factors[0][:, a], rcond=-1)[
                0
            ]
            Y -= self.factor_T @ self.coef_[:, [a]] @ self.Y_factors[1][:, [a]].T
            self.R2Y[a] = calcR2X(oY - self.Y_mean, self.predict(oXs) - self.Y_mean)
            # Y -= T b q' = T pinv(T) u q' = T lstsq(T, u) q'; b = inv(T'T) T' u = pinv(T) u

    def predict(self, Xs):
        """Predict target values for new coupled tensors."""
        Xs = [X.copy() for X in Xs]
        assert len(Xs) == self.Xs_len
        Xs_hasMiss = [np.any(np.isnan(X)) for X in Xs]
        Xs_miss = [np.isnan(X) for X in Xs]
        for ti, X in enumerate(Xs):
            if self.Xs_shape[ti][1:] != X.shape[1:]:
                raise ValueError(
                    f"Training X[{ti}] has shape {self.Xs_shape[ti]}, while the new X has shape {X.shape}"
                )
            Xs[ti] -= self.Xs_mean[ti]
        X_projection = np.zeros((Xs[0].shape[0], self.n_components))
        for a in range(self.n_components):
            X_projection[:, a] = np.average(
                [
                    multi_mode_dot(
                        Xs[ti],
                        [ff[:, a] for ff in self.Xs_factors[ti][1:]],
                        range(1, self.Xs_dim[ti]),
                    )
                    if not Xs_hasMiss[ti]
                    else miss_mmodedot(
                        Xs[ti],
                        [ff[:, a] for ff in self.Xs_factors[ti][1:]],
                        Xs_miss[ti],
                    )
                    for ti in range(self.Xs_len)
                ],
                axis=0,
            )
            for ti, X in enumerate(Xs):
                X -= factors_to_tensor(
                    [X_projection[:, [a]]]
                    + [ff[:, [a]] for ff in self.Xs_factors[ti][1:]]
                )
        return X_projection @ self.coef_ @ self.Y_factors[1].T + self.Y_mean

    def transform(self, Xs, Y=None):
        """Project tensors (and optional targets) into latent score space."""
        Xs = [X.copy() for X in Xs]
        assert len(Xs) == self.Xs_len
        Xs_hasMiss = [np.any(np.isnan(X)) for X in Xs]
        Xs_miss = [np.isnan(X) for X in Xs]
        for ti, X in enumerate(Xs):
            if self.Xs_shape[ti][1:] != X.shape[1:]:
                raise ValueError(
                    f"Training X[{ti}] has shape {self.Xs_shape[ti]}, while the new X has shape {X.shape}"
                )
            Xs[ti] -= self.Xs_mean[ti]
        X_scores = np.zeros((Xs[0].shape[0], self.n_components))

        for a in range(self.n_components):
            Ts = [
                multi_mode_dot(
                    Xs[ti],
                    [ff[:, a] for ff in self.Xs_factors[ti][1:]],
                    range(1, self.Xs_dim[ti]),
                )
                if not Xs_hasMiss[ti]
                else miss_mmodedot(
                    Xs[ti], [ff[:, a] for ff in self.Xs_factors[ti][1:]], Xs_miss[ti]
                )
                for ti in range(self.Xs_len)
            ]

            X_scores[:, a] = np.average(Ts, axis=0)
            for ti, X in enumerate(Xs):
                X -= factors_to_tensor(
                    [X_scores[:, [a]]] + [ff[:, [a]] for ff in self.Xs_factors[ti][1:]]
                )

        if Y is not None:
            Y = Y.copy()
            # Check on the shape of Y
            if (Y.ndim != 1) and (Y.ndim != 2):
                raise ValueError("Only a matrix (2-mode tensor) Y is allowed.")
            if Y.ndim == 1:
                Y = Y.reshape((-1, 1))
            if self.Y_shape[1:] != Y.shape[1:]:
                raise ValueError(
                    f"Training Y has shape {self.Y_shape}, while the new Y has shape {Y.shape}"
                )

            Y -= self.Y_mean
            Y_scores = np.zeros((Y.shape[0], self.n_components))
            for a in range(self.n_components):
                Y_scores[:, a] = Y @ self.Y_factors[1][:, a]
                Y -= X_scores @ self.coef_[:, [a]] @ self.Y_factors[1][:, [a]].T
            return X_scores, Y_scores

        return X_scores

    def Xs_reconstructed(self):
        """Reconstruct each input tensor from learned factors plus means."""
        return [
            factors_to_tensor(self.Xs_factors[ti]) + self.Xs_mean[ti]
            for ti in range(self.Xs_len)
        ]


def calcR2X(X, Xhat):
    """Compute explained-variance style R2 between original and reconstructed arrays."""
    if (Xhat.ndim == 2) and (X.ndim == 1):
        X = X.reshape(-1, 1)
    assert X.shape == Xhat.shape
    mask = np.isfinite(X)
    xIn = np.nan_to_num(X)
    top = norm(Xhat * mask - xIn) ** 2.0
    bottom = norm(xIn) ** 2.0
    return 1 - top / bottom


def factors_to_tensor(factors):
    """Build a full tensor from CP-style factor matrices."""
    full_tensor = factors[0] @ khatri_rao(factors, skip_matrix=0).T
    return fold(full_tensor, 0, [ff.shape[0] for ff in factors])


def miss_tensordot(X, u, missX=None):
    """Tensor-vector contraction that ignores missing entries in X."""
    # Equivalent to np.einsum("i...,i...->...", X, u), but X with missing values at missX
    Xdim = X.shape
    assert Xdim[0] == u.shape[0]
    if missX is None:
        missX = np.isnan(X)
    X = X.reshape(Xdim[0], -1)
    missX = missX.reshape(Xdim[0], -1)
    w = np.zeros((X.shape[1],))
    for i in range(X.shape[1]):
        m = np.where(~missX[:, i])[0]
        if len(m) > 0:
            w[i] = X[m, i].T @ u[m] / len(m) * Xdim[0]
    return w.reshape(Xdim[1:])


def miss_mmodedot(X, facs, missX=None):
    """Multi-mode tensor product that ignores missing entries in X."""
    # Equivalent to multi_mode_dot(X, fac, range(1, X.ndim)), but X with missing values at missX
    # facs ~= [ff[:, a] for ff in self.X_factors[1:]]
    Xdim = X.shape
    assert all([(Xdim[i + 1], ff.shape[0]) for (i, ff) in enumerate(facs)])
    if missX is None:
        missX = np.isnan(X)
    X = X.reshape(Xdim[0], -1)
    missX = missX.reshape(Xdim[0], -1)
    t = np.zeros((Xdim[0],))
    wkron = reduce(np.kron, facs)
    Wdim = wkron.shape[0]
    for i in range(Xdim[0]):
        m = np.where(~missX[i, :])[0]
        t[i] = X[i, m] @ wkron[m] / len(m) * Wdim
    return t