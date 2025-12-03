from typing import Optional

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist


class LaplacianEigenmaps:
    """
    Laplacian Eigenmaps solver for dimensionality reduction.

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions for the embedding.
    n_neighbors : int, default=10
        Number of neighbors for k-NN graph construction.
    epsilon : float, optional
        Radius for epsilon-neighborhood graph construction. Required if affinity='epsilon'.
    affinity : str, default='knn'
        Method to construct the affinity graph.
        - 'knn': k-Nearest Neighbors graph.
        - 'epsilon': Epsilon-neighborhood graph.
    weight : str, default='heat'
        Weighting scheme for the edges.
        - 'heat': Heat kernel weights.
        - 'binary': Binary weights (0/1).
    sigma : float, default=1.0
        Parameter for the heat kernel weighting.
    normalization : str, default='generalized'
        - 'generalized': Solves the generalized eigenvalue problem L f = λ D f.
            (Corresponds to Unnormalized Laplacian).
        - 'normalized': Solves the symmetric problem D^-1/2 L D^-1/2 f = λ f.
            (Corresponds to Normalized Laplacian).
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        epsilon: Optional[float] = None,
        affinity: str = "knn",
        weight: str = "heat",
        sigma: float = 1.0,
        normalization: str = "generalized",  # <--- NEW PARAMETER
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon
        self.affinity = affinity
        self.weight = weight
        self.sigma = sigma
        self.normalization = normalization
        self.random_state = random_state

        self.embedding_ = None
        self.eigenvalues_ = None
        self.affinity_matrix_ = None

    def _compute_affinity_matrix(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        distances = cdist(X, X, metric="euclidean")

        if self.affinity == "epsilon":
            if self.epsilon is None:
                raise ValueError("epsilon must be provided when affinity='epsilon'")
            adjacency = (distances <= self.epsilon).astype(float)
        else:
            # k-NN graph
            # Optimization: We could use argpartition instead of argsort for speed,
            # but argsort is fine for N=2000.
            adjacency = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                # indices of k nearest neighbors (excluding self at index 0)
                neighbors = np.argsort(distances[i])[1 : self.n_neighbors + 1]
                adjacency[i, neighbors] = 1

            # Symmetrize (OR logic: connected if i->j OR j->i)
            adjacency = np.maximum(adjacency, adjacency.T)

        np.fill_diagonal(adjacency, 0)

        if self.weight == "heat":
            W = adjacency * np.exp(-(distances**2) / (2 * self.sigma**2))
        else:
            W = adjacency

        return W

    def fit(self, X: np.ndarray) -> "LaplacianEigenmaps":
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 1. Affinity
        W = self._compute_affinity_matrix(X)
        self.affinity_matrix_ = W

        # 2. Laplacian & Degree
        d = np.sum(W, axis=1)
        D = np.diag(d)
        L = D - W

        # Determine the range of eigenvalues we want.
        # We generally want the smallest non-zero eigenvalues.
        # Index 0 is the trivial solution (constant vector), so we want indices [1, n_components]
        # range is inclusive for subset_by_index in scipy.
        eig_start = 1
        eig_end = self.n_components + 1

        if self.normalization == "generalized":
            # --- PATH A: Generalized Eigenvalue Problem ---
            # Solve: L y = λ D y
            # scipy.linalg.eigh can solve A x = λ B x directly.

            # Regularize D slightly to prevent numerical instability if graph has disconnected parts
            d_reg = d + 1e-10
            D_reg = np.diag(d_reg)

            eigenvalues, eigenvectors = eigh(
                L,
                b=D_reg,
                subset_by_index=(eig_start, eig_end - 1),  # Only fetch what we need!
            )

        elif self.normalization == "normalized":
            # --- PATH B: Symmetric Normalized Laplacian ---
            # Solve: D^-1/2 L D^-1/2 y = λ y

            # Inverse Square Root of D
            d_inv_sqrt = np.power(d + 1e-10, -0.5)
            D_inv_sqrt = np.diag(d_inv_sqrt)

            # L_sym = I - D^-1/2 W D^-1/2  OR  D^-1/2 L D^-1/2
            L_sym = D_inv_sqrt @ L @ D_inv_sqrt

            # Ensure symmetry numerically
            L_sym = (L_sym + L_sym.T) / 2.0

            eigenvalues, eigenvectors_sym = eigh(
                L_sym,
                subset_by_index=(eig_start, eig_end - 1),  # Only fetch what we need!
            )

            # Important: Transform eigenvectors back to original space
            # y = D^-1/2 * v
            eigenvectors = D_inv_sqrt @ eigenvectors_sym

        else:
            raise ValueError("Normalization must be 'generalized' or 'normalized'")

        self.eigenvalues_ = eigenvalues
        self.embedding_ = eigenvectors

        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_
