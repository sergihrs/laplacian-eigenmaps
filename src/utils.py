"""
Utility functions for Laplacian Eigenmaps.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import make_swiss_roll
import pandas as pd


def create_swiss_roll(
    n_samples: int = 1000,
    noise: float = 0.0,
    random_state: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a Swiss Roll dataset.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of sample points.
    noise : float, default=0.0
        Standard deviation of Gaussian noise added to the data.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        The generated Swiss Roll data points.
    t : ndarray of shape (n_samples,)
        The univariate position of the sample according to the main dimension
        of the points in the manifold (used for coloring).
    """
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
    X[:, 1] = X[:, 1] * 3  # Stretch the width by 3x
    return X, t


def compute_pairwise_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """
    Compute pairwise distances between all points in X.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    metric : str, default="euclidean"
        Distance metric to use.

    Returns
    -------
    distances : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix.
    """
    return cdist(X, X, metric=metric)


def knn_graph(
    distances: np.ndarray,
    k: int = 10,
    mode: str = "connectivity",
) -> np.ndarray:
    """
    Construct a k-nearest neighbors graph from pairwise distances.

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix.
    k : int, default=10
        Number of nearest neighbors.
    mode : str, default="connectivity"
        Type of graph to construct:
        - "connectivity": Binary adjacency matrix.
        - "distance": Weight by distances.

    Returns
    -------
    W : ndarray of shape (n_samples, n_samples)
        Adjacency/weight matrix.
    """
    n_samples = distances.shape[0]
    W = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        # Get k nearest neighbors (excluding self)
        neighbors = np.argsort(distances[i])[1 : k + 1]

        if mode == "connectivity":
            W[i, neighbors] = 1
        else:  # distance
            W[i, neighbors] = distances[i, neighbors]

    # Make symmetric
    W = np.maximum(W, W.T)

    return W


def epsilon_graph(
    distances: np.ndarray,
    epsilon: float,
    mode: str = "connectivity",
) -> np.ndarray:
    """
    Construct an epsilon-neighborhood graph from pairwise distances.

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix.
    epsilon : float
        Radius threshold.
    mode : str, default="connectivity"
        Type of graph to construct:
        - "connectivity": Binary adjacency matrix.
        - "distance": Weight by distances.

    Returns
    -------
    W : ndarray of shape (n_samples, n_samples)
        Adjacency/weight matrix.
    """
    if mode == "connectivity":
        W = (distances <= epsilon).astype(float)
    else:  # distance
        W = np.where(distances <= epsilon, distances, 0)

    # Remove self-loops
    np.fill_diagonal(W, 0)

    return W


def plot_embedding(
    X: np.ndarray,
    colors: Optional[np.ndarray] = None,
    title: str = "2D Embedding",
    figsize: tuple = (8, 6),
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D embedding.

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
        2D embedding coordinates.
    colors : ndarray, optional
        Values for coloring the points.
    title : str, default="2D Embedding"
        Plot title.
    figsize : tuple, default=(8, 6)
        Figure size.
    cmap : str, default="viridis"
        Colormap to use.
    ax : plt.Axes, optional
        Existing axes to plot on.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    ax : plt.Axes
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    scatter = ax.scatter(X[:, 0], X[:, 1], c=colors, cmap=cmap, s=10, alpha=0.7)

    if colors is not None:
        plt.colorbar(scatter, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")

    return fig, ax


def plot_comparison(
    embeddings: dict[str, np.ndarray],
    colors: Optional[np.ndarray] = None,
    figsize: tuple = (15, 5),
    cmap: str = "viridis",
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot multiple embeddings for comparison.

    Parameters
    ----------
    embeddings : dict
        Dictionary mapping method names to embedding arrays.
    colors : ndarray, optional
        Values for coloring the points.
    figsize : tuple, default=(15, 5)
        Figure size.
    cmap : str, default="viridis"
        Colormap to use.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    axes : ndarray
        Array of axes objects.
    """
    n_embeddings = len(embeddings)
    fig, axes = plt.subplots(1, n_embeddings, figsize=figsize)

    if n_embeddings == 1:
        axes = [axes]

    for ax, (name, X) in zip(axes, embeddings.items()):
        scatter = ax.scatter(X[:, 0], X[:, 1], c=colors, cmap=cmap, s=10, alpha=0.7)
        ax.set_title(name)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    if colors is not None:
        fig.colorbar(scatter, ax=axes, shrink=0.6)

    plt.tight_layout()

    return fig, np.array(axes)


def load_mammoth_data():
    """
    Loads the Mammoth 3D dataset (mammoth_a.csv).
    Returns a numpy array of shape (n_samples, 3).
    """
    print("Loading Mammoth 3D dataset...")
    df = pd.read_csv("data/mammoth_a.csv",  dtype=np.float32, header=None, sep=',')
    data = df.to_numpy()
    print(f"Data shape: {data.shape}")
    return data