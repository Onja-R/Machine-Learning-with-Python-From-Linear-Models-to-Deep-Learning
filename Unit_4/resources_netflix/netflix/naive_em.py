"""Mixture model using EM"""
import numpy as np
from common import GaussianMixture
from typing import NamedTuple
from typing import Tuple


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K = mixture.mu.shape[0]

    # Unpack parameters
    mu, var, pi = mixture.mu, mixture.var, mixture.p

    # Compute squared distances: (n, K)
    # ||X - mu_k||^2 for each k
    diff = X[:, None, :] - mu[None, :, :]
    sq_dist = np.sum(diff * diff, axis=2)

    # Gaussian likelihoods (n, K)
    coef = (2 * np.pi * var) ** (d / 2)
    log_prob = -sq_dist / (2 * var) - np.log(coef)

    # Multiply by priors → in log space
    log_weighted = log_prob + np.log(pi)

    # Log-sum-exp for numerical stability
    max_log = np.max(log_weighted, axis=1, keepdims=True)
    log_sum_exp = max_log + np.log(np.sum(np.exp(log_weighted - max_log), axis=1, keepdims=True))

    # Soft counts = posterior responsibilities
    soft_counts = np.exp(log_weighted - log_sum_exp)

    # Total log-likelihood
    ll = np.sum(log_sum_exp)

    return soft_counts, ll
   
    raise NotImplementedError


from numpy.linalg import norm

""" def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
"""
"""
    M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
"""
    nrow, ncol = X.shape
    _, K = post.shape

    up_mu = np.zeros((K, ncol))     # Initialize updates of mu
    up_var = np.zeros(K)            # Initialize updates of var
    n_hat = post.sum(axis=0)        # Nk
    # Updates
    up_p = 1/nrow * n_hat           # Updates of pq
    for j in range(K):
        # import pdb; pdb.set_trace()
        up_mu[j] = (post.T @ X)[j] / post.sum(axis=0)[j]
        sse = ((up_mu[j] - X)**2).sum(axis=1) @ post[:, j]
        up_var[j] = sse / (ncol * n_hat[j])
    
    return GaussianMixture(mu=up_mu, var=up_var, p=up_p)

    raise NotImplementedError """


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset.
    
    Args:
        X: (n, d) array holding the data, where n is number of samples
           and d is dimensionality
        post: (n, K) array holding the soft counts (responsibilities)
              for all components for all examples
    
    Returns:
        GaussianMixture: the updated gaussian mixture with new parameters
        
    Notes:
        - Assumes spherical covariances (same variance in all dimensions)
        - Uses weighted maximum likelihood estimation
    """
    n, d = X.shape
    _, K = post.shape
    
    # Effective number of points assigned to each component
    n_hat = post.sum(axis=0)  # Shape: (K,)
    
    # Avoid division by zero for empty clusters
    n_hat = np.maximum(n_hat, 1e-10)
    
    # Update mixing coefficients (prior probabilities)
    up_p = n_hat / n
    
    # Update means using vectorized operations
    # post.T @ X gives (K, d) directly: sum of weighted points per component
    up_mu = (post.T @ X) / n_hat[:, np.newaxis]
    
    # Update variances (spherical covariance assumption)
    # Compute squared distances from each point to each mean
    up_var = np.zeros(K)
    for j in range(K):
        # Squared Euclidean distances: (n,) array
        sq_distances = np.sum((X - up_mu[j])**2, axis=1)
        # Weighted sum of squared distances
        up_var[j] = (sq_distances @ post[:, j]) / (d * n_hat[j])
    
    # Ensure minimum variance to avoid numerical issues
    up_var = np.maximum(up_var, 1e-10)
    
    return GaussianMixture(mu=up_mu, var=up_var, p=up_p)


def mstep_vectorized(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """Fully vectorized version of M-step (faster for large K).
    
    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
    
    Returns:
        GaussianMixture: the updated gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    
    # Effective number of points per component
    n_hat = post.sum(axis=0)
    n_hat = np.maximum(n_hat, 1e-10)
    
    # Update mixing coefficients
    up_p = n_hat / n
    
    # Update means
    up_mu = (post.T @ X) / n_hat[:, np.newaxis]
    
    # Vectorized variance computation
    # Expand dimensions for broadcasting: X (n,1,d), up_mu (1,K,d)
    X_expanded = X[:, np.newaxis, :]  # (n, 1, d)
    mu_expanded = up_mu[np.newaxis, :, :]  # (1, K, d)
    
    # Squared distances: (n, K, d) -> sum over d -> (n, K)
    sq_distances = np.sum((X_expanded - mu_expanded)**2, axis=2)
    
    # Weighted sum: element-wise multiply and sum over n
    up_var = (sq_distances * post).sum(axis=0) / (d * n_hat)
    up_var = np.maximum(up_var, 1e-10)
    
    return GaussianMixture(mu=up_mu, var=up_var, p=up_p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_ll = None
    new_ll = None
    while (old_ll is None or new_ll - old_ll > 1e-6 * abs(new_ll)):
        old_ll = new_ll
        post, new_ll = estep(X, mixture)
        # soft_counts = post * 1/
        mixture = mstep(X, post)

    return mixture, post, new_ll
    
    raise NotImplementedError

