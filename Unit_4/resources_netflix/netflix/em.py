"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from numpy.linalg import norm


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    
    K, _ = mixture.mu.shape
    n, d = X.shape
    gprob = lambda x, m, s: (1 / (2*np.pi*s)**(d/2)) * (np.exp(-((x-m)**2).sum(axis=1) / (2*s)))
    soft_counts, ll_ = np.empty((0, K)), np.empty((0, K))

    for i in range(n):
        import pdb; pdb.set_trace()
        prob = gprob(np.tile(X[i], (K, 1)), mixture.mu, mixture.var)
        prob = prob.reshape(1, K)
        prob_post = (prob*mixture.p)/(prob*mixture.p).sum()
        soft_counts = np.append(soft_counts, prob_post, axis=0)
        ll_ = np.append(ll_, prob, axis=0)
    ll = np.log((ll_*mixture.p).sum(axis=1)).sum()

    return soft_counts, ll
    
    raise NotImplementedError


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    K, _ = mixture.mu.shape
    n, d = X.shape
    gprob = lambda x, m, s: (1 / (2*np.pi*s)**(d/2)) * (np.exp(-((x-m)**2).sum(axis=1) / (2*s)))
    soft_counts, ll_ = np.empty((0, K)), np.empty((0, K))

    for i in range(n):
        import pdb; pdb.set_trace()
        prob = gprob(np.tile(X[i], (K, 1)), mixture.mu, mixture.var)
        prob = prob.reshape(1, K)
        prob_post = (prob*mixture.p)/(prob*mixture.p).sum()
        soft_counts = np.append(soft_counts, prob_post, axis=0)
        ll_ = np.append(ll_, prob, axis=0)
    ll = np.log((ll_*mixture.p).sum(axis=1)).sum()

    return soft_counts, ll
    
    raise NotImplementedError


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
    old_ll = new_ll = post = None

    while (old_ll is None) or (new_ll - old_ll > 1e-6 * abs(new_ll)):
        old_ll = new_ll
        post, new_ll = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, old_ll, new_ll
    
    raise NotImplementedError



def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    old_ll = new_ll = post = None

    while (old_ll is None) or (new_ll - old_ll > 1e-6 * abs(new_ll)):
        old_ll = new_ll
        post, new_ll = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, old_ll, new_ll
    
    raise NotImplementedError
