"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



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
    K, _ = mixture.mu.shape
    n, d = X.shape
    gprob = lambda x, m, s: (1 / (2*np.pi*s)**(d/2)) * (np.exp(-((x-m)**2).sum(axis=1) / (2*s)))
    soft_counts, ll_ = np.empty((0,K)), np.empty((0,K))

    for i in range(n):
        prob = gprob(np.tile(X[i], (K,1)), mixture.mu, mixture.var)
        # import pdb; pdb.set_trace()
        # prob = MN.logpdf(np.tile(X[i], (K,1)), mixture.mu, mixture.var)
        prob = prob.reshape(1, K)
        prob1 = (prob*mixture.p)/(prob*mixture.p).sum()
        soft_counts = np.append(soft_counts, prob1, axis=0)
        ll_ =  np.append(ll_, prob, axis=0)
    ll = np.log((ll_*mixture.p).sum(axis=1)).sum()
    # ll = np.sum(np.log(np.sum(soft_counts*mixture.p, axis = 1)))

    return soft_counts, ll
   
    raise NotImplementedError


from numpy.linalg import norm

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    nrow, ncol = X.shape
    _, K = post.shape

    up_mu = np.zeros((K, ncol))     # Initialize updates of mu
    up_var = np.zeros(K)            # Initialize updates of var
    n_hat = post.sum(axis=0)        # Nk
    """Updates"""
    up_p = 1/nrow * n_hat           # Updates of pq
    for j in range(K):
        # import pdb; pdb.set_trace()
        up_mu[j] = (post.T @ X)[j] / post.sum(axis=0)[j]
        sse = ((up_mu[j] - X)**2).sum(axis=1) @ post[:, j]
        up_var[j] = sse / (ncol * n_hat[j])
    
    return GaussianMixture(mu=up_mu, var=up_var, p=up_p)

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
    old_ll = None
    new_ll = None
    while (old_ll is None or new_ll - old_ll > 1e-6 * abs(new_ll)):
        old_ll = new_ll
        post, new_ll = estep(X, mixture)
        # soft_counts = post * 1/
        mixture = mstep(X, post)

    return mixture, post, new_ll
    
    raise NotImplementedError

