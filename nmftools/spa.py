import numpy as np
from scipy.spatial.distance import pdist

from .nnls import nnls

def fit_nmf_spa(data, rank, nreplicates):
    """
    Fits NMF by the Successive Projection Algorithm [CITE].

    Args:
        data (ndarray) - matrix of nonnegative data
        rank (int) - number of components

    Definition - Separability:
        An m x n nonnegative matrix M is separable if admits the factorization
        M = W*H (where W is m x r and H is an r x n), and furthermore the
        columns of H can be permuted to form a diagonal submatrix.
    """
    R = data.copy()
    J = []
    for i in range(rank):
        j = np.argmax(np.sum(R**2, axis=0))
        u = R[:, j]
        R -= np.dot(u, np.dot(u.T, R)) / np.sum(u**2)
        J.append(j)
    W = data[:, np.array(j)]
    H = nnls(W, data)
    return W, H

def fit_nmf_cvxhull(X, rank, nreplicates):
    R = data.copy()
    XtX = np.dot(X.T, X)

    J = []
    for i in range(rank):
        j = np.argmax(np.sum(R**2, axis=0))
        u = R[:, j]
        R -= np.dot(u, np.dot(u.T, R)) / np.sum(u**2)
        J.append(j)
    W = data[:, np.array(j)]
    H = nnls(W, data)
    return W, H

