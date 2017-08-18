import numpy as np
import itertools as itr

def normalize_factors(A):
    W, H = A
    l1 = np.linalg.norm(W, axis=0)
    l2 = np.linalg.norm(H, axis=1)
    W /= l1[None,:]
    H /= l2[:,None]
    return W, H, l1*l2

def align_factors(A, B):
    """Align NMF factors A to B

    Inputs
    ------
    A : tuple of NMF factors
    B : tuple of NMF factors

    Returns
    -------
    aligned_A : permuted version of A 
    """

    W1, H1, lam1 = normalize_factors(A)
    W2, H2, lam2 = normalize_factors(B)

    rank = len(lam1)
    assert len(lam2) == rank

    sim = np.dot(W1.T, W2) * np.dot(H1, H2.T)

    score = -1
    best_perm = np.arange(rank)
    for p in itr.permutations(range(rank)):
        sc = sum([ sim[i,j] for j, i in enumerate(p)])
        if sc > score:
            best_perm = list(p)
            score = sc
    score /= rank

    # permute A to align with B
    W1 = W1[:, best_perm] * lam1
    H1 = H1[:, best_perm]
    return (W1, H1),  score

