import numpy as np
import itertools as itr


def normalize_factors(W, H):
    """
    Normalize Euclidean length of NMF factors.

    Parameters
    ----------
    W : ndarray
        First set of NMF model factors (tall-skinny matrix).
    H : ndarray
        Second set of NMF model factors (short-fat matrix).

    Returns
    -------
    W_nrm : ndarray
        Same as W but with normalized columns.
    H_nrm : ndarray
        Same as H but with normalized rows.
    lam : ndarray
        Vector holding the size / scale of each set of factors.
    """
    l1 = np.linalg.norm(W, axis=0, keepdims=True)
    l2 = np.linalg.norm(H, axis=1, keepdims=True)
    return W / l1, H / l2, l1.ravel() * l2.ravel()


def align_factors(model_1, model_2):
    """
    Align NMF factors of model_1 to model_2.

    Parameters
    ----------
    model_1 : tuple
        NMF factors from first model (W1, H1).
    model_2 : tuple
        NMF factors from second model (W2, H2).

    Returns
    -------
    permuted_model_1 : tuple
        Permuted NMF factors from model_1.
    score :
    """

    W1, H1, lam1 = normalize_factors(*model_1)
    W2, H2, lam2 = normalize_factors(*model_2)

    rank = len(lam1)
    assert len(lam2) == rank

    sim = np.dot(W1.T, W2) * np.dot(H1, H2.T)

    score = -1
    best_perm = np.arange(rank)
    for p in itr.permutations(range(rank)):
        sc = sum([sim[i, j] for j, i in enumerate(p)])
        if sc > score:
            best_perm = list(p)
            score = sc
    score /= rank

    # permute A to align with B
    W1 = W1[:, best_perm] * lam1
    H1 = H1[:, best_perm]
    return (W1, H1),  score
