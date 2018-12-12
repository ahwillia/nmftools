from sklearn.decomposition import NMF
from .utils import align_factors
from tqdm import tqdm
import numpy as np


def fit_ensemble(data, ranks, n_replicates=10, **model_args):

    model_args.pop('init', None)
    results = {}

    for r in tqdm(ranks):

        # set number of components
        model_args['n_components'] = r

        models = [NMF(init='random', **model_args) for _ in range(n_replicates - 1)]
        models.append(NMF(init='nndsvd', **model_args))

        # expand results
        results[r] = {
            'factors': [],
            'rmse': [],
            'similarity': [1.0]
        }

        for m in models:
            W = m.fit_transform(data)
            H = m.components_
            results[r]['factors'].append([W, H])
            est = np.dot(W, H)
            results[r]['rmse'].append(np.sqrt(np.mean((est - data)**2)))

        # resort by reconstruction error
        ii = np.argsort(results[r]['rmse'])
        for k in 'factors', 'rmse':
            results[r][k] = [results[r][k][i] for i in ii]

        # compute similarity to best fit
        best_fctr = results[r]['factors'][0]
        for i in range(1, n_replicates):
            fctr = results[r]['factors'][i]
            _, score = align_factors(fctr, best_fctr)
            results[r]['similarity'].append(score)

    # compute svd for comparison
    u, s, vt = np.linalg.svd(data, full_matrices=False)

    for r in ranks:
        est = np.dot(u[:, :r] * s[:r], vt[:r])
        results[r]['svd_rmse'] = np.sqrt(np.mean((est - data)**2))

    return results
