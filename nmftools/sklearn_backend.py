
def fit_nmf_sklearn(data, rank, nreplicates, **kwargs):
    
    model_args.pop('init', None)
    kwargs['n_components'] = rank
    models = [NMF(init='random', **model_args) for _ in range(nreplicates-1)]
    models.append(NMF(init='nndsvd', **model_args))