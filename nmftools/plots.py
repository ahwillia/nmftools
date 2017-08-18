import matplotlib.pyplot as plt
import numpy as np

def plot_scree(results, ax=None, jitter=0.1, scatter_kw=dict(), line_kw=dict()):

    if ax is None:
        ax = plt.gca()

    # compile statistics for plotting
    ranks, err, sim, min_err = [], [], [], []
    for r in results:
        # reconstruction errors for rank-r models
        e = results[r]['rmse']
        err.append(list(e))
        min_err.append(min(e))
        ranks.append([r for _ in range(len(e))])

        # similarity of fitted models
        sim += list(results[r]['similarity'][1:])

    # add horizontal jitter 
    ranks = np.array(ranks)
    if jitter is not None:
        ranks_jit = ranks + (np.random.rand(*ranks.shape)-0.5)*jitter

    # Scree plot
    ax.scatter(ranks_jit.ravel(), err, **scatter_kw)
    ax.plot(ranks[:, 0], min_err, **line_kw)
    
    return ax


def plot_similarity(results, ax=None, jitter=0.1, labels=True, scatter_kw=dict(), line_kw=dict()):
    
    if ax is None:
        ax = plt.gca()

    # compile statistics for plotting
    ranks, sim, mean_sim = [], [], []
    for r in results.keys():
        ranks.append([r for _ in range(len(results[r]['factors'])-1)])
        sim += list(results[r]['similarity'][1:])
        mean_sim.append(np.mean(results[r]['similarity'][1:]))

    # add horizontal jitter 
    ranks = np.array(ranks)
    if jitter is not None:
        ranks_jit = ranks + (np.random.rand(*ranks.shape)-0.5)*jitter

    # make plot
    ax.scatter(ranks_jit.ravel(), sim, **scatter_kw)
    ax.plot(ranks[:, 0], mean_sim, **line_kw)

    if labels:
        ax.set_xlabel('model rank')
        ax.set_ylabel('Norm of resids / Norm of data')
        
    ax.scatter(ranks_jit.ravel(), sim, **scatter_kw)

    # axis labels
    if labels:
        ax.set_xlabel('model rank')
        ax.set_ylabel('model similarity')
    
    return ax
