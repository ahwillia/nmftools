"""
Simple demonstration of plotting functions.
"""
import numpy as np
import numpy.random as npr
from nmftools import fit_ensemble, plot_rmse, plot_similarity
import matplotlib.pyplot as plt


# Size of data matrix.
m, n = 100, 101

# Rank of decomposition.
rank = 5

# True low-dimensional factors.
W_true = npr.rand(m, rank)
H_true = npr.rand(rank, n)

# Create noisy data.
noise = .2
data = np.dot(W_true, H_true) + noise * npr.rand(m, n)

# Fit models.
results = fit_ensemble(data, np.arange(1, 10), n_replicates=10)

# Plot objective.
plt.figure()
plot_rmse(results)

# Plot similarity / stability.
plt.figure()
plot_similarity(results)

plt.show()
