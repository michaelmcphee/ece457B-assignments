import numpy as np
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

np.random.seed(0)  # for reproducibility

dims = [2**i for i in range(11)]  # d = 1, 2, 4, ..., 1024

sq_euclidean_means = []
sq_euclidean_stds = []
manhattan_means = []
manhattan_stds = []

for d in dims:
    points = np.random.uniform(0, 1, size=(100, d))

    sq_euclidean_dists = pdist(points, metric='sqeuclidean')
    manhattan_dists = pdist(points, metric='cityblock')

    sq_euclidean_means.append(np.mean(sq_euclidean_dists))
    sq_euclidean_stds.append(np.std(sq_euclidean_dists))
    manhattan_means.append(np.mean(manhattan_dists))
    manhattan_stds.append(np.std(manhattan_dists))

# Plot: average distance vs dimension (log-scale x-axis)
plt.figure(figsize=(8, 5))
plt.plot(dims, sq_euclidean_means, marker='o', label='Squared Euclidean (mean)')
plt.plot(dims, manhattan_means, marker='o', label='Manhattan (mean)')
plt.xscale('log', base=2)
plt.xlabel('Dimension d')
plt.ylabel('Average pairwise distance')
plt.title('Average Pairwise Distance vs Dimension (log scale)')
plt.legend()
plt.grid(True)
plt.savefig('q2_2_mean_log.png', dpi=150)
plt.show()

# Plot: average distance vs dimension (linear x-axis)
plt.figure(figsize=(8, 5))
plt.plot(dims, sq_euclidean_means, marker='o', label='Squared Euclidean (mean)')
plt.plot(dims, manhattan_means, marker='o', label='Manhattan (mean)')
plt.xlabel('Dimension d')
plt.ylabel('Average pairwise distance')
plt.title('Average Pairwise Distance vs Dimension (linear scale)')
plt.legend()
plt.grid(True)
plt.savefig('q2_2_mean_linear.png', dpi=150)
plt.show()

# Plot: standard deviation of distance vs dimension (log-scale x-axis)
plt.figure(figsize=(8, 5))
plt.plot(dims, sq_euclidean_stds, marker='o', label='Squared Euclidean (std)')
plt.plot(dims, manhattan_stds, marker='o', label='Manhattan (std)')
plt.xscale('log', base=2)
plt.xlabel('Dimension d')
plt.ylabel('Standard deviation of pairwise distance')
plt.title('Std Dev of Pairwise Distance vs Dimension (log scale)')
plt.legend()
plt.grid(True)
plt.savefig('q2_2_std_log.png', dpi=150)
plt.show()

# Plot: standard deviation of distance vs dimension (linear x-axis)
plt.figure(figsize=(8, 5))
plt.plot(dims, sq_euclidean_stds, marker='o', label='Squared Euclidean (std)')
plt.plot(dims, manhattan_stds, marker='o', label='Manhattan (std)')
plt.xlabel('Dimension d')
plt.ylabel('Standard deviation of pairwise distance')
plt.title('Std Dev of Pairwise Distance vs Dimension (linear scale)')
plt.legend()
plt.grid(True)
plt.savefig('q2_2_std_linear.png', dpi=150)
plt.show()