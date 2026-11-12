import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def euclidean_distance(x, y):
    """
    Euclidean distance between two vectors x and y.
    d(x, y) = sqrt(sum((x_i - y_i)^2))
    """
    squared_diffs = (x - y) ** 2
    sum_squared_diffs = np.sum(squared_diffs)
    return np.sqrt(sum_squared_diffs)


def cosine_similarity(x, y):
    """
    Cosine similarity between two vectors x and y.
    cos(theta) = (x . y) / (||x|| * ||y||)
    """
    dot_product = np.sum(x * y)
    norm_x = np.sqrt(np.sum(x ** 2))
    norm_y = np.sqrt(np.sum(y ** 2))
    return dot_product / (norm_x * norm_y)


def knn_predict_one(x_test, X_train, y_train, k, metric):
    """
    Predict the label of a single test point x_test using KNN.
    """
    distances = []
    for i in range(len(X_train)):
        if metric == 'euclidean':
            dist = euclidean_distance(x_test, X_train[i])
        elif metric == 'cosine':
            dist = -cosine_similarity(x_test, X_train[i])
        else:
            raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")
        distances.append(dist)

    distances = np.array(distances)
    nearest_indices = np.argsort(distances)[:k]
    nearest_labels = y_train[nearest_indices]

    labels, counts = np.unique(nearest_labels, return_counts=True)
    prediction = labels[np.argmax(counts)]
    return prediction


def knn_predict_all(X_test, X_train, y_train, k, metric):
    """
    Predict labels for every point in X_test.
    """
    predictions = []
    for x_test in X_test:
        pred = knn_predict_one(x_test, X_train, y_train, k, metric)
        predictions.append(pred)
    return np.array(predictions)


def error_rate(y_true, y_pred):
    """
    Fraction of predictions that do NOT match the true labels.
    """
    return np.mean(y_true != y_pred)


def split_data(X, y, train_frac, val_frac, test_frac, random_state=0):
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=train_frac, random_state=random_state
    )
    val_frac_of_rest = val_frac / (val_frac + test_frac)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, train_size=val_frac_of_rest, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---- Load data ----
data = np.loadtxt('A1Q4NearestNeighbors.csv', delimiter=',')
X = data[:, :3]
Y = data[:, 3]

# ---- Settings to sweep over ----
splits = {
    '80/10/10': (0.8, 0.1, 0.1),
    '34/33/33': (0.34, 0.33, 0.33),
    '25/25/50': (0.25, 0.25, 0.5),
}
metrics = ['euclidean', 'cosine']
k_values = [1, 3, 5, 11]

results = []

for split_name, (train_frac, val_frac, test_frac) in splits.items():
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, Y, train_frac, val_frac, test_frac
    )

    for metric in metrics:
        for k in k_values:
            y_pred_train = knn_predict_all(X_train, X_train, y_train, k, metric)
            y_pred_val = knn_predict_all(X_val, X_train, y_train, k, metric)
            y_pred_test = knn_predict_all(X_test, X_train, y_train, k, metric)

            results.append({
                'split': split_name,
                'metric': metric,
                'k': k,
                'train_err': error_rate(y_train, y_pred_train),
                'val_err': error_rate(y_val, y_pred_val),
                'test_err': error_rate(y_test, y_pred_test),
            })

# ---- Print full results table ----
print(f"{'split':10s} | {'metric':10s} | {'k':3s} | {'train_err':10s} | {'val_err':10s} | {'test_err':10s}")
for r in results:
    print(f"{r['split']:10s} | {r['metric']:10s} | {r['k']:<3d} | "
          f"{r['train_err']:.4f}     | {r['val_err']:.4f}     | {r['test_err']:.4f}")

# ---- Plot: validation error vs k, one line per (split, metric) combination ----
plt.figure(figsize=(9, 6))
for split_name in splits:
    for metric in metrics:
        subset = [r for r in results if r['split'] == split_name and r['metric'] == metric]
        subset_sorted = sorted(subset, key=lambda r: r['k'])
        ks = [r['k'] for r in subset_sorted]
        val_errs = [r['val_err'] for r in subset_sorted]
        plt.plot(ks, val_errs, marker='o', label=f"{split_name}, {metric}")

plt.xlabel('k (number of neighbors)')
plt.ylabel('Validation error rate')
plt.title('Validation Error vs k, across splits and metrics')
plt.legend()
plt.grid(True)
plt.savefig('q3_3_val_error.png', dpi=150)
plt.show()

# ---- Plot: train vs val vs test error vs k, for each (split, metric) combination ----
for split_name in splits:
    for metric in metrics:
        subset = [r for r in results if r['split'] == split_name and r['metric'] == metric]
        subset_sorted = sorted(subset, key=lambda r: r['k'])
        ks = [r['k'] for r in subset_sorted]
        train_errs = [r['train_err'] for r in subset_sorted]
        val_errs = [r['val_err'] for r in subset_sorted]
        test_errs = [r['test_err'] for r in subset_sorted]

        plt.figure(figsize=(7, 5))
        plt.plot(ks, train_errs, marker='o', label='Train error')
        plt.plot(ks, val_errs, marker='o', label='Validation error')
        plt.plot(ks, test_errs, marker='o', label='Test error')
        plt.xlabel('k (number of neighbors)')
        plt.ylabel('Error rate')
        plt.title(f'Train/Val/Test Error vs k\n(split={split_name}, metric={metric})')
        plt.legend()
        plt.grid(True)
        safe_name = f"q3_3_{split_name.replace('/', '-')}_{metric}.png"
        plt.savefig(safe_name, dpi=150)
        plt.show()

# ---- Find best combination (lowest validation error) ----
best = min(results, key=lambda r: r['val_err'])
print("\nBest combination by validation error:")
print(f"  split={best['split']}, metric={best['metric']}, k={best['k']}, "
      f"val_err={best['val_err']:.4f}, test_err={best['test_err']:.4f}")
