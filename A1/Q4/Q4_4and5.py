import numpy as np

data = np.loadtxt('A1Q4-decision-trees.csv', delimiter=',', skiprows=1)

age = data[:, 1]
gender = data[:, 8]
survived = data[:, 9]

N = len(survived)


def gini(labels):
    n = len(labels)

    if n == 0:
        return 0.0

    p0 = np.mean(labels == 0)
    p1 = np.mean(labels == 1)

    return 1 - p0**2 - p1**2


def entropy(labels):
    n = len(labels)

    if n == 0:
        return 0.0

    p0 = np.mean(labels == 0)
    p1 = np.mean(labels == 1)

    h = 0.0

    if p0 > 0:
        h -= p0 * np.log2(p0)

    if p1 > 0:
        h -= p1 * np.log2(p1)

    return h


def weighted_score(masks, impurity_fn):
    score = 0.0

    for mask in masks:
        labels = survived[mask]
        score += len(labels) / N * impurity_fn(labels)

    return score


# Q4.4 - Gini

gender_gini = weighted_score(
    [gender == 0, gender == 1],
    gini
)

age_gini = weighted_score(
    [age <= 65, age > 65],
    gini
)

print("=== Q4.4 Gini Impurity ===")
print(f"Gender split: {gender_gini:.4f}")
print(f"Age <= 65 split: {age_gini:.4f}")

if gender_gini < age_gini:
    print("Gender has lower weighted Gini impurity.\n")
else:
    print("Age has lower weighted Gini impurity.\n")


# Q4.5 - Entropy

gender_entropy = weighted_score(
    [gender == 0, gender == 1],
    entropy
)

age_entropy = weighted_score(
    [age <= 65, age > 65],
    entropy
)

print("=== Q4.5 Entropy ===")
print(f"Gender split: {gender_entropy:.4f}")
print(f"Age <= 65 split: {age_entropy:.4f}")

if gender_entropy < age_entropy:
    print("Gender has lower weighted entropy.")
else:
    print("Age has lower weighted entropy.")