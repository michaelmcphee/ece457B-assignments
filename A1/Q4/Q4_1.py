import numpy as np

data = np.loadtxt('A1Q4-decision-trees.csv', delimiter=',', skiprows=1)
gender = data[:, 8]
survived = data[:, 9]

total_misclassified = 0

for gender_val, label in [(0, 'Female'), (1, 'Male')]:
    mask = (gender == gender_val)
    group_survived = survived[mask]

    n = len(group_survived)
    survivors = np.sum(group_survived == 1)
    non_survivors = np.sum(group_survived == 0)

    majority_pred = 1 if survivors > non_survivors else 0
    misclassified = min(survivors, non_survivors)
    total_misclassified += misclassified

    print(f"{label} (Gender={gender_val}): n={n}, survivors={survivors}, non_survivors={non_survivors}")
    print(f"  Majority prediction: {majority_pred}")
    print(f"  Misclassified: {misclassified}")

total = len(gender)
error_rate = total_misclassified / total

print(f"\nTotal misclassified: {total_misclassified}")
print(f"Total passengers: {total}")
print(f"Error rate: {error_rate:.4f}")
