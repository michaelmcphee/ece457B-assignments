import numpy as np

data = np.loadtxt('A1Q4-decision-trees.csv', delimiter=',', skiprows=1)
gender = data[:, 8]
age = data[:, 1]
survived = data[:, 9]


def report_group(mask, label):
    group_survived = survived[mask]
    n = len(group_survived)
    survivors = np.sum(group_survived == 1)
    non_survivors = np.sum(group_survived == 0)
    if n == 0:
        print(f"{label}: n=0 (empty group)")
        return 0
    majority_pred = 1 if survivors > non_survivors else 0
    misclassified = min(survivors, non_survivors)
    print(f"{label}: n={n}, survivors={survivors}, non_survivors={non_survivors}, "
          f"majority_pred={majority_pred}, misclassified={misclassified}")
    return misclassified


print("=== Split 1: Gender first, then Age <=65 vs >65 ===")
total_err_1 = 0
for gender_val, gender_label in [(0, 'Female'), (1, 'Male')]:
    gender_mask = (gender == gender_val)
    for age_cond, age_label in [(age <= 65, 'Age<=65'), (age > 65, 'Age>65')]:
        mask = gender_mask & age_cond
        total_err_1 += report_group(mask, f"{gender_label}, {age_label}")

total = len(gender)
print(f"Total error count: {total_err_1}, error rate: {total_err_1/total:.4f}\n")


print("=== Split 2 (reversed): Age <=65 vs >65 first, then Gender ===")
total_err_2 = 0
for age_cond, age_label in [(age <= 65, 'Age<=65'), (age > 65, 'Age>65')]:
    age_mask = age_cond
    for gender_val, gender_label in [(0, 'Female'), (1, 'Male')]:
        mask = age_mask & (gender == gender_val)
        total_err_2 += report_group(mask, f"{age_label}, {gender_label}")

print(f"Total error count: {total_err_2}, error rate: {total_err_2/total:.4f}")