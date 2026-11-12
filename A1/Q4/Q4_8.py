import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = np.loadtxt('A1Q4-decision-trees.csv', delimiter=',', skiprows=1)

X = data[:, :-1]
y = data[:, -1]

# 80/10/10 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.20,
    random_state=123123
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=123123
)

# Test different maximum depths
depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]

print("Max Depth | Train Accuracy | Validation Accuracy | Test Accuracy")
print("---------------------------------------------------------------")

for depth in depths:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=123123
    )

    tree.fit(X_train, y_train)

    train_accuracy = accuracy_score(
        y_train, tree.predict(X_train)
    )

    val_accuracy = accuracy_score(
        y_val, tree.predict(X_val)
    )

    test_accuracy = accuracy_score(
        y_test, tree.predict(X_test)
    )

    print(
        f"{depth:^9} | "
        f"{train_accuracy:^14.4f} | "
        f"{val_accuracy:^19.4f} | "
        f"{test_accuracy:^14.4f}"
    )