import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = np.loadtxt('A1Q4-decision-trees.csv', delimiter=',', skiprows=1)

X = data[:, :-1]
y = data[:, -1]

# Split ratios to test
ratios = [
    (0.80, 0.10, 0.10),
    (0.34, 0.33, 0.33),
    (0.25, 0.25, 0.50)
]

for train_ratio, val_ratio, test_ratio in ratios:

    # First split off validation + test data
    temp_ratio = val_ratio + test_ratio

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_ratio,
        random_state=123123
    )

    # Split remaining data into validation and test
    test_fraction_of_temp = test_ratio / temp_ratio

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_fraction_of_temp,
        random_state=123123
    )

    print("\n========================================")
    print(f"Split: {int(train_ratio*100)}/{int(val_ratio*100)}/{int(test_ratio*100)}")
    print("========================================")

    print(f"Train size: {len(y_train)}")
    print(f"Validation size: {len(y_val)}")
    print(f"Test size: {len(y_test)}")

    for criterion in ["gini", "entropy"]:

        # Train decision tree
        tree = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=3,
            random_state=123123
        )

        tree.fit(X_train, y_train)

        # Predictions
        train_pred = tree.predict(X_train)
        val_pred = tree.predict(X_val)
        test_pred = tree.predict(X_test)

        # Accuracy
        train_accuracy = accuracy_score(y_train, train_pred)
        val_accuracy = accuracy_score(y_val, val_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        print(f"\nCriterion: {criterion}")
        print(f"Train accuracy:      {train_accuracy:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Test accuracy:       {test_accuracy:.4f}")