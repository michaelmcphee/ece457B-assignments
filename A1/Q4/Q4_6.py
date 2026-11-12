import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load data
data = np.loadtxt('A1Q4-decision-trees.csv', delimiter=',', skiprows=1)

X = data[:, :-1]
y = data[:, -1]

feature_names = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Title",
    "FamilySize",
    "Gender"
]

# ------------------------------------------------------------
# Part 1: Default tree
# ------------------------------------------------------------

tree_default = DecisionTreeClassifier(
    max_depth=3,
    random_state=123123
)

tree_default.fit(X, y)

plt.figure(figsize=(16, 9))

plot_tree(
    tree_default,
    feature_names=feature_names,
    class_names=["Died", "Survived"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree - Default Criterion (Gini)")
plt.tight_layout()
plt.show()


# Print the features used at each internal node

print("=== Default tree ===")

for i in range(tree_default.tree_.node_count):

    if tree_default.tree_.children_left[i] != -1:

        feature = tree_default.tree_.feature[i]
        threshold = tree_default.tree_.threshold[i]

        print(
            f"Node {i}: {feature_names[feature]} "
            f"<= {threshold:.3f}"
        )


# ------------------------------------------------------------
# Part 2: Entropy
# ------------------------------------------------------------

tree_entropy = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=123123
)

tree_entropy.fit(X, y)

plt.figure(figsize=(16, 9))

plot_tree(
    tree_entropy,
    feature_names=feature_names,
    class_names=["Died", "Survived"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree - Entropy")
plt.tight_layout()
plt.show()


print("\n=== Entropy tree ===")

for i in range(tree_entropy.tree_.node_count):

    if tree_entropy.tree_.children_left[i] != -1:

        feature = tree_entropy.tree_.feature[i]
        threshold = tree_entropy.tree_.threshold[i]

        print(
            f"Node {i}: {feature_names[feature]} "
            f"<= {threshold:.3f}"
        )


# ------------------------------------------------------------
# Part 3: Random splitter
# ------------------------------------------------------------

tree_random = DecisionTreeClassifier(
    splitter="random",
    max_depth=3,
    random_state=123123
)

tree_random.fit(X, y)

plt.figure(figsize=(16, 9))

plot_tree(
    tree_random,
    feature_names=feature_names,
    class_names=["Died", "Survived"],
    filled=True,
    rounded=True
)

plt.title("Decision Tree - Random Splitter")
plt.tight_layout()
plt.show()


print("\n=== Random splitter tree ===")

for i in range(tree_random.tree_.node_count):

    if tree_random.tree_.children_left[i] != -1:

        feature = tree_random.tree_.feature[i]
        threshold = tree_random.tree_.threshold[i]

        print(
            f"Node {i}: {feature_names[feature]} "
            f"<= {threshold:.3f}"
        )