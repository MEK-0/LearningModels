# Decision Trees on Iris Dataset with Gini and Entropy — Classification and Visualization
# A step-by-step implementation of Decision Tree classifiers using both Gini and Entropy criteria,
# including 2D decision boundaries and tree structure visualization.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the Iris dataset (only first two features for 2D visualization)
iris = load_iris()
X = iris.data[:, :2]
y = iris.target

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Decision Tree with Gini criterion
tree_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
tree_gini.fit(X_train_scaled, y_train)

# 5. Decision Tree with Entropy criterion
tree_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
tree_entropy.fit(X_train_scaled, y_train)

# 6. Decision boundary plot function
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF']))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF']))
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.title(title)
   

# 7. Plot decision boundaries
plot_decision_boundary(tree_gini, X_train_scaled, y_train, "Decision Tree - Gini")
plot_decision_boundary(tree_entropy, X_train_scaled, y_train, "Decision Tree - Entropy")

# 8. Visualize tree structures
plt.figure(figsize=(16, 8))
plot_tree(tree_gini, filled=True, feature_names=iris.feature_names[:2], class_names=iris.target_names, rounded=True)
plt.title("Decision Tree Structure - Gini")


plt.figure(figsize=(16, 8))
plot_tree(tree_entropy, filled=True, feature_names=iris.feature_names[:2], class_names=iris.target_names, rounded=True)
plt.title("Decision Tree Structure - Entropy")
plt.show()

# 9. Evaluation reports
print("Gini - Classification Report:\n", classification_report(y_test, tree_gini.predict(X_test_scaled)))
print("Entropy - Classification Report:\n", classification_report(y_test, tree_entropy.predict(X_test_scaled)))
