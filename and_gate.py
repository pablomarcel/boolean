import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import _tree

# Example: Creating a sample dataset (Replace this with your actual data)
data = {
    'A': [0, 0, 1, 1],
    'B': [0, 1, 0, 1],
    'Output': [0, 0, 0, 1]
}
df = pd.DataFrame(data)

# Separating the inputs and the output
X = df.drop('Output', axis=1)  # Inputs
y = df['Output']  # Output

# Creating and training the decision tree classifier
clf = DecisionTreeClassifier(max_depth=3)  # Adjust the depth as needed
clf.fit(X, y)


# Function to generate boolean expression from the tree
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    paths = []

    def recurse(node, depth, path):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], depth + 1, path + [(name, "<=", threshold)])
            recurse(tree_.children_right[node], depth + 1, path + [(name, ">", threshold)])
        else:
            paths.append((path, tree_.value[node]))

    recurse(0, 1, [])

    # Convert paths to boolean expressions
    expressions = []
    for path, value in paths:
        if value[0][1] == 1:  # Only consider paths that lead to a True output
            expr = " and ".join([f"({f} {op} {t})" for f, op, t in path])
            expressions.append(expr)
    return " or ".join(expressions)


# Generating and printing the boolean expression
boolean_expression = tree_to_code(clf, X.columns)
print("Boolean Expression:", boolean_expression)

# Simplify and prettify the boolean expression for human readability
simplified_expression = boolean_expression.replace(" > 0.5", "")
simplified_expression = simplified_expression.replace(" and ", " ∧ ")  # Using the logical AND Unicode character
print("Simplified Boolean Expression:", simplified_expression)

# Visualizing the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=["False", "True"], rounded=True)
plt.show()
