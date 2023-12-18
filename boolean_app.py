import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

# streamlit run boolean_app.py


# Function to train the decision tree and generate a boolean expression
def train_and_generate_expression(data):
    df = pd.DataFrame(data)
    X = df.drop('Output', axis=1)  # Inputs
    y = df['Output']  # Output
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)

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

        expressions = []
        for path, value in paths:
            if value[0][1] == 1:  # Only consider paths that lead to a True output
                expr = " and ".join([f"({f} {op} {t})" for f, op, t in path])
                expressions.append(expr)
        return " or ".join(expressions)

    boolean_expression = tree_to_code(clf, X.columns)
    simplified_expression = boolean_expression.replace(" > 0.5", "")
    simplified_expression = simplified_expression.replace(" and ", " ∧ ")  # Using the logical AND Unicode character
    return simplified_expression


# Streamlit interface
st.title('AI Logix')
st.title('Boolean Expression Simplifier')

# Input fields to dynamically create the truth table
st.write('Enter truth table data:')
table_data = {
    'A': st.text_input('Input for A (comma-separated, e.g., 0,0,1,1):', '0,0,1,1'),
    'B': st.text_input('Input for B (comma-separated, e.g., 0,1,0,1):', '0,1,0,1'),
    'Output': st.text_input('Output (comma-separated, e.g., 0,0,0,1):', '0,0,0,1')
}

# Button to perform training and simplification
if st.button('Generate Simplified Boolean Expression'):
    try:
        # Prepare the data for training
        prepared_data = {k: [int(i) for i in v.split(',')] for k, v in table_data.items()}
        # Train the model and generate expression
        expression = train_and_generate_expression(prepared_data)
        # Display the result
        st.success(f'The Boolean expression is: {expression}')
    except Exception as e:
        st.error(f'An error occurred: {e}')

# Display a placeholder where the fixed Boolean expression can be shown if needed
# st.write('The Boolean expression for an AND gate is: (A) ∧ (B)')
