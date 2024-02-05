# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model
model = LogisticRegression(solver="liblinear", max_iter=200)

# Setup hyperparameters to tune
param_grid = {
    "C": [0.1, 1, 10, 100],  # Regularization strength
    "penalty": ["l1", "l2"],  # Norm used in the penalization
}

# Setup GridSearchCV
grid_search = GridSearchCV(
    estimator=model, param_grid=param_grid, cv=5, scoring="accuracy"
)

# Perform hyperparameter tuning
grid_search.fit(X_train, y_train)

# Identify best performing model configuration
best_model = grid_search.best_estimator_

# Predictions using the best model
predictions = best_model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)
print(f"Best model accuracy: {accuracy:.4f}")
