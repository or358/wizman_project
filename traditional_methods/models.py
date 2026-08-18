import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

def get_logistic_regression():
    """Returns a Logistic Regression model with default regularization."""
    return LogisticRegression(C=1.0, max_iter=1000, random_state=42)

def get_svm():
    """Returns a Linear SVM model optimized for text classification."""
    return LinearSVC(C=1.0, random_state=42)

def get_decision_tree():
    """Returns a Decision Tree classifier with limited depth to prevent overfitting."""
    return DecisionTreeClassifier(max_depth=20, random_state=42)

def train_model(model, X_train, y_train):
    """Trains the given model on the provided training data."""
    print(f"Training {model.__class__.__name__}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time

    print(f"Training completed. Time taken: {train_time:.4f} seconds.")   
    return model, train_time