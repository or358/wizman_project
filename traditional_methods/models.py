import time
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import config

def get_logistic_regression():
    """Returns a Logistic Regression model initialized with config parameters."""
    return LogisticRegression(
        C=config.LR_C, 
        max_iter=config.LR_MAX_ITER, 
        random_state=config.RANDOM_STATE
    )

def get_svm():
    """Returns a Linear SVM model initialized with config parameters."""
    return LinearSVC(
        C=config.SVM_C, 
        random_state=config.RANDOM_STATE
    )

def get_decision_tree():
    """Returns a Decision Tree classifier initialized with config parameters."""
    return DecisionTreeClassifier(
        max_depth=config.DT_MAX_DEPTH, 
        random_state=config.RANDOM_STATE
    )

def train_model(model, X_train, y_train):
    """
    Trains the given model and calculates the training runtime.
    """
    print(f"Training {model.__class__.__name__}...")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"Training complete. Runtime: {train_time:.4f} seconds.")
    return model, train_time