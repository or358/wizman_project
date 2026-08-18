from data_loader import load_data
from preprocessor import preprocess_and_split
import models
from evaluator import evaluate_model, plot_confusion_matrices

def main():
    """
    Main pipeline executing data loading, preprocessing, model training, 
    and evaluation, while tracking runtimes for visualization.
    """
    print("=== Movie Review Sentiment Analysis Project ===")
    
    print("\n[Step 1: Loading Data]")
    df = load_data()
    
    print("\n[Step 2: Preprocessing and Splitting]")
    X_train, X_test, y_train, y_test, vectorizer = preprocess_and_split(df)
    
    print("\n[Step 3: Initializing and Training Models]")
    classifiers = {
        "Logistic Regression": models.get_logistic_regression(),
        "Linear SVM": models.get_svm(),
        "Decision Tree": models.get_decision_tree()
    }
    
    trained_models = {}
    runtimes = {} # Dictionary to store training and prediction times
    
    for name, clf in classifiers.items():
        # Unpack both the model and its training time
        model, train_time = models.train_model(clf, X_train, y_train)
        trained_models[name] = model
        runtimes[name] = {'train': train_time}

    print("\n[Step 4: Evaluating Models on Test Set]")
    predictions = {}
    
    for name, clf in trained_models.items():
        # Unpack predictions, accuracy, and prediction time
        y_pred, accuracy, pred_time = evaluate_model(clf, X_test, y_test, name)
        predictions[name] = y_pred
        runtimes[name]['pred'] = pred_time

    print("\n[Step 5: Generating Visualizations]")
    plot_confusion_matrices(predictions, runtimes, y_test)
    
    print("\n=== Project Execution Completed Successfully ===")

if __name__ == "__main__":
    main()