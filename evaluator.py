import time
import matplotlib
matplotlib.use('Agg') # Disables the interactive GUI to prevent Tkinter crashes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test, model_name):
    """
    Predicts on the test set, calculates accuracy and prediction runtime,
    and returns the predictions, accuracy, and execution time.
    """
    print(f"\n--- Evaluating {model_name} ---")
    
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Prediction Runtime: {pred_time:.4f} seconds.")
    print(f"Accuracy: {accuracy:.4f} ({(accuracy * 100):.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    return y_pred, accuracy, pred_time

def plot_confusion_matrices(results, runtimes, y_test):
    """
    Plots confusion matrices for all models side by side, 
    including training and prediction times in the titles, 
    and saves the figure as an image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Confusion Matrices Comparison (with Runtimes)', fontsize=16)
    
    for idx, (model_name, y_pred) in enumerate(results.items()):
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Predicted Neg', 'Predicted Pos'],
                    yticklabels=['Actual Neg', 'Actual Pos'])
        
        # Adding runtimes to the subplot title
        t_train = runtimes[model_name]['train']
        t_pred = runtimes[model_name]['pred']
        axes[idx].set_title(f"{model_name}\nTrain: {t_train:.3f}s | Pred: {t_pred:.3f}s")
    
    plt.tight_layout()
    plt.savefig('confusion_matrices_with_runtimes.png')
    print("\nConfusion matrices saved as 'confusion_matrices_with_runtimes.png' in the project folder.")