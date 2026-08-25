import os
import pandas as pd
from data_loader import load_data
from preprocessor import preprocess_and_split
import models
from evaluator import evaluate_model, plot_confusion_matrices
import config

def main():
    """
    Main pipeline executing data loading, preprocessing, model training, and evaluation.
    """
    print(f"=== Sentiment Analysis Project ({config.DATASET_NAME}) ===")
    
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
    runtimes = {}
    
    for name, clf in classifiers.items():
        model, train_time = models.train_model(clf, X_train, y_train)
        trained_models[name] = model
        runtimes[name] = {'train': train_time}

    print("\n[Step 4: Evaluating Models on Test Set]")
    predictions = {}
    accuracies = {}
    
    for name, clf in trained_models.items():
        y_pred, accuracy, pred_time = evaluate_model(clf, X_test, y_test, name)
        predictions[name] = y_pred
        accuracies[name] = accuracy
        runtimes[name]['pred'] = pred_time

    print("\n[Step 5: Generating Visualizations and Saving Results]")
   
    output_dir = "traditional_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*85)
    print(f"RESULTS TABLE: {config.DATASET_NAME.upper()}")
    print("="*85)
    print(f"{'Method':<20} | {'Main Parameters':<25} | {'Train Time':<15} | {'Test Accuracy':<15}")
    print("-" * 85)
    
    params_map = {
        "Logistic Regression": f"TF-IDF ({config.MAX_FEATURES}), C={config.LR_C}",
        "Linear SVM": f"TF-IDF ({config.MAX_FEATURES}), C={config.SVM_C}",
        "Decision Tree": f"TF-IDF ({config.MAX_FEATURES}), max_depth={config.DT_MAX_DEPTH}"
    }
    
    results_data = []
    
    for name in trained_models.keys():
        t_train_raw = runtimes[name]['train']
        acc_raw = accuracies[name]
        
        t_train_str = f"{t_train_raw:.4f}s"
        acc_str = f"{(acc_raw * 100):.2f}%"
        print(f"{name:<20} | {params_map[name]:<25} | {t_train_str:<15} | {acc_str:<15}")
        
        results_data.append({
            "Method": name,
            "Main Parameters": params_map[name],
            "Train Time (s)": round(t_train_raw, 4),
            "Test Accuracy": round(acc_raw, 4)
        })
    print("="*85)

    df_results = pd.DataFrame(results_data)
    csv_path = os.path.join(output_dir, "traditional_summary_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nResults successfully saved to '{csv_path}'.")

    plot_confusion_matrices(predictions, runtimes, y_test, output_dir)
    
    print("\n=== Project Execution Completed Successfully ===")

if __name__ == "__main__":
    main()