# deep_learning_methods/trainer.py

import os
import time
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, learning_rate, device, experiment_name):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.experiment_name = experiment_name
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.epoch_data = []
        
    def calculate_accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, 1)
        corrects = torch.sum(preds == labels).item()
        return corrects / len(labels)
        
    def run_epoch(self, dataloader, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()
            
        total_loss = 0.0
        total_acc = 0.0
        
        with torch.set_grad_enabled(is_train):
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                total_loss += loss.item() * inputs.size(0)
                total_acc += self.calculate_accuracy(outputs, labels) * inputs.size(0)
                
        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = total_acc / len(dataloader.dataset)
        return epoch_loss, epoch_acc
        
    def train(self, num_epochs=5):
        best_eval_acc = 0.0
        best_eval_epoch = 0
        best_test_acc = 0.0
        best_test_epoch = 0
        best_induced_test_acc = 0.0
        
        # הדפסת הטבלה המדויקת שנדרשה במסמך
        print(f"{'Epoch':<7} | {'Train Loss':<11} | {'Train Acc':<10} | {'Eval Acc':<9} | {'Test Acc':<9} | {'Epoch Time':<11} | {'Best Induced Test'}")
        print("-" * 85)
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            train_loss, train_acc = self.run_epoch(self.train_loader, is_train=True)
            _, eval_acc = self.run_epoch(self.val_loader, is_train=False)
            _, test_acc = self.run_epoch(self.test_loader, is_train=False)
            
            epoch_time = time.time() - start_time
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_epoch = epoch
                
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_eval_epoch = epoch
                best_induced_test_acc = test_acc
                
            print(f"{epoch:<7} | {train_loss:<11.4f} | {train_acc:<10.4f} | {eval_acc:<9.4f} | {test_acc:<9.4f} | {epoch_time:<9.2f}s | {best_induced_test_acc:.4f}")
            
            self.epoch_data.append({
                "Epoch": epoch,
                "Train Loss": round(train_loss, 4),
                "Train Acc": round(train_acc, 4),
                "Eval Acc": round(eval_acc, 4),
                "Test Acc": round(test_acc, 4),
                "Epoch Time (s)": round(epoch_time, 2),
                "Best Induced Test": round(best_induced_test_acc, 4)
            })
            
        print("-" * 85)
        print("Training Summary:")
        print(f"Best eval results:          {best_eval_acc:.4f} (Obtained on Epoch {best_eval_epoch})")
        print(f"Best test results:          {best_test_acc:.4f} (Obtained on Epoch {best_test_epoch})")
        print(f"Best induced test results:  {best_induced_test_acc:.4f}")
        
        self.save_results_and_graphs(best_eval_acc, best_test_acc, best_induced_test_acc)
        
    def save_results_and_graphs(self, best_eval, best_test, best_induced):
        os.makedirs("results", exist_ok=True)
        
        # 1. שמירת המידע המפורט לכל אפוק (לגיבוי ולבדיקות עומק)
        df_epochs = pd.DataFrame(self.epoch_data)
        df_epochs.to_csv(f"results/{self.experiment_name}_epochs.csv", index=False)
        
        # 2. חילוץ חכם של הפרמטרים מתוך שם הניסוי כדי לבנות עמודות מושלמות למצגת
        dataset_name = 'rotten_tomatoes' if 'rotten_tomatoes' in self.experiment_name else 'imdb'
        model = 'CNN' if 'CNN' in self.experiment_name else 'LSTM'
        emb = 'Word2Vec' if 'Word2Vec' in self.experiment_name else 'GloVe' if 'GloVe' in self.experiment_name else 'Random'
        freeze = 'True' if 'FreezeTrue' in self.experiment_name else 'False'
        
        # חישוב זמן האימון הממוצע לאפוק (דרישה מהמסמך)
        avg_time = round(df_epochs["Epoch Time (s)"].mean(), 2)
        
        # יצירת השורה לטבלת המאסטר לפי הדרישות המדויקות
        summary_data = [{
            "Dataset": dataset_name,
            "Method": model,
            "Embedding": emb,
            "Freeze": freeze,
            "Train Time per Epoch (s)": avg_time,
            "Accuracy on Val": round(best_eval, 4),
            "Accuracy on Test": round(best_test, 4),
            "Best Induced Test": round(best_induced, 4)
        }]
        
        df_summary = pd.DataFrame(summary_data)
        master_file = "results/presentation_master_results.csv"
        
        if os.path.exists(master_file):
            df_summary.to_csv(master_file, mode='a', header=False, index=False)
        else:
            df_summary.to_csv(master_file, index=False)
            
        # 3. יצירת גרפים מותאמים למצגת
        self.generate_graphs()
        
    def generate_graphs(self):
        epochs = [d["Epoch"] for d in self.epoch_data]
        train_acc = [d["Train Acc"] for d in self.epoch_data]
        eval_acc = [d["Eval Acc"] for d in self.epoch_data]
        train_loss = [d["Train Loss"] for d in self.epoch_data]
        
        # שימוש בעיצוב יפה וברור למצגות
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # כותרת ראשית שמתארת את הניסוי
        clean_title = self.experiment_name.replace('_', ' | ')
        fig.suptitle(f"Experiment Progress: {clean_title}", fontsize=14, fontweight='bold')
        
        # גרף 1: Accuracy (Train vs Validation)
        ax1.plot(epochs, train_acc, label='Train Accuracy', marker='o', linewidth=2, color='#1f77b4')
        ax1.plot(epochs, eval_acc, label='Validation Accuracy', marker='s', linewidth=2, color='#ff7f0e')
        ax1.set_title('Accuracy over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.set_xticks(epochs)
        ax1.legend()
        
        # גרף 2: Train Loss
        ax2.plot(epochs, train_loss, label='Train Loss', marker='x', linewidth=2, color='#d62728')
        ax2.set_title('Loss over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('CrossEntropy Loss')
        ax2.set_xticks(epochs)
        ax2.legend()
        
        plt.tight_layout()
        # שמירה ברזולוציה גבוהה (dpi=300) כדי שלא יטשטש במצגת
        plt.savefig(f"results/{self.experiment_name}_graph.png", dpi=300)
        plt.close()