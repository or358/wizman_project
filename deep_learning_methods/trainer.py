# deep_learning_methods/trainer.py

import time
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

class ModelTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                 test_loader: DataLoader, learning_rate: float, device: str = 'cpu',
                 experiment_name: str = "experiment"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.experiment_name = experiment_name
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.best_eval_acc = 0.0
        self.best_eval_epoch = 0
        self.best_test_acc = 0.0
        self.best_test_epoch = 0
        self.best_induced_test_acc = 0.0
        
        self.epoch_results_data = []

    def calculate_accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, 1)
        corrects = torch.sum(preds == labels).item()
        return corrects / len(labels)

    def _run_epoch(self, dataloader, is_train: bool):
        if is_train:
            self.model.train()
        else:
            self.model.eval()
            
        total_loss = 0.0
        total_acc = 0.0
        
        with torch.set_grad_enabled(is_train):
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).long() 
                
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

    def save_results_to_csv(self):
        os.makedirs("results", exist_ok=True)
        
        df_epochs = pd.DataFrame(self.epoch_results_data)
        epochs_filename = f"results/{self.experiment_name}_epochs.csv"
        df_epochs.to_csv(epochs_filename, index=False)
        
        summary_data = [{
            "Experiment": self.experiment_name,
            "Best Eval Acc": self.best_eval_acc,
            "Best Eval Epoch": self.best_eval_epoch,
            "Best Test Acc": self.best_test_acc,
            "Best Test Epoch": self.best_test_epoch,
            "Best Induced Test Acc": self.best_induced_test_acc
        }]
        df_summary = pd.DataFrame(summary_data)
        
        master_summary_file = "results/master_summary_results.csv"
        if os.path.exists(master_summary_file):
            df_summary.to_csv(master_summary_file, mode='a', header=False, index=False)
        else:
            df_summary.to_csv(master_summary_file, index=False)

    def save_graphs(self):
        """Generates and saves Training curves (Accuracy and Loss) as an image."""
        os.makedirs("results", exist_ok=True)
        
        epochs = [d["Epoch"] for d in self.epoch_results_data]
        train_acc = [d["Train Acc"] for d in self.epoch_results_data]
        eval_acc = [d["Eval Acc"] for d in self.epoch_results_data]
        train_loss = [d["Train Loss"] for d in self.epoch_results_data]
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
        plt.plot(epochs, eval_acc, label='Eval Accuracy', marker='o')
        plt.title(f'Accuracy: {self.experiment_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_loss, label='Train Loss', color='red', marker='o')
        plt.title(f'Loss: {self.experiment_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save figure
        graph_filename = f"results/{self.experiment_name}_graph.png"
        plt.tight_layout()
        plt.savefig(graph_filename)
        plt.close() # Close to free up memory

    def train(self, num_epochs: int):
        print("\nStarting Training...")
        print(f"{'Epoch':<7} | {'Train Loss':<11} | {'Train Acc':<10} | {'Eval Acc':<9} | {'Test Acc':<9} | {'Epoch Time':<11} | {'Best Induced Test'}")
        print("-" * 85)
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            train_loss, train_acc = self._run_epoch(self.train_loader, is_train=True)
            _, eval_acc = self._run_epoch(self.val_loader, is_train=False)
            _, test_acc = self._run_epoch(self.test_loader, is_train=False)
            
            epoch_time = time.time() - start_time
            
            if test_acc > self.best_test_acc:
                self.best_test_acc = test_acc
                self.best_test_epoch = epoch
                
            if eval_acc > self.best_eval_acc:
                self.best_eval_acc = eval_acc
                self.best_eval_epoch = epoch
                self.best_induced_test_acc = test_acc 
                
            print(f"{epoch:<7} | {train_loss:<11.4f} | {train_acc:<10.4f} | {eval_acc:<9.4f} | {test_acc:<9.4f} | {epoch_time:<9.2f}s | {self.best_induced_test_acc:.4f}")
            
            self.epoch_results_data.append({
                "Epoch": epoch,
                "Train Loss": round(train_loss, 4),
                "Train Acc": round(train_acc, 4),
                "Eval Acc": round(eval_acc, 4),
                "Test Acc": round(test_acc, 4),
                "Epoch Time (s)": round(epoch_time, 2),
                "Best Induced Test": round(self.best_induced_test_acc, 4)
            })

        print("-" * 85)
        print("Training Summary:")
        print(f"Best eval results:          {self.best_eval_acc:.4f} (Obtained on Epoch {self.best_eval_epoch})")
        print(f"Best test results:          {self.best_test_acc:.4f} (Obtained on Epoch {self.best_test_epoch})")
        print(f"Best induced test results:  {self.best_induced_test_acc:.4f}")
        
        # Save CSVs and Graphs
        self.save_results_to_csv()
        self.save_graphs()