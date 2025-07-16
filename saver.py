# saver.py
import os
import torch
import csv

class ModelSaver:
    def __init__(self, model_name):
        self.best_metrics = {
            'OA': 0.0, 'AUC': 0.0, 'F1': 0.0, 'P': 0.0, 'Se': 0.0, 'Sp': 0.0, 'Kappa': 0.0
        }
        self.save_dir = os.path.join('best_models', model_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def check_and_save(self, net, current_metrics):
        for metric in ['OA', 'AUC', 'F1', 'Kappa']:
            if current_metrics[metric] > self.best_metrics[metric]:
                self.best_metrics[metric] = current_metrics[metric]
                torch.save(
                    net.state_dict(),
                    os.path.join(self.save_dir, f'best_{metric}.pth')
                )
        torch.save(
            net.state_dict(),
            os.path.join(self.save_dir, 'final_model.pth')
        )

def save_metrics_to_csv(metrics, file_path):
    fieldnames = ['epoch'] + list(metrics.keys())
    epochs = range(1, len(next(iter(metrics.values()))) + 1)

    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in epochs:
            row = {'epoch': epoch}
            for key in metrics.keys():
                row[key] = metrics[key][epoch-1]
            writer.writerow(row)
