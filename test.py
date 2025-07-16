import os
import glob
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, cohen_kappa_score, confusion_matrix
)
from models import *
from dataset_loader import get_dataloaders
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="One-click Test Script")
parser.add_argument('--dataset', type=str, default='PneumoniaMNIST',
                    help='Dataset name (must match get_dataloaders)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for test loader')
parser.add_argument('--weight_dir', type=str, default='best_models',
                    help='Directory containing saved weights for each model')
parser.add_argument('--models', type=str, nargs='+',
                    default=[
                        'convnext_tiny', 'effnetv2_s', 'FasterNet', 'fasternet_s',
                        'mambaout_kobe', 'mambaout_tiny',
                        'mobilenetv4_conv_medium', 'mobilenetv4_conv_large',
                        'resnet18', 'resnet50',
                        'SepViT', 'SwinTransformer',
                        'VSSM', 'MedCTM_T', 'MedCTM_L'
                    ],
                    help='List of model names to test (must match models/__init__.py)')
args = parser.parse_args()

DATASET_NAME = args.dataset
BATCH_SIZE = args.batch_size
MODEL_WEIGHT_DIR = args.weight_dir
MODEL_NAMES = args.models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Config: DATASET={DATASET_NAME} | BATCH_SIZE={BATCH_SIZE} | WEIGHT_DIR={MODEL_WEIGHT_DIR}")
print(f"Models: {MODEL_NAMES}")

_, test_loader = get_dataloaders(
    dataset_name=DATASET_NAME,
    batch_size=BATCH_SIZE
)

num_classes = len(test_loader.dataset.classes) if hasattr(test_loader.dataset, 'classes') \
              else len(set([y for _, y in test_loader.dataset]))

def calculate_specificity(cm):
    specificity = 0.0
    for i in range(cm.shape[0]):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        denominator = tn + fp
        specificity += tn / denominator if denominator != 0 else 0.0
    return specificity / cm.shape[0]

def evaluate_model(model, test_loader):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    all_probs, all_preds, all_labels = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
            all_preds.append(torch.argmax(probs, dim=1).cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    avg_loss = total_loss / len(test_loader)
    oa = 100 * np.mean(all_preds == all_labels)
    precision = 100 * precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = 100 * recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = 100 * f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    specificity = 100 * calculate_specificity(cm)

    try:
        if num_classes == 2:
            auc_score = 100 * roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auc_score = 100 * roc_auc_score(
                all_labels, all_probs, multi_class='ovr', average='macro'
            )
    except Exception as e:
        print(f"AUC error: {e}")
        auc_score = 0.0

    kappa = 100 * cohen_kappa_score(all_labels, all_preds)

    return {
        "loss": avg_loss,
        "P": precision,
        "Se": recall,
        "Sp": specificity,
        "F1": f1,
        "OA": oa,
        "AUC": auc_score,
        "Kappa": kappa
    }

os.makedirs('test_result', exist_ok=True)
all_results = []

for model_name in MODEL_NAMES:
    try:
        model = eval(model_name)().to(DEVICE)
        weight_dir = os.path.join(MODEL_WEIGHT_DIR, model_name)
        weight_files = glob.glob(os.path.join(weight_dir, '*.pth'))

        if not weight_files:
            print(f" The weight file for {model_name} was not found, so it has been skipped.")
            continue

        checkpoint = torch.load(weight_files[0], map_location=DEVICE)
        if isinstance(checkpoint, dict) and ('state_dict' in checkpoint):
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict) and ('model' in checkpoint):
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

        metrics = evaluate_model(model, test_loader)

        save_dir = os.path.join('test_result', model_name)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
            for k, v in metrics.items():
                f.write(f"{k}: {v:.2f}\n")

        all_results.append({
            "Model": model_name, **metrics
        })

        print(f" {model_name:16} | "
              f"P: {metrics['P']:.1f} | Se: {metrics['Se']:.1f} | "
              f"Sp: {metrics['Sp']:.1f} | F1: {metrics['F1']:.1f} | "
              f"OA: {metrics['OA']:.1f} | AUC: {metrics['AUC']:.1f} | "
              f"Kappa: {metrics['Kappa']:.1f}")

    except Exception as e:
        print(f" {model_name} test failure: {e}")

if all_results:
    df = pd.DataFrame(all_results)
    csv_path = os.path.join('test_result', f"{DATASET_NAME}_test_result.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nAll results have been saved to: {csv_path}")

print("\nTest is completed")
