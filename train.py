import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models import *
from dataset_loader import get_dataloaders
from saver import ModelSaver, save_metrics_to_csv
from trainer import train, val
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="One-click Train Script")

parser.add_argument('--model', type=str, default='MedCTM_T',
                    help='Model name, must match models/__init__.py')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--dataset', type=str, default='PneumoniaMNIST',
                    help='Dataset name')

args = parser.parse_args()

MODEL_NAME = args.model
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
DATASET_NAME = args.dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Config: MODEL={MODEL_NAME} | EPOCHS={NUM_EPOCHS} | BATCH_SIZE={BATCH_SIZE} | LR={LR} | DATASET={DATASET_NAME}")

train_loader, test_loader = get_dataloaders(
    dataset_name=DATASET_NAME,
    batch_size=BATCH_SIZE
)

net = eval(MODEL_NAME)().to(DEVICE)

optimizer = optim.Adam(net.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

model_saver = ModelSaver(MODEL_NAME)
train_metrics = {k: [] for k in ['loss', 'P', 'Se', 'Sp', 'F1', 'OA', 'Kappa']}
val_metrics = {k: [] for k in ['loss', 'P', 'Se', 'Sp', 'F1', 'OA', 'AUC', 'Kappa']}

for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    train(epoch, net, optimizer, criterion, train_loader, train_metrics)
    val(epoch, net, criterion, test_loader, val_metrics, model_saver)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} done | Time: {time.time() - start_time:.1f}s")

os.makedirs(MODEL_NAME, exist_ok=True)
save_metrics_to_csv(train_metrics, os.path.join(MODEL_NAME, 'train_metrics.csv'))
save_metrics_to_csv(val_metrics, os.path.join(MODEL_NAME, 'val_metrics.csv'))

print(f"Training Finished. Logs saved in: {MODEL_NAME}/")
