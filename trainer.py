# trainer.py
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score
)
import torch.nn.functional as F
from metrics import calculate_specificity, calculate_auc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(epoch, net, optimizer, criterion, train_loader, train_metrics):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0
    targets_all, predicted_all = [], []

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        targets_all.extend(targets.cpu().numpy())
        predicted_all.extend(predicted.cpu().numpy())

    OA = 100 * accuracy_score(targets_all, predicted_all)
    P = 100 * precision_score(targets_all, predicted_all, average='macro')
    Se = 100 * recall_score(targets_all, predicted_all, average='macro')
    Sp = 100 * calculate_specificity(targets_all, predicted_all)
    F1 = 100 * f1_score(targets_all, predicted_all, average='macro')
    Kappa = 100 * cohen_kappa_score(targets_all, predicted_all)

    train_metrics['loss'].append(train_loss / len(train_loader))
    train_metrics['P'].append(P)
    train_metrics['Se'].append(Se)
    train_metrics['Sp'].append(Sp)
    train_metrics['F1'].append(F1)
    train_metrics['OA'].append(OA)
    train_metrics['Kappa'].append(Kappa)

    print(f'Train Loss: {train_loss/len(train_loader):.3f} | '
          f'OA: {OA:.1f}% | P: {P:.1f} | Se: {Se:.1f} | '
          f'Sp: {Sp:.1f} | F1: {F1:.1f} | Kappa: {Kappa:.1f}')


def val(epoch, net, criterion, test_loader, val_metrics, model_saver):
    net.eval()
    val_loss = 0
    targets_all, predicted_all, probabilities_all = [], [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            probabilities_all.extend(probs.cpu().numpy())
            targets_all.extend(targets.cpu().numpy())
            predicted_all.extend(predicted.cpu().numpy())

    OA = 100 * accuracy_score(targets_all, predicted_all)
    P = 100 * precision_score(targets_all, predicted_all, average='macro')
    Se = 100 * recall_score(targets_all, predicted_all, average='macro')
    Sp = 100 * calculate_specificity(targets_all, predicted_all)
    F1 = 100 * f1_score(targets_all, predicted_all, average='macro')
    Kappa = 100 * cohen_kappa_score(targets_all, predicted_all)
    AUC = calculate_auc(targets_all, probabilities_all)

    val_metrics['loss'].append(val_loss / len(test_loader))
    val_metrics['P'].append(P)
    val_metrics['Se'].append(Se)
    val_metrics['Sp'].append(Sp)
    val_metrics['F1'].append(F1)
    val_metrics['OA'].append(OA)
    val_metrics['AUC'].append(AUC)
    val_metrics['Kappa'].append(Kappa)

    current_metrics = {'OA': OA, 'AUC': AUC, 'F1': F1, 'P': P, 'Se': Se, 'Sp': Sp, 'Kappa': Kappa}
    model_saver.check_and_save(net, current_metrics)

    print(f'Val Loss: {val_loss/len(test_loader):.3f} | '
          f'OA: {OA:.1f}% | P: {P:.1f} | Se: {Se:.1f} | '
          f'Sp: {Sp:.1f} | F1: {F1:.1f} | AUC: {AUC:.1f} | Kappa: {Kappa:.1f}')
