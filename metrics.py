# metrics.py
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

def calculate_specificity(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)
    specificity = 0.0
    for i in range(cm.shape[0]):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        denominator = tn + fp
        specificity += tn / denominator if denominator != 0 else 0.0
    return specificity / cm.shape[0]

def calculate_auc(y_true, y_probs):

    try:
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            return 100 * roc_auc_score(y_true, np.array(y_probs)[:, 1])
        else:
            return 100 * roc_auc_score(
                y_true, y_probs,
                multi_class='ovr', average='macro'
            )
    except Exception as e:
        print(f"AUC calculation failed: {str(e)}")
        return 0.0
