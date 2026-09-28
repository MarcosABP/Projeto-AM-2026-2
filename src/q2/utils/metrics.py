from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def metricas(y_true, y_pred, media='macro'):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average=media, zero_division=0
    )
    return np.array([(y_true != y_pred).mean(), p, r, f])
