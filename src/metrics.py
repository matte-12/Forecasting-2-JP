import numpy as np

def metric_mse(pred, true):
    return np.mean((pred - true) ** 2)

def metric_mae(pred, true):
    return np.mean(np.abs(pred - true))

def metric_mase(pred, true):
    """
    Mean Absolute Scaled Error (MASE) batch-level.
    Calcola l'errore rispetto al Naive Forecast (shift di 1 step) del target.
    Aggiunge epsilon al denominatore per evitare divisioni per zero.
    """
    mae_pred = np.mean(np.abs(pred - true))
    # Naive forecast basato sulla vera serie storica (Y_t - Y_{t-1})
    naive_err = np.abs(true[:, 1:, :] - true[:, :-1, :])
    mae_naive = np.mean(naive_err)
    
    return mae_pred / (mae_naive + 1e-8)