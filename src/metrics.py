import numpy as np

def metric_mse(pred, true):
    return np.mean((pred - true) ** 2)

def metric_mae(pred, true):
    return np.mean(np.abs(pred - true))

def metric_mase(pred, true):
    """
    Mean Absolute Scaled Error (MASE) robusto per dataset multivariati ad alta dimensionalità.
    Ignora automaticamente le feature costanti per evitare l'esplosione della media.
    """
    # MAE del modello per singola feature: shape [Num_Features]
    mae_pred_per_feature = np.mean(np.abs(pred - true), axis=(0, 1))
    
    # MAE del Naive forecast (Random Walk) per singola feature
    naive_diff = np.abs(true[:, 1:, :] - true[:, :-1, :])
    mae_naive_per_feature = np.mean(naive_diff, axis=(0, 1))
    
    # Maschera booleana: identifica i contatori attivi (denominatore > soglia di rumore)
    active_sensors_mask = mae_naive_per_feature > 1e-5
    
    if not np.any(active_sensors_mask):
        return np.nan # Se l'intero dataset è piatto, il MASE non è calcolabile
        
    # Calcola il rapporto solo sulle feature valide
    mase_valid = mae_pred_per_feature[active_sensors_mask] / mae_naive_per_feature[active_sensors_mask]
    
    # Restituisce la media calcolata esclusivamente sui sensori operativi
    return np.mean(mase_valid)