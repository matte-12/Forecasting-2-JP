import numpy as np

def metric_mse(pred, true):
    """
    Mean Squared Error.
    """
    return np.mean((pred - true) ** 2)

def metric_mae(pred, true):
    """
    Mean Absolute Error.
    """
    return np.mean(np.abs(pred - true))

def metric_mase(pred, true):
    """
    Mean Absolute Scaled Error (MASE) corretto per serie multivariate.
    
    pred shape: [Campioni, Pred_Len, Num_Features]
    true shape: [Campioni, Pred_Len, Num_Features]
    """
    # MAE calcolato per ogni singola feature (media su Campioni e Pred_Len)
    # Shape risultante: [Num_Features]
    mae_pred_per_feature = np.mean(np.abs(pred - true), axis=(0, 1))
    
    # errore del Naive Forecast per ogni singola feature
    # Il Naive in-sample calcola la differenza assoluta tra t e t-1
    naive_diff = np.abs(true[:, 1:, :] - true[:, :-1, :])
    
    # shape risultante: [Num_Features]
    mae_naive_per_feature = np.mean(naive_diff, axis=(0, 1))
    
    # 3 MASE calcolato dividendo feature per feature, 1e-8 previene la divisione per zero se una feature è costante
    mase_per_feature = mae_pred_per_feature / (mae_naive_per_feature + 1e-8)
    
    # media finale dei MASE di tutte le feature
    return np.mean(mase_per_feature)