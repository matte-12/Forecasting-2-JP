# ETTh1, input fisso a 96, predizione a 96

CONFIG = {
    "dataset_name": "ETTh1",
    "csv_path": "data/ETT-small/ETTh1.csv",
    "seq_len": 96,
    "pred_len": 96,
    "batch_size": 32,
    "num_workers": 0,
    "shuffle_train": True,
    "drop_last_train": True,
    "num_features": 7,
    "epochs": 20,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "seed": 42,
    "top_k": 3,
    "use_fft": True,
    "fixed_period": 24,
    "use_inception": True,
}