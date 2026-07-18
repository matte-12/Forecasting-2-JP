# 96 seq_len (24 ore) di input, 24 ore di output

CONFIG = {
    # Dataset
    "dataset_name": "ETTm1",
    "csv_path": "/content/drive/MyDrive/dataset_forecast/ETT-small/ETTm1.csv",

    # ETTm1: un timestep ogni 15 minuti
    # 96 timestep = 24 ore di input
    "seq_len": 96,

    # 24 timestep = 6 ore di previsione
    "pred_len": 24,

    # DataLoader
    "batch_size": 32,
    "num_workers": 0,
    "shuffle_train": True,
    "drop_last_train": True,

    # Numero di variabili di ETTm1
    "num_features": 7,

    # Training
    "epochs": 20,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,

    # Riproducibilità
    "seed": 42,
}

