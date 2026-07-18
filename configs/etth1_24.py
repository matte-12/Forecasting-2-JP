# 24 ore di input, 24 ore di output

CONFIG = {
    # Dataset
    "dataset_name": "ETTh1",
    "csv_path": "data/ETT-small/ETTh1.csv",

    # Dimensioni temporali
    # ETTh1 ha frequenza oraria: 1 timestep = 1 ora
    "seq_len": 24,
    "pred_len": 24,

    # Dataloader
    "batch_size": 32,
    "num_workers": 0,
    "shuffle_train": True,
    "drop_last_train": True,

    # Modello
    "num_features": 7,

    # Training
    "epochs": 20,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,

    # Riproducibilità
    "seed": 42,
}