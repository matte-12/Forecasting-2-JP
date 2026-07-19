from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Dataset per ETTh1 ed ETTm1.

    Restituisce:
        x: finestra di input con shape (seq_len, num_features)
        y: finestra futura con shape (pred_len, num_features)
    """

    def __init__(
        self,
        csv_path: str,
        flag: str,
        seq_len: int,
        pred_len: int,
    ):
        # Controllo dello split richiesto
        if flag not in {"train", "val", "test"}:
            raise ValueError("flag deve essere train, val oppure test")

        # seq_len e pred_len devono arrivare dal file YAML
        if seq_len <= 0 or pred_len <= 0:
            raise ValueError("seq_len e pred_len devono essere positivi")

        csv_path = Path(csv_path)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV non trovato: {csv_path}")

        self.seq_len = seq_len
        self.pred_len = pred_len

        # Lettura del dataset
        df = pd.read_csv(csv_path)

        if "date" not in df.columns:
            raise ValueError("Il CSV deve contenere la colonna 'date'")

        # Rimuoviamo la data e manteniamo le 7 feature numeriche
        values = df.drop(columns=["date"]).to_numpy(dtype=np.float32)

        # Split temporale: 70% train, 10% validation, 20% test
        n = len(values)
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)

        # Lo scaler viene addestrato solo sui dati di training
        self.scaler = StandardScaler()
        self.scaler.fit(values[:train_end])

        # Tutti gli split vengono trasformati con lo stesso scaler
        values = self.scaler.transform(values).astype(np.float32)

        if flag == "train":
            split = values[:train_end]

        elif flag == "val":
            # Manteniamo seq_len punti precedenti per creare
            # la prima finestra di validation
            split = values[train_end - seq_len:val_end]

        else:
            # Stessa logica per il test set
            split = values[val_end - seq_len:]

        self.data = torch.from_numpy(split)

        # Numero totale di finestre disponibili nello split
        self.length = len(self.data) - seq_len - pred_len + 1

        if self.length <= 0:
            raise ValueError(
                f"Split troppo corto per seq_len={seq_len}, "
                f"pred_len={pred_len}"
            )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Intervallo della finestra di input
        x_start = index
        x_end = x_start + self.seq_len

        # La previsione parte subito dopo la finestra di input
        y_start = x_end
        y_end = y_start + self.pred_len

        x = self.data[x_start:x_end]
        y = self.data[y_start:y_end]

        return x, y