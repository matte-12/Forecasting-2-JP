from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        csv_path,
        flag="train",
        seq_len=96,
        pred_len=24,
    ):
        """
        Dataset multivariato per ETTh1, ETTh2, ETTm1 ed ETTm2.

        Args:
            csv_path:
                Percorso assoluto o relativo al file CSV.

            flag:
                Split da utilizzare: "train", "val" oppure "test".

            seq_len:
                Numero di timestep della finestra di input.

            pred_len:
                Numero di timestep futuri da prevedere.
        """
        if flag not in {"train", "val", "test"}:
            raise ValueError(
                "flag deve essere 'train', 'val' oppure 'test'."
            )

        if seq_len <= 0:
            raise ValueError("seq_len deve essere maggiore di zero.")

        if pred_len <= 0:
            raise ValueError("pred_len deve essere maggiore di zero.")

        self.csv_path = Path(csv_path).expanduser()
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"File CSV non trovato:\n{self.csv_path}"
            )

        # Lettura del CSV.
        df_raw = pd.read_csv(self.csv_path)

        if "date" not in df_raw.columns:
            raise ValueError(
                "Il CSV deve contenere una colonna chiamata 'date'."
            )

        # Manteniamo solo i canali numerici.
        df_data = df_raw.drop(columns=["date"])

        if df_data.empty:
            raise ValueError(
                "Il CSV non contiene colonne numeriche utilizzabili."
            )

        non_numeric_columns = [
            column
            for column in df_data.columns
            if not pd.api.types.is_numeric_dtype(df_data[column])
        ]

        if non_numeric_columns:
            raise ValueError(
                "Sono presenti colonne non numeriche: "
                f"{non_numeric_columns}"
            )

        values = df_data.to_numpy(dtype=np.float32)

        self.feature_names = list(df_data.columns)
        self.num_features = len(self.feature_names)

        # Split 70% train, 10% validation, 20% test.
        num_samples = len(values)
        num_train = int(num_samples * 0.7)
        num_val = int(num_samples * 0.1)
        num_test = num_samples - num_train - num_val

        # Validation e test includono seq_len timestep precedenti,
        # necessari per costruire la prima finestra di input.
        border1s = {
            "train": 0,
            "val": num_train - self.seq_len,
            "test": num_train + num_val - self.seq_len,
        }

        border2s = {
            "train": num_train,
            "val": num_train + num_val,
            "test": num_samples,
        }

        border1 = border1s[flag]
        border2 = border2s[flag]

        if border1 < 0:
            raise ValueError(
                f"seq_len={self.seq_len} è troppo grande "
                f"per lo split '{flag}'."
            )

        # Lo scaler viene adattato esclusivamente sui dati di training.
        self.scaler = StandardScaler()
        self.scaler.fit(values[:num_train])

        scaled_values = self.scaler.transform(values).astype(
            np.float32
        )

        # Isoliamo lo split selezionato.
        self.data_x = scaled_values[border1:border2]

        # Input e target appartengono alla stessa serie multivariata.
        # Non è una copia separata: è sufficiente conservare un array.
        self.data_y = self.data_x

        if len(self) <= 0:
            raise ValueError(
                f"Lo split '{flag}' non contiene abbastanza dati per "
                f"seq_len={self.seq_len} e pred_len={self.pred_len}."
            )

    def __len__(self):
        """
        Numero totale di finestre disponibili.
        """
        return (
            len(self.data_x)
            - self.seq_len
            - self.pred_len
            + 1
        )

    def __getitem__(self, index):
        """
        Restituisce:

            seq_x: [seq_len, num_features]
            seq_y: [pred_len, num_features]
        """
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Indice {index} non valido. "
                f"Il dataset contiene {len(self)} finestre."
            )

        input_start = index
        input_end = input_start + self.seq_len

        target_start = input_end
        target_end = target_start + self.pred_len

        seq_x = self.data_x[input_start:input_end]
        seq_y = self.data_y[target_start:target_end]

        return (
            torch.from_numpy(seq_x),
            torch.from_numpy(seq_y),
        )

    def inverse_transform(self, data):
        """
        Riporta dati standardizzati alla scala originale.

        Accetta tensori o array con forma [..., num_features].
        """
        is_tensor = torch.is_tensor(data)

        if is_tensor:
            original_device = data.device
            original_dtype = data.dtype
            data_numpy = data.detach().cpu().numpy()    # usa cuda o mps se disponibile
        else:
            data_numpy = np.asarray(data)

        original_shape = data_numpy.shape

        if original_shape[-1] != self.num_features:
            raise ValueError(
                f"L'ultima dimensione deve essere "
                f"{self.num_features}, ricevuta {original_shape[-1]}."
            )

        flattened = data_numpy.reshape(-1, self.num_features)
        restored = self.scaler.inverse_transform(flattened)
        restored = restored.reshape(original_shape)

        if is_tensor:
            return torch.as_tensor(
                restored,
                dtype=original_dtype,
                device=original_device,
            )

        return restored