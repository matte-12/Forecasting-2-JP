import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_path, flag='train', seq_len=96, pred_len=24):
        """
        Args:
            csv_path: percorso al file CSV (es. 'data/ETT-small/ETTh1.csv')
            flag: 'train', 'val', o 'test' per gestire split e scaling
            seq_len: finestra temporale di input (T)
            pred_len: orizzonte di previsione (H)
        """
        assert flag in ['train', 'val', 'test']
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.flag = flag

        # tengo solo i 7 canali numerici
        df_raw = pd.read_csv(csv_path)
        df_data = df_raw.drop(columns=['date'])

        # split per ETT: 70% train, 10% val, 20% test
        num_train = int(len(df_data) * 0.7)
        num_val = int(len(df_data) * 0.1)
        num_test = len(df_data) - num_train - num_val

        # indici di partenza e fine
        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_data)]
        
        type_map = {'train': 0, 'val': 1, 'test': 2}
        idx = type_map[flag]
        border1 = border1s[idx]
        border2 = border2s[idx]

        # scaling su train data
        self.scaler = StandardScaler()
        train_data = df_data.iloc[border1s[0]:border2s[0], :].values
        self.scaler.fit(train_data)

        self.data_x = self.scaler.transform(df_data.values)
        self.data_y  = self.data_x(df_data.values)  # stessa serie?

        """
        isolamento del contesto (prevenzione leakage), così le due istanze create in train.py sono separate:
            train_dataset = TimeSeriesDataset(flag='train')
            val_dataset = TimeSeriesDataset(flag='val')
        """
        self.data_x = self.data_x[border1:border2]
        self.data_y = self.data_y[border1:border2]

def __len__(self):
    return len(self.data_x) - self.seq_len - self.pred_len + 1

def __getitem__(self, index):
    s_begin = index
    s_end = s_begin + self.seq_len
    r_begin = s_end
    r_end = r_begin + self.pred_len

    seq_x = self.data_x[s_begin:s_end]
    seq_y = self.data_y[r_begin:r_end]

    return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

# test vari lavoraci te
if __name__ == '__main__':
    try:
        dataset = TimeSeriesDataset(csv_path='../data/ETT-small/ETTh1.csv', flag='train')
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        x, y = next(iter(dataloader))
        print(f"Input Shape [B, T, C]:  {x.shape}")
        print(f"Output Shape [B, H, C]: {y.shape}")

        assert x.shape == (32, 96, 7), "ERRORE: dim input non allineata."
        assert y.shape == (32, 24, 7), "ERRORE: dim target non allineata."
    except Exception as e:
        print(f"Test Fallito: {e}")