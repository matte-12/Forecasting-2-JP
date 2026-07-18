import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from src.data import TimeSeriesDataset
from src.models_1d import DLinear, CausalTCN
from src.models_2d import TimesNet

class EarlyStopping:
    """
    Interrompe il training se la validation loss non migliora per 'patience' epoche consecutive.
    Previene l'overfitting e salva i pesi del modello migliore.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    # questi parametri poi li mettiamo in un config.yaml
    csv_name = "ETTh1.csv"
    seq_len = 96
    pred_len = 24
    enc_in = 7
    batch_size = 32
    learning_rate = 1e-3
    epochs = 20
    model_name = "TimesNet" # set manuale: "DLinear", "CausalTCN", "TimesNet"
    
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "data" / "ETT-small" / csv_name
    checkpoint_path = project_root / f"{model_name}_checkpoint.pth"

    device = get_device()
    print(f"Avvio Training: Modello={model_name}, Dispositivo={device}")

    train_dataset = TimeSeriesDataset(str(csv_path), flag="train", seq_len=seq_len, pred_len=pred_len)
    val_dataset = TimeSeriesDataset(str(csv_path), flag="val", seq_len=seq_len, pred_len=pred_len)
    test_dataset = TimeSeriesDataset(str(csv_path), flag="test", seq_len=seq_len, pred_len=pred_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model_dict = {
        "DLinear": DLinear(seq_len, pred_len, enc_in),
        "CausalTCN": CausalTCN(seq_len, pred_len, enc_in),
        "TimesNet": TimesNet(seq_len, pred_len, enc_in, d_model=32, top_k=3)
    }
    
    model = model_dict[model_name].to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=5, verbose=True, path=str(checkpoint_path))

    for epoch in range(epochs):
        model.train()
        train_loss = []
        epoch_start_time = time.time()
        
        for _, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss.append(loss.item())
                
        train_loss_avg = np.average(train_loss)
        val_loss_avg = np.average(val_loss)
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch: {epoch + 1}/{epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")
        
        early_stopping(val_loss_avg, model)
        if early_stopping.early_stop:
            print("!! Early stopping !!")
            break

    print("\nEsecuzione Test Set col miglior modello...")
    model.load_state_dict(torch.load(str(checkpoint_path)))
    model.eval()
    
    test_loss_mse = []
    test_loss_mae = []
    mae_criterion = nn.L1Loss()
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            test_loss_mse.append(criterion(outputs, batch_y).item())
            test_loss_mae.append(mae_criterion(outputs, batch_y).item())

    print(f"Risultati Test -> MSE: {np.average(test_loss_mse):.4f} | MAE: {np.average(test_loss_mae):.4f}")

if __name__ == "__main__":
    main()