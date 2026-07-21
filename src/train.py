import argparse
import importlib
import time
from pathlib import Path

import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import yaml

from src.data import build_dataloader
from src.models_1d import DLinear, CausalTCN
from models_2d import TimesNet
from models.fixed_period_inception import FixedPeriodInception2D

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
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). '
                'Saving model...'
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def load_config(config_name):
    """
    Carica configs/<config_name>.yaml.
    Accetta il nome senza estensione, ad esempio 'etth1_24'.
    """
    config_path = Path(__file__).resolve().parent.parent / "configs" / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"File config non trovato: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config YAML non valido: {config_path}")

    return config

"""pianifico di usare sti comandi per testare i 3 modelli, limitiamo a 2 orizzonti?

orizzonte breve
python -m src.train --config etth1_24 --model DLinear
python -m src.train --config etth1_24 --model CausalTCN
python -m src.train --config etth1_24 --model TimesNet

orizzonte medio
python -m src.train --config etth1_96 --model DLinear
python -m src.train --config etth1_96 --model CausalTCN
python -m src.train --config etth1_96 --model TimesNet
"""
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Nome del file di configurazione senza estensione, ad esempio etth1_24.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["DLinear", "CausalTCN", "TimesNet"],
        help="Modello da usare.",
    )

    return parser.parse_args()

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # metal, macos
    return torch.device("cpu")

# modifica per salvataggio .pth in root del progetto di default oppure su colab drive
# con l'istruzione:
# import os
# os.environ["EXPERIMENTS_DIR"] = "/content/drive/MyDrive/.../experiments"

def build_checkpoint_path(args, config):
    experiments_root = Path(
        os.environ.get(
            "EXPERIMENTS_DIR",
            Path(__file__).resolve().parent.parent / "experiments"
        )
    )

    suffix_parts = []

    if not config.get("use_fft", True):
        suffix_parts.append("noFFT")

    if not config.get("use_inception", True):
        suffix_parts.append("noIncep")

    suffix = "" if not suffix_parts else "_" + "_".join(suffix_parts)

    exp_name = (
        f"{args.model}_"
        f"{config['dataset_name']}_"
        f"H{config['pred_len']}"
        f"{suffix}"
    )

    exp_dir = experiments_root / exp_name

    # crea la cartella esperimento
    exp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return exp_dir / "checkpoint.pth"

# modificato il main per config save
def main():
    # parametri poi li mettiamo in un config.yaml ? attualmente gestiti con 6 comandi separati per isolare 
    # i modelli e poter fare training separato
    args = parse_args()
    config = load_config(args.config)

    # prima crea il path
    checkpoint_path = build_checkpoint_path(args, config)

    # poi usa checkpoint_path

    # mod per salvare visualizzare file config usato

    import shutil

    shutil.copy(
    Path(__file__).resolve().parent.parent /
    "configs" /
    f"{args.config}.yaml",

    checkpoint_path.parent /
    "config_used.yaml"
    )
    #

    seq_len = config["seq_len"]
    pred_len = config["pred_len"]
    enc_in = config["num_features"]
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]
    top_k = config.get("top_k", 3)
    use_fft = config.get("use_fft", True)
    fixed_period = config.get("fixed_period", 24)
    use_inception = config.get("use_inception", True)

 
    # project_root = Path(__file__).resolve().parent.parent
    # checkpoint_path = build_checkpoint_path(project_root, args, config)

    # modificato per salvare i checkpoint in una cartella dedicata, 
    # con possibilità di cambiare la root tramite variabile d'ambiente
    # checkpoint_path = build_checkpoint_path(args, config)
    # sposato in alto

    device = get_device()
    print(
        f"Avvio Training: Modello={args.model}, "
        f"Config={args.config}, Dispositivo={device}"
    )

    train_dataset, train_loader = build_dataloader(config, flag="train")
    val_dataset, val_loader = build_dataloader(config, flag="val")
    test_dataset, test_loader = build_dataloader(config, flag="test")

    model_dict = {
        "DLinear": DLinear(seq_len, pred_len, enc_in),
        "CausalTCN": CausalTCN(seq_len, pred_len, enc_in),
        "TimesNet": TimesNet(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=32,
            top_k=top_k,
            use_fft=use_fft,
            fixed_period=fixed_period,
            use_inception=use_inception,
        ),
    }
    
    model = model_dict[args.model].to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(
        patience=5,
        verbose=True,
        path=str(checkpoint_path),
    )

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
        
        print(
            f"Epoch: {epoch + 1}/{epochs} | Time: {epoch_time:.2f}s | "
            f"Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}"
        )
        
        early_stopping(val_loss_avg, model)
        if early_stopping.early_stop:
            print("!! Early stopping !!")
            break

    print("\nEsecuzione Test Set col miglior modello...")
    model.load_state_dict(torch.load(str(checkpoint_path)))
    model.eval()
    
    #aggiungo predictions per fare: 
    # - confronto grafico previsione vs reale 
    # - confronto tra modelli
    # ese: predictions.csv
    #timestamp,prediction,actual
    #2025-01-01 00:00,101.2,100.8

    predictions = []
    targets = []

    test_loss_mse = []
    test_loss_mae = []
    mae_criterion = nn.L1Loss()
    
    #modificato per predictions.csv
    with torch.no_grad():
        for batch_x, batch_y in test_loader:

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)

            test_loss_mse.append(
                criterion(outputs, batch_y).item()
            )

            test_loss_mae.append(
                mae_criterion(outputs, batch_y).item()
            )

            predictions.extend(
                outputs.cpu().numpy().reshape(-1)
            )

            targets.extend(
                batch_y.cpu().numpy().reshape(-1)
            )

    print(
        f"Risultati Test -> MSE: {np.average(test_loss_mse):.4f} | "
        f"MAE: {np.average(test_loss_mae):.4f}"
    )

    # salvataggio metriche in file json
    import json

    metrics = {
        "model": args.model,
        "config": args.config,
        "dataset": config["dataset_name"],
        "pred_len": config["pred_len"],
        "MSE": float(np.average(test_loss_mse)),
        "MAE": float(np.average(test_loss_mae))
    }

    with open(
        checkpoint_path.parent / "metrics.json",
        "w"
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        f"Metriche salvate in: {checkpoint_path.parent / 'metrics.json'}"
    )

    #salva prediction.csv

    import pandas as pd

    pred_df = pd.DataFrame(
        {
            "prediction": predictions,
            "actual": targets
        }
    )

    pred_df.to_csv(
        checkpoint_path.parent / "predictions.csv",
        index=False
    )

    print(
        f"Predizioni salvate in: {checkpoint_path.parent / 'predictions.csv'}"
    )

if __name__ == "__main__":
    main()