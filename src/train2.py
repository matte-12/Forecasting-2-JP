import argparse
import os
import random
import time
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import yaml

from src.data import build_dataloader
from src.models_1d import DLinear, CausalTCN
from src.models_2d import TimesNet
from src.models_light import LightTimesNet
from src.metrics import metric_mse, metric_mae, metric_mase

def set_seed(seed=42):
    """Rende l'addestramento riproducibile."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        choices=["DLinear", "CausalTCN", "TimesNet", "LightTimesNet"],
        help="Modello da usare.",
    )
    return parser.parse_args()

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # metal, macos
    return torch.device("cpu")

def build_checkpoint_path(args, config):
    experiments_root = Path(
        os.environ.get(
            "EXPERIMENTS_DIR",
            Path(__file__).resolve().parent.parent / "experiments"
        )
    )
    exp_name = f"{args.model}_{config['dataset_name']}_H{config['pred_len']}"
    exp_dir = experiments_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir / "checkpoint.pth"

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # 1. Applicazione Seed (CRITICO)
    set_seed(config.get("seed", 42))

    checkpoint_path = build_checkpoint_path(args, config)

    # Copia config
    shutil.copy(
        Path(__file__).resolve().parent.parent / "configs" / f"{args.config}.yaml",
        checkpoint_path.parent / "config_used.yaml"
    )

    seq_len = config["seq_len"]
    pred_len = config["pred_len"]
    enc_in = config["num_features"]
    learning_rate = config["learning_rate"]
    epochs = config["epochs"]
    fixed_period = config.get("fixed_period", 24)

    device = get_device()
    print(f"Avvio Training: Modello={args.model}, Config={args.config}, Dispositivo={device}")

    train_dataset, train_loader = build_dataloader(config, flag="train")
    val_dataset, val_loader = build_dataloader(config, flag="val")
    test_dataset, test_loader = build_dataloader(config, flag="test")

    # Iniezione dei modelli
    model_dict = {
        "DLinear": DLinear(seq_len, pred_len, enc_in),
        "CausalTCN": CausalTCN(seq_len, pred_len, enc_in),
        "TimesNet": TimesNet(seq_len, pred_len, enc_in, d_model=32, top_k=config.get("top_k", 3)),
        "LightTimesNet": LightTimesNet(seq_len, pred_len, enc_in, d_model=32, fixed_period=fixed_period)
    }
    
    if args.model not in model_dict:
        raise ValueError(f"Modello {args.model} non supportato.")
        
    model = model_dict[args.model].to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=5, verbose=True, path=str(checkpoint_path))

    for epoch in range(epochs):
        model.train()
        train_loss = []
        epoch_start_time = time.time()
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            loss.backward()
            
            # Gradient Clipping per stabilità reti profonde
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
    model.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
    model.eval()
    
    predictions, targets = [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            
            predictions.append(outputs.cpu().numpy())
            targets.append(batch_y.cpu().numpy())

    # Calcolo metriche vettorializzato su tutto il test set
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    mse = metric_mse(predictions, targets)
    mae = metric_mae(predictions, targets)
    mase = metric_mase(predictions, targets)

    print(f"Risultati Test -> MSE: {mse:.4f} | MAE: {mae:.4f} | MASE: {mase:.4f}")

    metrics = {
        "model": args.model,
        "config": args.config,
        "dataset": config["dataset_name"],
        "pred_len": config["pred_len"],
        "MSE": float(mse),
        "MAE": float(mae),
        "MASE": float(mase)
    }

    with open(checkpoint_path.parent / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Salvataggio ridotto per evitare CSV enormi: salva il target flattened
    pred_df = pd.DataFrame({
        "prediction": predictions.reshape(-1),
        "actual": targets.reshape(-1)
    })
    pred_df.to_csv(checkpoint_path.parent / "predictions.csv", index=False)

if __name__ == "__main__":
    main()