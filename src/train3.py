import argparse
import json
import random
import time
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import yaml

from src.data import build_dataloader
from src.metrics import metric_mse, metric_mae, metric_mase
from src.models_1d import DLinear, CausalTCN
from src.models_2d import TimesNet
from src.models_light import LightTimesNet

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    def __init__(self, patience=7, path='checkpoint.pth'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Es: etth1_24")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["DLinear", "CausalTCN", "TimesNet", "LightTimesNet_Single", "LightTimesNet_Multi"])
    return parser.parse_args()

def main():
    args = parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "configs" / f"{args.config}.yaml"
    
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
        
    set_seed(config.get("seed", 42))
    device = get_device()
    
    seq_len = config["seq_len"]
    pred_len = config["pred_len"]
    enc_in = config["num_features"]
    
    # Factory esplicita dei modelli
    if args.model == "DLinear":
        model = DLinear(seq_len, pred_len, enc_in)
    elif args.model == "CausalTCN":
        model = CausalTCN(seq_len, pred_len, enc_in)
    elif args.model == "TimesNet":
        model = TimesNet(seq_len, pred_len, enc_in, d_model=32, top_k=config.get("top_k", 3))
    elif args.model == "LightTimesNet_Single":
        model = LightTimesNet(seq_len, pred_len, enc_in, d_model=32, fixed_period=config.get("fixed_period", 24), kernel_type="single")
    elif args.model == "LightTimesNet_Multi":
        model = LightTimesNet(seq_len, pred_len, enc_in, d_model=32, fixed_period=config.get("fixed_period", 24), kernel_type="multi")

    model = model.to(device)
    
    exp_dir = project_root / "experiments" / f"{args.model}_{config['dataset_name']}_H{pred_len}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, exp_dir / "config_used.yaml")
    ckpt_path = exp_dir / "checkpoint.pth"

    print(f"  Training: {args.model} su {args.config} | Device: {device}")

    _, train_loader = build_dataloader(config, "train")
    _, val_loader = build_dataloader(config, "val")
    _, test_loader = build_dataloader(config, "test")

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=5, path=str(ckpt_path))

    # --- TRAINING LOOP ---
    train_times = []
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = []
        t0 = time.time()
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            train_loss.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
        train_times.append(time.time() - t0)
        
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                out = model(batch_x.to(device))
                val_loss.append(criterion(out, batch_y.to(device)).item())
                
        avg_vloss = np.mean(val_loss)
        print(f"Epoch {epoch+1:02d} | Train MSE: {np.mean(train_loss):.4f} | Val MSE: {avg_vloss:.4f} | Time: {train_times[-1]:.2f}s")
        
        early_stopping(avg_vloss, model)
        if early_stopping.early_stop:
            print("  Early Stopping innescato.")
            break

    # --- TESTING LOOP ---
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    model.eval()
    
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            preds.append(model(batch_x.to(device)).cpu().numpy())
            trues.append(batch_y.numpy())
            
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    mse, mae, mase = metric_mse(preds, trues), metric_mae(preds, trues), metric_mase(preds, trues)
    
    print(f"\n  Risultati Test -> MSE: {mse:.4f} | MAE: {mae:.4f} | MASE: {mase:.4f}")

    metrics = {
        "model": args.model,
        "config": args.config,
        "test_mse": float(mse),
        "test_mae": float(mae),
        "test_mase": float(mase),
        "average_epoch_time": float(np.mean(train_times)),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()