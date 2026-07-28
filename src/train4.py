import argparse
import random
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shutil
import torch
import torch.nn as nn
from torch import optim
import yaml

# Moduli custom (Assicurati che i nomi dei file riflettano quelli nel tuo ambiente locale)
from src.data import build_dataloader
from src.metrics import metric_mse, metric_mae, metric_mase
from src.models_1d import DLinear, CausalTCN
from src.timesnet_original import TimesNetOriginal
from src.fixed_period_inception import FixedPeriodInception2D
from src.models_light import (
    LightTimesNetMultiScale, 
    LightTimesNetDepthwise, 
    LightTimesNetGroup, 
    LightTimesNetSingleKernel
)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    def __init__(self, patience=5, path='checkpoint.pth'):
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

def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

def measure_inference_time(model, dataloader, device, warmup_batches=5, max_batches=30):
    """Modulo di misurazione latenza ottimizzato di Matteo"""
    model.eval()
    warmup_count = 0
    with torch.no_grad():
        for batch_x, _ in dataloader:
            if warmup_count >= warmup_batches:
                break
            _ = model(batch_x.to(device, non_blocking=True))
            warmup_count += 1

    synchronize_device(device)
    measured_batches = 0
    total_samples = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_x, _ in dataloader:
            if measured_batches >= max_batches:
                break
            batch_x = batch_x.to(device, non_blocking=True)
            _ = model(batch_x)
            measured_batches += 1
            total_samples += batch_x.size(0)

    synchronize_device(device)
    elapsed_seconds = time.perf_counter() - start_time

    return {
        "measured_batches": measured_batches,
        "measured_samples": total_samples,
        "total_inference_seconds": elapsed_seconds,
        "inference_ms_per_batch": (elapsed_seconds / measured_batches * 1000) if measured_batches else 0,
        "inference_ms_per_sample": (elapsed_seconds / total_samples * 1000) if total_samples else 0,
        "samples_per_second": (total_samples / elapsed_seconds) if elapsed_seconds else 0,
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="E.g., etth1_24")
    parser.add_argument("--model", type=str, required=True, help="Nome del modello da instanziare")
    parser.add_argument("--override-seq-len", type=int, default=None, help="Sovrascrive la seq_len")
    parser.add_argument("--override-period", type=int, default=None, help="Sovrascrive il fixed_period")
    parser.add_argument("--override-top-k", type=int, default=None, help="Sovrascrive il top_k")
    parser.add_argument("--override-num-blocks", type=int, default=None, help="Sovrascrive num_blocks")
    return parser.parse_args()

def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "configs" / f"{args.config}.yaml"
    
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    set_seed(config.get("seed", 42))
    device = get_device()
    
    # Override Dinamici
    seq_len = args.override_seq_len if args.override_seq_len else config["seq_len"]
    pred_len = config["pred_len"]
    enc_in = config["num_features"]
    
    period = args.override_period if args.override_period else config.get("fixed_period", 24)
    top_k = args.override_top_k if args.override_top_k else config.get("top_k", 3)
    num_blocks = args.override_num_blocks if args.override_num_blocks else config.get("num_blocks", 1)

    # Aggiorna il config per passarlo al Dataloader
    config["seq_len"] = seq_len
    
    # ----------------------------------------------------------------------
    # EXPLICIT MODEL FACTORY
    # ----------------------------------------------------------------------
    if args.model == "DLinear":
        model = DLinear(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
    elif args.model == "CausalTCN":
        model = CausalTCN(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
    elif args.model == "TimesNetOriginal":
        model = TimesNetOriginal(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, d_model=32, d_ff=64, top_k=top_k, num_blocks=num_blocks)
    elif args.model == "FixedPeriodInception":
        model = FixedPeriodInception2D(seq_len=seq_len, pred_len=pred_len, num_features=enc_in, period=period, d_model=32, d_ff=64, num_blocks=num_blocks)
    elif args.model == "LightTimesNet_MultiScale":
        model = LightTimesNetMultiScale(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, d_model=32, fixed_period=period, num_blocks=num_blocks)
    elif args.model == "LightTimesNet_Depthwise":
        model = LightTimesNetDepthwise(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, d_model=32, fixed_period=period, num_blocks=num_blocks)
    elif args.model == "LightTimesNet_Group":
        model = LightTimesNetGroup(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, d_model=32, fixed_period=period, num_blocks=num_blocks)
    elif args.model == "LightTimesNet_SingleKernel":
        model = LightTimesNetSingleKernel(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, d_model=32, fixed_period=period, num_blocks=num_blocks)
    else:
        raise ValueError(f"Modello {args.model} non supportato o nome errato nel JSON.")

    model = model.to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ----------------------------------------------------------------------
    # NAMING INTELLIGENTE DELLE DIRECTORY ESPERIMENTI
    # ----------------------------------------------------------------------
    exp_name = f"{args.model}_{config['dataset_name']}_S{seq_len}_H{pred_len}"
    
    if args.model == "TimesNetOriginal":
        exp_name += f"_K{top_k}_B{num_blocks}"
    elif args.model not in ["DLinear", "CausalTCN"]:
        exp_name += f"_P{period}_B{num_blocks}"

    exp_dir = project_root / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_path = exp_dir / "best_model.pth"
    shutil.copy(config_path, exp_dir / "config_used.yaml")

    print(f"🚀 Modello: {args.model} | Config: {args.config} | Parametri: {parameter_count:,}")
    
    _, train_loader = build_dataloader(config, flag="train")
    _, val_loader = build_dataloader(config, flag="val")
    _, test_loader = build_dataloader(config, flag="test")

    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config.get("weight_decay", 0.0))
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=config.get("patience", 5), path=str(ckpt_path))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # --- TRAINING LOOP ---
    train_history = {"train_mse": [], "val_mse": [], "epoch_time_seconds": []}
    training_start = time.perf_counter()

    for epoch in range(config["epochs"]):
        model.train()
        train_loss = []
        epoch_start = time.perf_counter()
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch_x)
            loss = criterion(out, batch_y)
            train_loss.append(loss.item() * batch_x.size(0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
        epoch_time = time.perf_counter() - epoch_start
        train_mse_epoch = np.sum(train_loss) / len(train_loader.dataset)
        
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                out = model(batch_x.to(device, non_blocking=True))
                val_loss.append(criterion(out, batch_y.to(device, non_blocking=True)).item() * batch_x.size(0))
                
        val_mse_epoch = np.sum(val_loss) / len(val_loader.dataset)
        
        train_history["train_mse"].append(float(train_mse_epoch))
        train_history["val_mse"].append(float(val_mse_epoch))
        train_history["epoch_time_seconds"].append(float(epoch_time))
        
        print(f"Epoch {epoch+1:02d} | Train MSE: {train_mse_epoch:.4f} | Val MSE: {val_mse_epoch:.4f} | Time: {epoch_time:.2f}s")
        
        early_stopping(val_mse_epoch, model)
        if early_stopping.early_stop:
            print("🛑 Early Stopping Innescato.")
            break

    total_training_time = time.perf_counter() - training_start

    # --- TESTING E INFERENZA ---
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
    
    inf_stats = measure_inference_time(model, test_loader, device)

    # Raccolta Metriche Finali
    metrics = {
        "model": args.model,
        "config": args.config,
        "dataset": config["dataset_name"],
        "pred_len": pred_len,
        "fixed_period": period,
        "top_k": top_k,
        "num_blocks": num_blocks,
        "test_mse": float(mse),
        "test_mae": float(mae),
        "test_mase": float(mase),
        "trainable_parameters": int(parameter_count),
        "total_training_time_seconds": float(total_training_time),
        "average_epoch_time_seconds": float(np.mean(train_history["epoch_time_seconds"])),
        "inference_ms_per_sample": float(inf_stats["inference_ms_per_sample"]),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None,
        "checkpoint_size_mb": float(ckpt_path.stat().st_size / 1024**2)
    }
    
    print(f"\n📊 Risultati Test -> MSE: {mse:.4f} | MAE: {mae:.4f} | MASE: {mase:.4f}")

    with open(exp_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    with open(exp_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(train_history, f, indent=4)

if __name__ == "__main__":
    main()