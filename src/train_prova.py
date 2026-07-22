import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import optim
import os

from src.data import build_dataloader

"""
comando colab per eseguire top k timesnet da 1 a 5 automatico: !python -m src.train_prova \
    --config etth1 \
    --model timesnet
"""



def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Training delle reti 2D TimesNet-inspired."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Nome del file YAML nella cartella configs "
            "oppure percorso completo del file."
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "timesnet",
            "fixed_period_inception",
            "timesnet_light_depthwise",
            "timesnet_light",
        ],
        help="Architettura 2D da allenare.",
    )

    parser.add_argument(
        "--top-k-values",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help=(
            "Valori di top_k da provare con TimesNet. "
            "Default: 1 2 3 4 5."
        ),
    )

    parser.add_argument(
        "--block",
        type=str,
        default="inception",
        choices=["inception", "depthwise", "group", "residual"]
    )

    return parser.parse_args()


def resolve_config_path(
    config_argument: str,
) -> Path:
    """
    Accetta:

        --config etth1_pred24

    e cerca:

        configs/etth1_pred24.yaml

    Accetta anche:

        --config configs/etth1_pred24.yaml
    """
    config_path = Path(
        config_argument
    ).expanduser()

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    if config_path.suffix == "":
        config_path = (
            project_root
            / "configs"
            / f"{config_argument}.yaml"
        )

    elif not config_path.is_absolute():
        config_path = (
            project_root
            / config_path
        )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configurazione non trovata: "
            f"{config_path}"
        )

    return config_path


def load_config(
    config_argument: str,
) -> tuple[dict, Path]:
    config_path = resolve_config_path(
        config_argument
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configurazione YAML non valida: "
            f"{config_path}"
        )

    return config, config_path


def validate_config(
    config: dict,
) -> None:
    required_keys = [
        "dataset_name",
        "csv_path",
        "seq_len",
        "pred_len",
        "batch_size",
        "num_features",
        "epochs",
        "learning_rate",
        "seed",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in config
    ]

    if missing_keys:
        raise KeyError(
            "Chiavi mancanti nel file YAML: "
            f"{missing_keys}"
        )

    positive_keys = [
        "seq_len",
        "pred_len",
        "batch_size",
        "num_features",
        "epochs",
        "learning_rate",
    ]

    for key in positive_keys:
        if config[key] <= 0:
            raise ValueError(
                f"{key} deve essere positivo."
            )


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")

def get_model_fixed_period(
    model: nn.Module,
):
    """
    Restituisce il periodo effettivamente usato dal modello,
    indipendentemente dalla sua implementazione interna.
    """
    if hasattr(model, "period"):
        return int(model.period)

    if hasattr(model, "fixed_period"):
        return int(model.fixed_period)

    if (
        hasattr(model, "times_block")
        and hasattr(model.times_block, "fixed_period")
    ):
        return int(model.times_block.fixed_period)

    return None

def build_model(
    model_name: str,
    config: dict,
    top_k: int|None = None,
) -> nn.Module:
    """
    Factory dedicata ai modelli TimesNet-inspired.

    Gli import sono interni ai singoli rami. In questo modo
    la futura top_k_inception.py non è obbligatoria finché
    non viene selezionata da riga di comando.
    """

    if model_name == "timesnet":
        from src.models_2d import (
            TimesNet,
        )
        effective_top_k = (
            top_k
            if top_k is not None
            else config.get("top_k", 3)
        )

        return TimesNet(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            enc_in=config["num_features"],
            d_model=config.get("d_model", 32),
            top_k=effective_top_k,
            use_fft=config.get("use_fft", True),
            fixed_period=config.get("fixed_period", 24),
            use_inception=config.get("use_inception", True),
        )

    if model_name == "fixed_period_inception":
        from src.models.fixed_period_inception import (
            FixedPeriodInception2D,
        )

        return FixedPeriodInception2D(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            period=config["fixed_period"],
            num_features=config["num_features"],
            d_model=config.get(
                "d_model",
                32,
            ),
            d_ff=config.get(
                "d_ff",
                64,
            ),
            kernel_sizes=tuple(
                config.get(
                    "kernel_sizes",
                    [1, 3, 5],
                )
            ),
            dropout=config.get(
                "dropout",
                0.1,
            ),
        )

    if model_name == "top_k_inception":
        try:
            from src.models.top_k_inception import (
                TopKInception2D,
            )
        except ImportError as error:
            raise ImportError(
                "Il modello top_k_inception è stato "
                "selezionato, ma il file "
                "'src/models_2d/top_k_inception.py' "
                "o la classe 'TopKInception2D' "
                "non sono ancora disponibili."
            ) from error

        return TopKInception2D(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            num_features=config["num_features"],
            top_k=config.get(
                "top_k",
                3,
            ),
            d_model=config.get(
                "d_model",
                32,
            ),
            d_ff=config.get(
                "d_ff",
                64,
            ),
            kernel_sizes=tuple(
                config.get(
                    "kernel_sizes",
                    [1, 3, 5],
                )
            ),
            dropout=config.get(
                "dropout",
                0.1,
            ),
        )

    if model_name == "timesnet_light_depthwise":
        from models.models_light_depthwise import (
            LightTimesNet,
        )

        return LightTimesNet(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            num_features=config["num_features"],
            d_model=config.get("d_model", 32),
            d_ff=config.get("d_ff", 64),
            top_k=config.get("top_k", 3),
            dropout=config.get("dropout", 0.1),
        )

    raise ValueError(
        f"Modello non riconosciuto: {model_name}"
    )


def create_experiment_directory(
    model_name: str,
    config_path: Path,
    model: nn.Module | None = None,
    top_k: int | None = None,
) -> Path:
    """
    Crea cartelle diverse in base al tipo di esperimento.

    TimesNet:
        experiments/timesnet_etth1_24/top_k_1/

    Modelli fixed-period:
        experiments/fixed_period_inception_etth1_24/period_24/
    """
    project_root = Path(__file__).resolve().parent.parent

    experiments_root = Path(
        os.environ.get(
            "EXPERIMENTS_DIR",
            project_root / "experiments",
        )
    )

    config_name = config_path.stem

    base_directory = (
        experiments_root
        / f"{model_name}_{config_name}"
    )

    if model_name == "timesnet":
        if top_k is None:
            raise ValueError(
                "top_k deve essere specificato per TimesNet."
            )

        experiment_directory = (
            base_directory
            / f"top_k_{int(top_k)}"
        )

    else:
        fixed_period = (
            get_model_fixed_period(model)
            if model is not None
            else None
        )

        if fixed_period is not None:
            experiment_directory = (
                base_directory
                / f"period_{fixed_period}"
            )
        else:
            experiment_directory = base_directory

    experiment_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return experiment_directory


def run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
) -> float:
    """
    Se optimizer è presente esegue training.
    Se optimizer è None esegue valutazione.
    """
    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    context = (
        torch.enable_grad()
        if is_training
        else torch.no_grad()
    )

    with context:
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(
                device,
                non_blocking=True,
            )

            batch_y = batch_y.to(
                device,
                non_blocking=True,
            )

            if is_training:
                optimizer.zero_grad()

            predictions = model(batch_x)

            if predictions.shape != batch_y.shape:
                raise RuntimeError(
                    "Shape non compatibili: "
                    f"prediction={predictions.shape}, "
                    f"target={batch_y.shape}."
                )

            loss = criterion(
                predictions,
                batch_y,
            )

            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = batch_x.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "Il DataLoader non contiene campioni."
        )

    return total_loss / total_samples


def evaluate_test(
    model: nn.Module,
    dataloader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    element_count = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(
                device,
                non_blocking=True,
            )

            batch_y = batch_y.to(
                device,
                non_blocking=True,
            )

            predictions = model(batch_x)

            squared_error_sum += (
                (predictions - batch_y)
                .pow(2)
                .sum()
                .item()
            )

            absolute_error_sum += (
                (predictions - batch_y)
                .abs()
                .sum()
                .item()
            )

            element_count += batch_y.numel()

    if element_count == 0:
        raise RuntimeError(
            "Il test DataLoader non contiene elementi."
        )

    mse = squared_error_sum / element_count
    mae = absolute_error_sum / element_count

    return mse, mae

def synchronize_device(device: torch.device) -> None:
    """
    Attende il completamento delle operazioni asincrone.

    È necessario soprattutto su CUDA per misurare correttamente
    i tempi di esecuzione.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elif device.type == "mps":
        torch.mps.synchronize()


def measure_inference_time(
    model: nn.Module,
    dataloader,
    device: torch.device,
    warmup_batches: int = 5,
    max_batches: int = 30,
) -> dict:
    """
    Misura il tempo di inferenza del modello.

    Restituisce:
        - millisecondi medi per batch;
        - millisecondi medi per campione;
        - campioni elaborati al secondo.

    Il warm-up evita di includere nel risultato inizializzazioni
    CUDA e allocazioni eseguite soltanto al primo forward.
    """
    model.eval()

    batches = list(dataloader)

    if not batches:
        raise RuntimeError(
            "Impossibile misurare l'inferenza: "
            "il DataLoader è vuoto."
        )

    # Warm-up
    with torch.no_grad():
        for index in range(
            min(warmup_batches, len(batches))
        ):
            batch_x, _ = batches[index]

            batch_x = batch_x.to(
                device,
                non_blocking=True,
            )

            _ = model(batch_x)

    synchronize_device(device)

    measured_batches = min(
        max_batches,
        len(batches),
    )

    total_samples = 0

    start_time = time.perf_counter()

    with torch.no_grad():
        for index in range(measured_batches):
            batch_x, _ = batches[index]

            batch_x = batch_x.to(
                device,
                non_blocking=True,
            )

            _ = model(batch_x)

            total_samples += batch_x.size(0)

    synchronize_device(device)

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    if measured_batches == 0 or total_samples == 0:
        raise RuntimeError(
            "Nessun batch misurato durante l'inferenza."
        )

    return {
        "measured_batches": measured_batches,
        "measured_samples": total_samples,
        "total_inference_seconds": elapsed_seconds,
        "inference_ms_per_batch": (
            elapsed_seconds
            / measured_batches
            * 1000
        ),
        "inference_ms_per_sample": (
            elapsed_seconds
            / total_samples
            * 1000
        ),
        "samples_per_second": (
            total_samples / elapsed_seconds
        ),
    }

def run_experiment(
    args,
    config: dict,
    config_path: Path,
    top_k: int,
) -> dict:
    """
    Allena e valuta una singola configurazione TimesNet
    con uno specifico valore di top_k.
    """

    # Fondamentale: stesso seed per ogni k.
    set_seed(config["seed"])

    device = get_device()

    train_dataset, train_loader = build_dataloader(
        config,
        flag="train",
    )

    val_dataset, val_loader = build_dataloader(
        config,
        flag="val",
    )

    test_dataset, test_loader = build_dataloader(
        config,
        flag="test",
    )

    model = build_model(
        model_name=args.model,
        config=config,
        top_k=top_k,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get(
            "weight_decay",
            0.0,
        ),
    )

    criterion = nn.MSELoss()

    experiment_directory = create_experiment_directory(
        model_name=args.model,
        config_path=config_path,
        top_k=top_k,
         model=model,
    )

    checkpoint_path = (
        experiment_directory / "best_model.pth"
    )

    config_used = dict(config)
    config_used["top_k"] = int(top_k)

    with (
        experiment_directory / "config_used.yaml"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config_used,
            file,
            sort_keys=False,
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\n" + "=" * 70)
    print(f"Modello: TimesNet")
    print(f"Dataset: {config['dataset_name']}")
    print(f"seq_len: {config['seq_len']}")
    print(f"pred_len: {config['pred_len']}")
    print(f"top_k: {top_k}")
    print(f"Device: {device}")
    print(f"Parametri: {parameter_count:,}")
    print(f"Output: {experiment_directory}")
    print("=" * 70)

    # Verifica iniziale delle shape.
    sample_x, sample_y = next(iter(train_loader))

    with torch.no_grad():
        sample_prediction = model(
            sample_x.to(device)
        ).cpu()

    if sample_prediction.shape != sample_y.shape:
        raise RuntimeError(
            "Shape non compatibili: "
            f"prediction={sample_prediction.shape}, "
            f"target={sample_y.shape}"
        )

    best_validation_loss = float("inf")
    patience = config.get("patience", 5)
    epochs_without_improvement = 0

    history = {
        "train_mse": [],
        "val_mse": [],
        "epoch_time_seconds": [],
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    synchronize_device(device)
    training_start = time.perf_counter()

    for epoch in range(
        1,
        config["epochs"] + 1,
    ):
        synchronize_device(device)
        epoch_start = time.perf_counter()

        train_mse = run_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        val_mse = run_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        synchronize_device(device)

        epoch_time = (
            time.perf_counter() - epoch_start
        )

        history["train_mse"].append(
            float(train_mse)
        )

        history["val_mse"].append(
            float(val_mse)
        )

        history["epoch_time_seconds"].append(
            float(epoch_time)
        )

        print(
            f"k={top_k} | "
            f"Epoch {epoch:03d}/{config['epochs']} | "
            f"time={epoch_time:.2f}s | "
            f"train MSE={train_mse:.6f} | "
            f"val MSE={val_mse:.6f}"
        )

        if val_mse < best_validation_loss:
            best_validation_loss = val_mse
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping per top_k={top_k}."
                )
                break

    synchronize_device(device)

    total_training_time = (
        time.perf_counter() - training_start
    )

    completed_epochs = len(
        history["epoch_time_seconds"]
    )

    average_epoch_time = float(
        np.mean(history["epoch_time_seconds"])
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    test_mse, test_mae = evaluate_test(
        model=model,
        dataloader=test_loader,
        device=device,
    )

    inference_stats = measure_inference_time(
        model=model,
        dataloader=test_loader,
        device=device,
        warmup_batches=config.get(
            "inference_warmup_batches",
            5,
        ),
        max_batches=config.get(
            "inference_measure_batches",
            30,
        ),
    )

    checkpoint_size_mb = (
        checkpoint_path.stat().st_size
        / 1024**2
    )

    if device.type == "cuda":
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated(device)
            / 1024**2
        )

        peak_gpu_reserved_mb = (
            torch.cuda.max_memory_reserved(device)
            / 1024**2
        )
    else:
        peak_gpu_memory_mb = None
        peak_gpu_reserved_mb = None

    metrics = {
        "model": "timesnet",
        "config": config_path.stem,
        "dataset": config["dataset_name"],
        "seq_len": int(config["seq_len"]),
        "pred_len": int(config["pred_len"]),
        "batch_size": int(config["batch_size"]),
        "top_k": int(top_k),
        "use_fft": bool(
            config.get("use_fft", True)
        ),
        "use_inception": bool(
            config.get("use_inception", True)
        ),
        "device": str(device),

        "trainable_parameters": int(
            parameter_count
        ),
        "checkpoint_size_mb": float(
            checkpoint_size_mb
        ),

        "completed_epochs": int(
            completed_epochs
        ),
        "best_validation_mse": float(
            best_validation_loss
        ),
        "total_training_time_seconds": float(
            total_training_time
        ),
        "average_epoch_time_seconds": float(
            average_epoch_time
        ),

        "test_mse": float(test_mse),
        "test_mae": float(test_mae),

        "inference_ms_per_batch": float(
            inference_stats[
                "inference_ms_per_batch"
            ]
        ),
        "inference_ms_per_sample": float(
            inference_stats[
                "inference_ms_per_sample"
            ]
        ),
        "samples_per_second": float(
            inference_stats[
                "samples_per_second"
            ]
        ),

        "peak_gpu_memory_mb": (
            float(peak_gpu_memory_mb)
            if peak_gpu_memory_mb is not None
            else None
        ),
        "peak_gpu_reserved_mb": (
            float(peak_gpu_reserved_mb)
            if peak_gpu_reserved_mb is not None
            else None
        ),
    }

    metrics_path = (
        experiment_directory / "metrics.json"
    )

    history_path = (
        experiment_directory / "history.json"
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=4,
        )

    print(
        f"Completato top_k={top_k} | "
        f"test MSE={test_mse:.6f} | "
        f"test MAE={test_mae:.6f}"
    )

    return metrics


def compare_top_k_results(
    results: list[dict],
    config_path: Path,
) -> None:
    """
    Crea tabella e grafici per il confronto tra k=1,...,5.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if not results:
        raise ValueError(
            "Nessun risultato da confrontare."
        )

    project_root = Path(__file__).resolve().parent.parent

    experiments_root = Path(
        os.environ.get(
            "EXPERIMENTS_DIR",
            project_root / "experiments",
        )
    )

    comparison_directory = (
        experiments_root
        / f"timesnet_{config_path.stem}"
        / "comparison"
    )

    comparison_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(results)

    dataframe = dataframe.sort_values(
        by="top_k"
    )

    comparison_columns = [
        "top_k",
        "test_mse",
        "test_mae",
        "best_validation_mse",
        "trainable_parameters",
        "average_epoch_time_seconds",
        "total_training_time_seconds",
        "inference_ms_per_sample",
        "samples_per_second",
        "peak_gpu_memory_mb",
        "checkpoint_size_mb",
    ]

    comparison_csv = (
        comparison_directory
        / "comparison_top_k.csv"
    )

    dataframe[
        comparison_columns
    ].to_csv(
        comparison_csv,
        index=False,
    )

    print("\nConfronto finale:")
    print(
        dataframe[
            comparison_columns
        ].to_string(
            index=False
        )
    )

    plot_specs = [
        (
            "test_mse",
            "Test MSE",
            "mse_top_k.png",
        ),
        (
            "test_mae",
            "Test MAE",
            "mae_top_k.png",
        ),
        (
            "average_epoch_time_seconds",
            "Secondi medi per epoca",
            "epoch_time_top_k.png",
        ),
        (
            "inference_ms_per_sample",
            "Inferenza ms/campione",
            "inference_time_top_k.png",
        ),
        (
            "peak_gpu_memory_mb",
            "Peak GPU memory (MB)",
            "gpu_memory_top_k.png",
        ),
    ]

    for metric, ylabel, filename in plot_specs:
        plot_data = dataframe[
            ["top_k", metric]
        ].dropna()

        if plot_data.empty:
            continue

        plt.figure(figsize=(7, 5))

        plt.plot(
            plot_data["top_k"],
            plot_data[metric],
            marker="o",
        )

        plt.xlabel("top_k")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} al variare di top_k")
        plt.xticks(
            plot_data["top_k"]
        )
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            comparison_directory / filename,
            dpi=200,
        )

        plt.close()

    best_row = dataframe.loc[
    dataframe["best_validation_mse"].idxmin()
    ]

    summary = {
        "selection_metric": "best_validation_mse",
        "best_top_k": int(best_row["top_k"]),
        "best_validation_mse": float(
            best_row["best_validation_mse"]
        ),
        "corresponding_test_mse": float(
            best_row["test_mse"]
        ),
        "corresponding_test_mae": float(
            best_row["test_mae"]
        ),
    }

    with (
        comparison_directory
        / "best_top_k.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        "\nMiglior top_k sulla validation: "
        f"{summary['best_top_k']}"
    )

    print(
        "Validation MSE: "
        f"{summary['best_validation_mse']:.6f}"
    )

    print(
        "Test MSE corrispondente: "
        f"{summary['corresponding_test_mse']:.6f}"
    )

    print(
        "Test MAE corrispondente: "
        f"{summary['corresponding_test_mae']:.6f}"
    )

def main():
    args = parse_args()

    config, config_path = load_config(
        args.config
    )

    validate_config(config)

    if args.model != "timesnet":
        raise ValueError(
            "Questa modalità di sweep top-k è "
            "dedicata al modello timesnet."
        )

    top_k_values = sorted(
        set(args.top_k_values)
    )

    if any(k <= 0 for k in top_k_values):
        raise ValueError(
            "Tutti i valori di top_k devono "
            "essere maggiori di zero."
        )

    print(
        "Avvio sweep TimesNet per top_k: "
        f"{top_k_values}"
    )

    all_results = []

    for top_k in top_k_values:
        metrics = run_experiment(
            args=args,
            config=config,
            config_path=config_path,
            top_k=top_k,
        )

        all_results.append(metrics)

    compare_top_k_results(
        results=all_results,
        config_path=config_path,
    )



if __name__ == "__main__":
    main()