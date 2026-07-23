import argparse
import importlib
import inspect
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import optim

from src.data import build_dataloader


# ============================================================
# ARGOMENTI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Training e confronto di modelli per forecasting."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Nome del file YAML dentro configs oppure "
            "percorso completo del file."
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
            "models_1d",
        ],
        help="Famiglia di modello da allenare.",
    )

    parser.add_argument(
        "--model-class",
        type=str,
        default=None,
        help=(
            "Nome della classe Python da caricare. "
            "È particolarmente utile per models_1d.py, "
            "che può contenere più architetture."
        ),
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

    return parser.parse_args()


# ============================================================
# CONFIGURAZIONE
# ============================================================

def resolve_config_path(
    config_argument: str,
) -> Path:
    config_path = Path(config_argument).expanduser()

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
            "Configurazione non trovata: "
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
            "Configurazione YAML non valida: "
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


# ============================================================
# SEED E DEVICE
# ============================================================

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


def synchronize_device(
    device: torch.device,
) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elif device.type == "mps":
        torch.mps.synchronize()


# ============================================================
# CARICAMENTO DINAMICO DEI MODELLI
# ============================================================

def find_model_class(
    module_path: str,
    requested_class: Optional[str] = None,
    preferred_classes: Optional[list[str]] = None,
):
    """
    Cerca una classe nn.Module dentro un modulo.

    Ordine:
    1. classe richiesta con --model-class;
    2. nomi preferiti;
    3. unica classe nn.Module definita nel modulo.
    """
    module = importlib.import_module(
        module_path
    )

    if requested_class is not None:
        if not hasattr(module, requested_class):
            available = [
                name
                for name, obj in inspect.getmembers(
                    module,
                    inspect.isclass,
                )
                if (
                    issubclass(obj, nn.Module)
                    and obj is not nn.Module
                    and obj.__module__ == module.__name__
                )
            ]

            raise AttributeError(
                f"La classe '{requested_class}' non è "
                f"presente in {module_path}. "
                f"Classi disponibili: {available}"
            )

        model_class = getattr(
            module,
            requested_class,
        )

        if not issubclass(model_class, nn.Module):
            raise TypeError(
                f"{requested_class} non è una nn.Module."
            )

        return model_class

    for class_name in preferred_classes or []:
        if hasattr(module, class_name):
            model_class = getattr(
                module,
                class_name,
            )

            if (
                inspect.isclass(model_class)
                and issubclass(model_class, nn.Module)
            ):
                return model_class

    available_classes = [
        obj
        for _, obj in inspect.getmembers(
            module,
            inspect.isclass,
        )
        if (
            issubclass(obj, nn.Module)
            and obj is not nn.Module
            and obj.__module__ == module.__name__
        )
    ]

    if len(available_classes) == 1:
        return available_classes[0]

    available_names = [
        cls.__name__
        for cls in available_classes
    ]

    raise ValueError(
        f"Non è possibile scegliere automaticamente una "
        f"classe da {module_path}. "
        f"Classi disponibili: {available_names}. "
        f"Usa --model-class NOME_CLASSE."
    )


def instantiate_from_signature(
    model_class,
    config: dict,
    top_k: Optional[int] = None,
) -> nn.Module:
    """
    Costruisce una classe usando solo gli argomenti accettati
    dal suo __init__.

    Include alias comuni usati dai modelli 1D e 2D.
    """
    effective_top_k = (
        int(top_k)
        if top_k is not None
        else int(config.get("top_k", 3))
    )

    num_features = int(
        config["num_features"]
    )

    seq_len = int(config["seq_len"])
    pred_len = int(config["pred_len"])

    candidate_kwargs = {
        # Lunghezza temporale
        "seq_len": seq_len,
        "input_len": seq_len,
        "context_len": seq_len,
        "lookback": seq_len,

        # Orizzonte
        "pred_len": pred_len,
        "prediction_len": pred_len,
        "horizon": pred_len,
        "forecast_horizon": pred_len,

        # Numero di feature
        "enc_in": num_features,
        "num_features": num_features,
        "input_size": num_features,
        "input_dim": num_features,
        "n_features": num_features,
        "c_in": num_features,
        "in_channels": num_features,

        # Output multivariato
        "output_size": num_features,
        "output_dim": num_features,
        "c_out": num_features,
        "out_channels": num_features,

        # Dimensioni interne
        "d_model": int(
            config.get("d_model", 32)
        ),
        "d_ff": int(
            config.get("d_ff", 64)
        ),
        "hidden_size": int(
            config.get(
                "hidden_size",
                config.get("d_model", 32),
            )
        ),
        "hidden_dim": int(
            config.get(
                "hidden_dim",
                config.get("d_model", 32),
            )
        ),
        "num_layers": int(
            config.get("num_layers", 1)
        ),

        # Periodicità
        "fixed_period": int(
            config.get("fixed_period", 24)
        ),
        "period": int(
            config.get("fixed_period", 24)
        ),
        "top_k": effective_top_k,

        # Regolarizzazione
        "dropout": float(
            config.get("dropout", 0.1)
        ),

        "groups": int(
            config.get("groups", 4)
        ),

        # Convoluzioni
        "kernel_size": int(
            config.get("kernel_size", 3)
        ),
        "kernel_sizes": tuple(
            config.get(
                "kernel_sizes",
                [1, 3, 5],
            )
        ),

        # Opzioni TimesNet
        "use_fft": bool(
            config.get("use_fft", True)
        ),
        "use_inception": bool(
            config.get("use_inception", True)
        ),
    }

    signature = inspect.signature(
        model_class.__init__
    )

    accepted_kwargs = {}

    accepts_kwargs = any(
        parameter.kind
        == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    for name, value in candidate_kwargs.items():
        if accepts_kwargs or name in signature.parameters:
            accepted_kwargs[name] = value

    missing_required = []

    for name, parameter in signature.parameters.items():
        if name == "self":
            continue

        if (
            parameter.default
            is inspect.Parameter.empty
            and parameter.kind
            not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
            and name not in accepted_kwargs
        ):
            missing_required.append(name)

    if missing_required:
        raise TypeError(
            f"Impossibile costruire "
            f"{model_class.__name__}. "
            f"Argomenti obbligatori non riconosciuti: "
            f"{missing_required}. "
            f"Firma: {signature}"
        )

    print(
        f"Classe selezionata: "
        f"{model_class.__module__}."
        f"{model_class.__name__}"
    )

    print(
        "Argomenti modello:",
        accepted_kwargs,
    )

    return model_class(
        **accepted_kwargs
    )


def build_model(
    model_name: str,
    config: dict,
    top_k: Optional[int] = None,
    requested_class: Optional[str] = None,
) -> nn.Module:

    # --------------------------------------------------------
    # TimesNet originale: src/models_2d.py
    # --------------------------------------------------------
    if model_name == "timesnet":
        from src.models_2d import TimesNet

        effective_top_k = (
            int(top_k)
            if top_k is not None
            else int(config.get("top_k", 3))
        )

        return TimesNet(
            seq_len=int(config["seq_len"]),
            pred_len=int(config["pred_len"]),
            enc_in=int(config["num_features"]),
            d_model=int(
                config.get("d_model", 32)
            ),
            top_k=effective_top_k,
            use_fft=bool(
                config.get("use_fft", True)
            ),
            fixed_period=int(
                config.get("fixed_period", 24)
            ),
            use_inception=bool(
                config.get(
                    "use_inception",
                    True,
                )
            ),
        )

    # --------------------------------------------------------
    # Fixed-period Inception:
    # src/models/fixed_period_inception.py
    # --------------------------------------------------------
    if model_name == "fixed_period_inception":
        from src.models.fixed_period_inception import (
            FixedPeriodInception2D,
        )

        return FixedPeriodInception2D(
            seq_len=int(config["seq_len"]),
            pred_len=int(config["pred_len"]),
            num_features=int(
                config["num_features"]
            ),
            period=int(
                config.get("fixed_period", 24)
            ),
            d_model=int(
                config.get("d_model", 32)
            ),
            d_ff=int(
                config.get("d_ff", 64)
            ),
            kernel_sizes=tuple(
                config.get(
                    "kernel_sizes",
                    [1, 3, 5],
                )
            ),
            dropout=float(
                config.get("dropout", 0.1)
            ),
        )

    # --------------------------------------------------------
    # Light Depthwise:
    # src/models/models_light_depthwise.py
    # --------------------------------------------------------
    if model_name == "timesnet_light_depthwise":
        model_class = find_model_class(
            module_path=(
                "src.models.models_light_depthwise"
            ),
            requested_class=requested_class,
            preferred_classes=[
                "LightTimesNetDepthwise",
                "LightTimesNet",
                "TimesNetLightDepthwise",
            ],
        )

        return instantiate_from_signature(
            model_class=model_class,
            config=config,
            top_k=top_k,
        )

    # --------------------------------------------------------
    # TimesNet Light:
    # src/models_light.py
    # --------------------------------------------------------
    if model_name == "timesnet_light":
        model_class = find_model_class(
            module_path="src.models_light",
            requested_class=requested_class,
            preferred_classes=[
                "LightTimesNet",
                "TimesNetLight",
                "LightTimesNetModel",
            ],
        )

        return instantiate_from_signature(
            model_class=model_class,
            config=config,
            top_k=top_k,
        )

    # --------------------------------------------------------
    # Modelli 1D:
    # src/models_1d.py
    # --------------------------------------------------------
    if model_name == "models_1d":
        model_class = find_model_class(
            module_path="src.models_1d",
            requested_class=requested_class,
            preferred_classes=[],
        )

        return instantiate_from_signature(
            model_class=model_class,
            config=config,
            top_k=top_k,
        )

    raise ValueError(
        f"Modello non riconosciuto: {model_name}"
    )


# ============================================================
# INFORMAZIONI SUL MODELLO
# ============================================================

def get_model_fixed_period(
    model: nn.Module,
) -> Optional[int]:
    if hasattr(model, "period"):
        return int(model.period)

    if hasattr(model, "fixed_period"):
        return int(model.fixed_period)

    if (
        hasattr(model, "times_block")
        and hasattr(
            model.times_block,
            "fixed_period",
        )
    ):
        return int(
            model.times_block.fixed_period
        )

    return None


def get_experiment_model_name(
    model_name: str,
    requested_class: Optional[str],
) -> str:
    if (
        model_name == "models_1d"
        and requested_class is not None
    ):
        safe_class_name = (
            requested_class
            .replace(" ", "_")
            .lower()
        )

        return (
            f"{model_name}_{safe_class_name}"
        )

    return model_name


# ============================================================
# CARTELLE
# ============================================================

def create_experiment_directory(
    model_name: str,
    config_path: Path,
    model: nn.Module,
    top_k: Optional[int] = None,
    requested_class: Optional[str] = None,
) -> Path:
    project_root = (
        Path(__file__).resolve().parent.parent
    )

    experiments_root = Path(
        os.environ.get(
            "EXPERIMENTS_DIR",
            project_root / "experiments",
        )
    )

    experiment_model_name = (
        get_experiment_model_name(
            model_name=model_name,
            requested_class=requested_class,
        )
    )

    base_directory = (
        experiments_root
        / f"{experiment_model_name}_"
          f"{config_path.stem}"
    )

    if model_name == "timesnet":
        if top_k is None:
            raise ValueError(
                "top_k deve essere specificato "
                "per TimesNet."
            )

        experiment_directory = (
            base_directory
            / f"top_k_{int(top_k)}"
        )

    else:
        fixed_period = get_model_fixed_period(
            model
        )

        if fixed_period is not None:
            experiment_directory = (
                base_directory
                / f"period_{fixed_period}"
            )
        else:
            experiment_directory = (
                base_directory
            )

    experiment_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return experiment_directory


# ============================================================
# TRAINING E VALUTAZIONE
# ============================================================

def run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
) -> float:
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
                optimizer.zero_grad(
                    set_to_none=True
                )

            predictions = model(batch_x)

            if predictions.shape != batch_y.shape:
                raise RuntimeError(
                    "Shape non compatibili: "
                    f"prediction={predictions.shape}, "
                    f"target={batch_y.shape}. "
                    "Il modello deve restituire "
                    "[B, pred_len, num_features]."
                )

            loss = criterion(
                predictions,
                batch_y,
            )

            if is_training:
                loss.backward()

                gradient_clip = getattr(
                    model,
                    "gradient_clip",
                    None,
                )

                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        gradient_clip,
                    )

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

            if predictions.shape != batch_y.shape:
                raise RuntimeError(
                    "Shape test non compatibili: "
                    f"prediction={predictions.shape}, "
                    f"target={batch_y.shape}."
                )

            error = predictions - batch_y

            squared_error_sum += (
                error.pow(2).sum().item()
            )

            absolute_error_sum += (
                error.abs().sum().item()
            )

            element_count += batch_y.numel()

    if element_count == 0:
        raise RuntimeError(
            "Il test DataLoader non contiene elementi."
        )

    mse = squared_error_sum / element_count
    mae = absolute_error_sum / element_count

    return mse, mae


def measure_inference_time(
    model: nn.Module,
    dataloader,
    device: torch.device,
    warmup_batches: int = 5,
    max_batches: int = 30,
) -> dict:
    model.eval()

    warmup_count = 0

    with torch.no_grad():
        for batch_x, _ in dataloader:
            if warmup_count >= warmup_batches:
                break

            batch_x = batch_x.to(
                device,
                non_blocking=True,
            )

            _ = model(batch_x)
            warmup_count += 1

    synchronize_device(device)

    measured_batches = 0
    total_samples = 0

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch_x, _ in dataloader:
            if measured_batches >= max_batches:
                break

            batch_x = batch_x.to(
                device,
                non_blocking=True,
            )

            _ = model(batch_x)

            measured_batches += 1
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


# ============================================================
# SINGOLO ESPERIMENTO
# ============================================================

def run_experiment(
    args,
    config: dict,
    config_path: Path,
    top_k: Optional[int] = None,
) -> dict:
    set_seed(int(config["seed"]))

    device = get_device()

    _, train_loader = build_dataloader(
        config,
        flag="train",
    )

    _, val_loader = build_dataloader(
        config,
        flag="val",
    )

    _, test_loader = build_dataloader(
        config,
        flag="test",
    )

    model = build_model(
        model_name=args.model,
        config=config,
        top_k=top_k,
        requested_class=args.model_class,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(
            config.get("weight_decay", 0.0)
        ),
    )

    criterion = nn.MSELoss()

    experiment_directory = (
        create_experiment_directory(
            model_name=args.model,
            config_path=config_path,
            model=model,
            top_k=top_k,
            requested_class=args.model_class,
        )
    )

    checkpoint_path = (
        experiment_directory
        / "best_model.pth"
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    effective_period = get_model_fixed_period(
        model
    )

    config_used = dict(config)

    if top_k is not None:
        config_used["top_k"] = int(top_k)

    if effective_period is not None:
        config_used["fixed_period"] = int(
            effective_period
        )

    config_used["selected_model"] = args.model

    if args.model_class is not None:
        config_used["selected_model_class"] = (
            args.model_class
        )

    with (
        experiment_directory
        / "config_used.yaml"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config_used,
            file,
            sort_keys=False,
        )

    print("\n" + "=" * 72)
    print(f"Modello: {args.model}")

    if args.model_class is not None:
        print(
            f"Classe: {args.model_class}"
        )

    print(
        f"Dataset: {config['dataset_name']}"
    )
    print(f"seq_len: {config['seq_len']}")
    print(f"pred_len: {config['pred_len']}")

    if top_k is not None:
        print(f"top_k: {top_k}")

    if effective_period is not None:
        print(
            f"fixed_period: {effective_period}"
        )

    print(f"Device: {device}")
    print(f"Parametri: {parameter_count:,}")
    print(f"Output: {experiment_directory}")
    print("=" * 72)

    # Controllo shape prima del training
    sample_x, sample_y = next(
        iter(train_loader)
    )

    model.eval()

    with torch.no_grad():
        sample_prediction = model(
            sample_x.to(device)
        ).cpu()

    if sample_prediction.shape != sample_y.shape:
        raise RuntimeError(
            "Shape non compatibili prima del training: "
            f"prediction={sample_prediction.shape}, "
            f"target={sample_y.shape}. "
            "Il modello deve restituire "
            "[B, pred_len, num_features]."
        )

    best_validation_loss = float("inf")

    patience = int(
        config.get("patience", 5)
    )

    epochs_without_improvement = 0

    history = {
        "train_mse": [],
        "val_mse": [],
        "epoch_time_seconds": [],
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(
            device
        )

    synchronize_device(device)

    training_start = time.perf_counter()

    for epoch in range(
        1,
        int(config["epochs"]) + 1,
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
            time.perf_counter()
            - epoch_start
        )

        history["train_mse"].append(
            float(train_mse)
        )

        history["val_mse"].append(
            float(val_mse)
        )

        history[
            "epoch_time_seconds"
        ].append(
            float(epoch_time)
        )

        experiment_label = (
            f"top_k={top_k}"
            if top_k is not None
            else args.model
        )

        print(
            f"{experiment_label} | "
            f"Epoch {epoch:03d}/"
            f"{config['epochs']} | "
            f"time={epoch_time:.2f}s | "
            f"train MSE={train_mse:.6f} | "
            f"val MSE={val_mse:.6f}"
        )

        if val_mse < best_validation_loss:
            best_validation_loss = float(
                val_mse
            )

            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

            print(
                "Nessun miglioramento: "
                f"{epochs_without_improvement}/"
                f"{patience}"
            )

            if (
                epochs_without_improvement
                >= patience
            ):
                print("Early stopping.")
                break

    synchronize_device(device)

    total_training_time = (
        time.perf_counter()
        - training_start
    )

    completed_epochs = len(
        history["epoch_time_seconds"]
    )

    if completed_epochs == 0:
        raise RuntimeError(
            "Nessuna epoca completata."
        )

    average_epoch_time = float(
        np.mean(
            history["epoch_time_seconds"]
        )
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
        warmup_batches=int(
            config.get(
                "inference_warmup_batches",
                5,
            )
        ),
        max_batches=int(
            config.get(
                "inference_measure_batches",
                30,
            )
        ),
    )

    checkpoint_size_mb = (
        checkpoint_path.stat().st_size
        / 1024**2
    )

    if device.type == "cuda":
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated(
                device
            )
            / 1024**2
        )

        peak_gpu_reserved_mb = (
            torch.cuda.max_memory_reserved(
                device
            )
            / 1024**2
        )
    else:
        peak_gpu_memory_mb = None
        peak_gpu_reserved_mb = None

    metrics = {
        "model": args.model,
        "model_class": (
            model.__class__.__name__
        ),
        "config": config_path.stem,
        "dataset": config["dataset_name"],

        "seq_len": int(config["seq_len"]),
        "pred_len": int(config["pred_len"]),
        "batch_size": int(
            config["batch_size"]
        ),

        "top_k": (
            int(top_k)
            if top_k is not None
            else None
        ),

        "fixed_period": (
            int(effective_period)
            if effective_period is not None
            else None
        ),

        "use_fft": (
            bool(config.get("use_fft", True))
            if args.model == "timesnet"
            else None
        ),

        "use_inception": (
            bool(
                config.get(
                    "use_inception",
                    True,
                )
            )
            if args.model == "timesnet"
            else None
        ),
        "backbone_2d": getattr(
            model,
            "block_type",
            None,
        ),

        "kernel_sizes": (
            config.get("kernel_sizes")
            if getattr(model, "block_type", None)
            == "multiscale"
            else None
        ),

        "kernel_size": (
            config.get("kernel_size", 3)
            if getattr(model, "block_type", None)
            == "single_kernel"
            else None
        ),

        "groups": (
            config.get("groups", 4)
            if getattr(model, "block_type", None)
            == "group"
            else None
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

        "inference_measured_batches": int(
            inference_stats[
                "measured_batches"
            ]
        ),

        "inference_measured_samples": int(
            inference_stats[
                "measured_samples"
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
        experiment_directory
        / "metrics.json"
    )

    history_path = (
        experiment_directory
        / "history.json"
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

    print("\nEsperimento completato")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(
        "Tempo medio per epoca: "
        f"{average_epoch_time:.3f}s"
    )
    print(
        "Inferenza: "
        f"{inference_stats['inference_ms_per_sample']:.6f} "
        "ms/campione"
    )
    print(
        f"Metriche salvate in: {metrics_path}"
    )

    return metrics


# ============================================================
# CONFRONTO TOP-K
# ============================================================

def compare_top_k_results(
    results: list[dict],
    config_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd

    if not results:
        raise ValueError(
            "Nessun risultato da confrontare."
        )

    project_root = (
        Path(__file__).resolve().parent.parent
    )

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

    dataframe = pd.DataFrame(
        results
    ).sort_values(
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

    print("\nConfronto finale:\n")

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
            "best_validation_mse",
            "Validation MSE",
            "validation_mse_top_k.png",
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

        plt.title(
            f"{ylabel} al variare di top_k"
        )

        plt.xticks(
            plot_data["top_k"]
        )

        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            comparison_directory
            / filename,
            dpi=200,
        )

        plt.close()

    # Selezione iperparametro sulla validation
    best_row = dataframe.loc[
        dataframe[
            "best_validation_mse"
        ].idxmin()
    ]

    summary = {
        "selection_metric": (
            "best_validation_mse"
        ),
        "best_top_k": int(
            best_row["top_k"]
        ),
        "best_validation_mse": float(
            best_row[
                "best_validation_mse"
            ]
        ),
        "corresponding_test_mse": float(
            best_row["test_mse"]
        ),
        "corresponding_test_mae": float(
            best_row["test_mae"]
        ),
    }

    summary_path = (
        comparison_directory
        / "best_top_k.json"
    )

    with summary_path.open(
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

    print(
        f"Confronto salvato in: "
        f"{comparison_directory}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    config, config_path = load_config(
        args.config
    )

    validate_config(config)

    # TimesNet originale: sweep top-k
    if args.model == "timesnet":
        top_k_values = sorted(
            set(args.top_k_values)
        )

        if not top_k_values:
            raise ValueError(
                "Devi specificare almeno "
                "un valore di top_k."
            )

        if any(k <= 0 for k in top_k_values):
            raise ValueError(
                "Tutti i valori di top_k "
                "devono essere positivi."
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

        return

    # Tutti gli altri modelli:
    # un singolo esperimento
    print(
        f"Avvio singolo esperimento: "
        f"{args.model}"
    )

    metrics = run_experiment(
        args=args,
        config=config,
        config_path=config_path,
        top_k=None,
    )

    print("\nRiepilogo:")
    print(f"Modello: {metrics['model']}")
    print(
        f"Classe: {metrics['model_class']}"
    )
    print(
        f"Test MSE: {metrics['test_mse']:.6f}"
    )
    print(
        f"Test MAE: {metrics['test_mae']:.6f}"
    )


if __name__ == "__main__":
    main()