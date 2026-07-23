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
# ARGOMENTI DA TERMINALE
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
            "timesnet_light",
            "models_1d",
            "dlinear",
            "tcn",
        ],
        help=(
            "Famiglia di modello da allenare."
        ),
    )

    parser.add_argument(
        "--model-class",
        type=str,
        default=None,
        help=(
            "Classe concreta da caricare. "
            "Esempi: LightTimesNetMultiScale, "
            "LightTimesNetDepthwise, "
            "LightTimesNetGroup, "
            "LightTimesNetSingleKernel, "
            "DLinear, CausalTCN."
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

    parser.add_argument(
        "--num-blocks-values",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help=(
            "Numero di TimesBlock da provare con "
            "fixed_period_inception. "
            "Default: 1 2 3."
        ),
    )

    return parser.parse_args()


# ============================================================
# CONFIGURAZIONE YAML
# ============================================================

def resolve_config_path(
    config_argument: str,
) -> Path:
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
        if float(config[key]) <= 0:
            raise ValueError(
                f"{key} deve essere positivo."
            )

    if "groups" in config:
        if int(config["groups"]) <= 0:
            raise ValueError(
                "groups deve essere positivo."
            )

    if "kernel_size" in config:
        if int(config["kernel_size"]) <= 0:
            raise ValueError(
                "kernel_size deve essere positivo."
            )

    if "num_channels" in config:
        num_channels = config[
            "num_channels"
        ]

        if not isinstance(
            num_channels,
            (list, tuple),
        ):
            raise TypeError(
                "num_channels deve essere una lista, "
                "per esempio [32, 64]."
            )

        if not num_channels:
            raise ValueError(
                "num_channels non può essere vuoto."
            )

        if any(
            int(channel) <= 0
            for channel in num_channels
        ):
            raise ValueError(
                "num_channels deve contenere "
                "soltanto valori positivi."
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
# CARICAMENTO DINAMICO DELLE CLASSI
# ============================================================

def list_module_model_classes(
    module,
) -> list[str]:
    return [
        name
        for name, obj in inspect.getmembers(
            module,
            inspect.isclass,
        )
        if (
            issubclass(obj, nn.Module)
            and obj is not nn.Module
            and obj.__module__
            == module.__name__
        )
    ]


def find_model_class(
    module_path: str,
    requested_class: Optional[str] = None,
    preferred_classes: Optional[list[str]] = None,
):
    module = importlib.import_module(
        module_path
    )

    available_names = (
        list_module_model_classes(
            module
        )
    )

    if requested_class is not None:
        if not hasattr(
            module,
            requested_class,
        ):
            raise AttributeError(
                f"La classe '{requested_class}' "
                f"non è presente in {module_path}. "
                f"Classi disponibili: "
                f"{available_names}"
            )

        model_class = getattr(
            module,
            requested_class,
        )

        if not inspect.isclass(
            model_class
        ):
            raise TypeError(
                f"{requested_class} non è una classe."
            )

        if not issubclass(
            model_class,
            nn.Module,
        ):
            raise TypeError(
                f"{requested_class} non è "
                "una sottoclasse di nn.Module."
            )

        return model_class

    for class_name in (
        preferred_classes or []
    ):
        if class_name in available_names:
            return getattr(
                module,
                class_name,
            )

    available_classes = [
        getattr(module, name)
        for name in available_names
    ]

    if len(available_classes) == 1:
        return available_classes[0]

    raise ValueError(
        "Non è possibile scegliere automaticamente "
        f"una classe da {module_path}. "
        f"Classi disponibili: {available_names}. "
        "Usa --model-class NOME_CLASSE."
    )


# ============================================================
# COSTRUZIONE DINAMICA
# ============================================================

def instantiate_from_signature(
    model_class,
    config: dict,
    top_k: Optional[int] = None,
) -> nn.Module:
    seq_len = int(
        config["seq_len"]
    )

    pred_len = int(
        config["pred_len"]
    )

    num_features = int(
        config["num_features"]
    )

    effective_top_k = (
        int(top_k)
        if top_k is not None
        else int(
            config.get(
                "top_k",
                3,
            )
        )
    )

    num_channels = [
        int(channel)
        for channel in config.get(
            "num_channels",
            [32, 64],
        )
    ]

    candidate_kwargs = {
        # Lunghezza input
        "seq_len": seq_len,
        "input_len": seq_len,
        "context_len": seq_len,
        "lookback": seq_len,

        # Orizzonte
        "pred_len": pred_len,
        "prediction_len": pred_len,
        "horizon": pred_len,
        "forecast_horizon": pred_len,

        # Feature input
        "enc_in": num_features,
        "num_features": num_features,
        "input_size": num_features,
        "input_dim": num_features,
        "n_features": num_features,
        "c_in": num_features,
        "in_channels": num_features,

        # Feature output
        "output_size": num_features,
        "output_dim": num_features,
        "c_out": num_features,
        "out_channels": num_features,

        # Dimensioni interne
        "d_model": int(
            config.get(
                "d_model",
                32,
            )
        ),
        "d_ff": int(
            config.get(
                "d_ff",
                64,
            )
        ),
        "hidden_size": int(
            config.get(
                "hidden_size",
                config.get(
                    "d_model",
                    32,
                ),
            )
        ),
        "hidden_dim": int(
            config.get(
                "hidden_dim",
                config.get(
                    "d_model",
                    32,
                ),
            )
        ),
        "num_layers": int(
            config.get(
                "num_layers",
                1,
            )
        ),

        # TCN
        "num_channels": num_channels,

        # Periodicità
        "fixed_period": int(
            config.get(
                "fixed_period",
                24,
            )
        ),
        "period": int(
            config.get(
                "fixed_period",
                24,
            )
        ),
        "top_k": effective_top_k,

        # Regolarizzazione
        "dropout": float(
            config.get(
                "dropout",
                0.1,
            )
        ),

        # Convoluzioni
        "kernel_size": int(
            config.get(
                "kernel_size",
                3,
            )
        ),
        "kernel_sizes": tuple(
            int(kernel)
            for kernel in config.get(
                "kernel_sizes",
                [1, 3, 5],
            )
        ),
        "groups": int(
            config.get(
                "groups",
                4,
            )
        ),

        # TimesNet
        "use_fft": bool(
            config.get(
                "use_fft",
                True,
            )
        ),
        "use_inception": bool(
            config.get(
                "use_inception",
                True,
            )
        ),
    }

    signature = inspect.signature(
        model_class.__init__
    )

    accepts_kwargs = any(
        parameter.kind
        == inspect.Parameter.VAR_KEYWORD
        for parameter
        in signature.parameters.values()
    )

    accepted_kwargs = {}

    for name, value in (
        candidate_kwargs.items()
    ):
        if (
            accepts_kwargs
            or name in signature.parameters
        ):
            accepted_kwargs[name] = value

    missing_required = []

    for name, parameter in (
        signature.parameters.items()
    ):
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
            missing_required.append(
                name
            )

    if missing_required:
        raise TypeError(
            f"Impossibile costruire "
            f"{model_class.__name__}. "
            f"Argomenti obbligatori non riconosciuti: "
            f"{missing_required}. "
            f"Firma: {signature}"
        )

    print(
        "Classe selezionata: "
        f"{model_class.__module__}."
        f"{model_class.__name__}"
    )

    print(
        "Argomenti passati al modello:",
        accepted_kwargs,
    )

    return model_class(
        **accepted_kwargs
    )


# ============================================================
# FACTORY DEI MODELLI
# ============================================================

def build_model(
    model_name: str,
    config: dict,
    top_k: Optional[int] = None,
    requested_class: Optional[str] = None,
    num_blocks: Optional[int] = None,
) -> nn.Module:

    # --------------------------------------------------------
    # TIMESNET ORIGINALE
    # --------------------------------------------------------
    if model_name == "timesnet":
        from src.models_2d import TimesNet

        effective_top_k = (
            int(top_k)
            if top_k is not None
            else int(
                config.get(
                    "top_k",
                    3,
                )
            )
        )

        return TimesNet(
            seq_len=int(
                config["seq_len"]
            ),
            pred_len=int(
                config["pred_len"]
            ),
            enc_in=int(
                config["num_features"]
            ),
            d_model=int(
                config.get(
                    "d_model",
                    32,
                )
            ),
            top_k=effective_top_k,
            use_fft=bool(
                config.get(
                    "use_fft",
                    True,
                )
            ),
            fixed_period=int(
                config.get(
                    "fixed_period",
                    24,
                )
            ),
            use_inception=bool(
                config.get(
                    "use_inception",
                    True,
                )
            ),
        )

    # --------------------------------------------------------
    # FIXED PERIOD INCEPTION
    # --------------------------------------------------------
    if model_name == "fixed_period_inception":
        from src.models.fixed_period_inception import (
            FixedPeriodInception2D,
        )

        effective_num_blocks = (
            int(num_blocks)
            if num_blocks is not None
            else int(
                config.get(
                    "num_blocks",
                    1,
                )
            )
        )

        return FixedPeriodInception2D(
            seq_len=int(
                config["seq_len"]
            ),
            pred_len=int(
                config["pred_len"]
            ),
            num_features=int(
                config["num_features"]
            ),
            period=int(
                config.get(
                    "fixed_period",
                    24,
                )
            ),
            d_model=int(
                config.get(
                    "d_model",
                    32,
                )
            ),
            d_ff=int(
                config.get(
                    "d_ff",
                    64,
                )
            ),
            kernel_sizes=tuple(
                int(kernel)
                for kernel in config.get(
                    "kernel_sizes",
                    [1, 3, 5],
                )
            ),
            dropout=float(
                config.get(
                    "dropout",
                    0.1,
                )
            ),
            num_blocks=effective_num_blocks,
        )

    # --------------------------------------------------------
    # TIMESNET LIGHT
    # --------------------------------------------------------
    if model_name == "timesnet_light":
        allowed_light_classes = {
            "LightTimesNet",
            "LightTimesNetMultiScale",
            "LightTimesNetDepthwise",
            "LightTimesNetGroup",
            "LightTimesNetSingleKernel",
        }

        if (
            requested_class is not None
            and requested_class
            not in allowed_light_classes
        ):
            raise ValueError(
                "Classe TimesNet Light non supportata: "
                f"{requested_class}. "
                f"Classi disponibili: "
                f"{sorted(allowed_light_classes)}"
            )

        model_class = find_model_class(
            module_path="src.models_light",
            requested_class=requested_class,
            preferred_classes=[
                "LightTimesNetMultiScale",
                "LightTimesNet",
            ],
        )

        return instantiate_from_signature(
            model_class=model_class,
            config=config,
            top_k=None,
        )

    # --------------------------------------------------------
    # DLINEAR
    # --------------------------------------------------------
    if model_name == "dlinear":
        if (
            requested_class is not None
            and requested_class != "DLinear"
        ):
            raise ValueError(
                "Con --model dlinear la classe "
                "deve essere DLinear."
            )

        model_class = find_model_class(
            module_path="src.models_1d",
            requested_class="DLinear",
        )

        return instantiate_from_signature(
            model_class=model_class,
            config=config,
        )

    # --------------------------------------------------------
    # TCN
    # --------------------------------------------------------
    if model_name == "tcn":
        if (
            requested_class is not None
            and requested_class != "CausalTCN"
        ):
            raise ValueError(
                "Con --model tcn la classe "
                "deve essere CausalTCN."
            )

        model_class = find_model_class(
            module_path="src.models_1d",
            requested_class="CausalTCN",
        )

        return instantiate_from_signature(
            model_class=model_class,
            config=config,
        )

    # --------------------------------------------------------
    # MODELLI 1D GENERICI
    # --------------------------------------------------------
    if model_name == "models_1d":
        allowed_1d_classes = {
            "DLinear",
            "CausalTCN",
        }

        if requested_class is None:
            raise ValueError(
                "Con --model models_1d devi specificare "
                "--model-class DLinear oppure "
                "--model-class CausalTCN."
            )

        if requested_class not in allowed_1d_classes:
            raise ValueError(
                f"Classe 1D non supportata: "
                f"{requested_class}. "
                f"Classi disponibili: "
                f"{sorted(allowed_1d_classes)}"
            )

        model_class = find_model_class(
            module_path="src.models_1d",
            requested_class=requested_class,
        )

        return instantiate_from_signature(
            model_class=model_class,
            config=config,
        )

    raise ValueError(
        f"Modello non riconosciuto: "
        f"{model_name}"
    )


# ============================================================
# INFORMAZIONI MODELLO
# ============================================================

def get_model_fixed_period(
    model: nn.Module,
) -> Optional[int]:
    if hasattr(model, "period"):
        return int(
            model.period
        )

    if hasattr(model, "fixed_period"):
        return int(
            model.fixed_period
        )

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


def get_model_num_blocks(
    model: nn.Module,
) -> Optional[int]:
    if hasattr(model, "num_blocks"):
        return int(
            model.num_blocks
        )

    if hasattr(model, "times_blocks"):
        try:
            return int(
                len(model.times_blocks)
            )
        except TypeError:
            return None

    return None


def get_model_block_type(
    model: nn.Module,
) -> Optional[str]:
    if hasattr(model, "block_type"):
        return str(
            model.block_type
        )

    if (
        hasattr(model, "times_block")
        and hasattr(
            model.times_block,
            "block_type",
        )
    ):
        return str(
            model.times_block.block_type
        )

    return None


def sanitize_name(
    name: str,
) -> str:
    return (
        str(name)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


def get_experiment_model_name(
    model_name: str,
    requested_class: Optional[str],
    model: nn.Module,
) -> str:
    if model_name == "timesnet":
        return "timesnet"

    class_name = (
        requested_class
        if requested_class is not None
        else model.__class__.__name__
    )

    return (
        f"{sanitize_name(model_name)}_"
        f"{sanitize_name(class_name)}"
    )


# ============================================================
# DIRECTORY ESPERIMENTI
# ============================================================

def create_experiment_directory(
    model_name: str,
    config_path: Path,
    model: nn.Module,
    top_k: Optional[int] = None,
    requested_class: Optional[str] = None,
    num_blocks: Optional[int] = None,
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

    config_name = sanitize_name(
        config_path.stem
    )

    # --------------------------------------------------------
    # TIMESNET TOP-K
    # --------------------------------------------------------
    if model_name == "timesnet":
        if top_k is None:
            raise ValueError(
                "top_k deve essere specificato "
                "per TimesNet."
            )

        experiment_directory = (
            experiments_root
            / f"timesnet_{config_name}"
            / f"top_k_{int(top_k)}"
        )

    # --------------------------------------------------------
    # FIXED PERIOD: STUDIO NUMERO TIMESBLOCK
    # --------------------------------------------------------
    elif model_name == "fixed_period_inception":
        fixed_period = (
            get_model_fixed_period(
                model
            )
        )

        if fixed_period is None:
            raise ValueError(
                "Impossibile determinare fixed_period "
                "per FixedPeriodInception2D."
            )

        effective_num_blocks = (
            get_model_num_blocks(
                model
            )
        )

        if effective_num_blocks is None:
            effective_num_blocks = (
                int(num_blocks)
                if num_blocks is not None
                else 1
            )

        experiment_directory = (
            experiments_root
            / (
                "fixed_period_inception_timesblocks_"
                f"{config_name}"
            )
            / f"period_{int(fixed_period)}"
            / (
                "num_times_blocks_"
                f"{int(effective_num_blocks)}"
            )
        )

    # --------------------------------------------------------
    # ALTRI MODELLI
    # --------------------------------------------------------
    else:
        experiment_model_name = (
            get_experiment_model_name(
                model_name=model_name,
                requested_class=requested_class,
                model=model,
            )
        )

        base_directory = (
            experiments_root
            / (
                f"{experiment_model_name}_"
                f"{config_name}"
            )
        )

        fixed_period = (
            get_model_fixed_period(
                model
            )
        )

        if fixed_period is not None:
            experiment_directory = (
                base_directory
                / f"period_{int(fixed_period)}"
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
# TRAINING DI UNA EPOCA
# ============================================================

def run_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    optimizer=None,
) -> float:
    is_training = (
        optimizer is not None
    )

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

            predictions = model(
                batch_x
            )

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
                loss.item()
                * batch_size
            )

            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError(
            "Il DataLoader non contiene campioni."
        )

    return (
        total_loss
        / total_samples
    )


# ============================================================
# TEST
# ============================================================

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

            predictions = model(
                batch_x
            )

            if predictions.shape != batch_y.shape:
                raise RuntimeError(
                    "Shape test non compatibili: "
                    f"prediction={predictions.shape}, "
                    f"target={batch_y.shape}."
                )

            error = (
                predictions
                - batch_y
            )

            squared_error_sum += (
                error.pow(2)
                .sum()
                .item()
            )

            absolute_error_sum += (
                error.abs()
                .sum()
                .item()
            )

            element_count += (
                batch_y.numel()
            )

    if element_count == 0:
        raise RuntimeError(
            "Il test DataLoader non contiene elementi."
        )

    mse = (
        squared_error_sum
        / element_count
    )

    mae = (
        absolute_error_sum
        / element_count
    )

    return mse, mae


# ============================================================
# TEMPO INFERENZA
# ============================================================

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

    synchronize_device(
        device
    )

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
            total_samples += (
                batch_x.size(0)
            )

    synchronize_device(
        device
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    if (
        measured_batches == 0
        or total_samples == 0
    ):
        raise RuntimeError(
            "Nessun batch misurato durante "
            "l'inferenza."
        )

    return {
        "measured_batches": int(
            measured_batches
        ),
        "measured_samples": int(
            total_samples
        ),
        "total_inference_seconds": float(
            elapsed_seconds
        ),
        "inference_ms_per_batch": float(
            elapsed_seconds
            / measured_batches
            * 1000
        ),
        "inference_ms_per_sample": float(
            elapsed_seconds
            / total_samples
            * 1000
        ),
        "samples_per_second": float(
            total_samples
            / elapsed_seconds
        ),
    }


# ============================================================
# SINGO ESPERIMENTO
# ============================================================

def run_experiment(
    args,
    config: dict,
    config_path: Path,
    top_k: Optional[int] = None,
    num_blocks: Optional[int] = None,
) -> dict:
    set_seed(
        int(config["seed"])
    )

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
        num_blocks=num_blocks,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(
            config["learning_rate"]
        ),
        weight_decay=float(
            config.get(
                "weight_decay",
                0.0,
            )
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
            num_blocks=num_blocks,
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

    effective_period = (
        get_model_fixed_period(
            model
        )
    )

    effective_num_blocks = (
        get_model_num_blocks(
            model
        )
    )

    block_type = (
        get_model_block_type(
            model
        )
    )

    # --------------------------------------------------------
    # CONFIGURAZIONE EFFETTIVAMENTE UTILIZZATA
    # --------------------------------------------------------
    config_used = dict(
        config
    )

    config_used["selected_model"] = (
        args.model
    )

    config_used["selected_model_class"] = (
        model.__class__.__name__
    )

    config_used["backbone_2d"] = (
        block_type
    )

    if top_k is not None:
        config_used["top_k"] = int(
            top_k
        )

    if effective_period is not None:
        config_used["fixed_period"] = int(
            effective_period
        )

    if effective_num_blocks is not None:
        config_used["num_blocks"] = int(
            effective_num_blocks
        )

    if (
        model.__class__.__name__
        == "CausalTCN"
    ):
        config_used["num_channels"] = [
            int(channel)
            for channel in config.get(
                "num_channels",
                [32, 64],
            )
        ]

        config_used["kernel_size"] = int(
            config.get(
                "kernel_size",
                3,
            )
        )

        config_used["dropout"] = float(
            config.get(
                "dropout",
                0.2,
            )
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

    # --------------------------------------------------------
    # RIEPILOGO
    # --------------------------------------------------------
    print("\n" + "=" * 72)

    print(
        f"Modello: {args.model}"
    )

    print(
        "Classe: "
        f"{model.__class__.__name__}"
    )

    if block_type is not None:
        print(
            f"Backbone 2D: {block_type}"
        )

    print(
        f"Dataset: "
        f"{config['dataset_name']}"
    )

    print(
        f"seq_len: "
        f"{config['seq_len']}"
    )

    print(
        f"pred_len: "
        f"{config['pred_len']}"
    )

    if top_k is not None:
        print(
            f"top_k: {top_k}"
        )

    if effective_period is not None:
        print(
            "fixed_period: "
            f"{effective_period}"
        )

    if effective_num_blocks is not None:
        print(
            "Numero TimesBlock: "
            f"{effective_num_blocks}"
        )

    print(
        f"Device: {device}"
    )

    print(
        "Parametri allenabili: "
        f"{parameter_count:,}"
    )

    print(
        f"Output: "
        f"{experiment_directory}"
    )

    print("=" * 72)

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------
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
            "Shape non compatibili prima "
            "del training: "
            f"prediction={sample_prediction.shape}, "
            f"target={sample_y.shape}. "
            "Il modello deve restituire "
            "[B, pred_len, num_features]."
        )

    best_validation_loss = float(
        "inf"
    )

    patience = int(
        config.get(
            "patience",
            5,
        )
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

    synchronize_device(
        device
    )

    training_start = time.perf_counter()

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------
    for epoch in range(
        1,
        int(config["epochs"]) + 1,
    ):
        synchronize_device(
            device
        )

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

        synchronize_device(
            device
        )

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

        if top_k is not None:
            experiment_label = (
                f"top_k={top_k}"
            )

        elif effective_num_blocks is not None:
            experiment_label = (
                "num_times_blocks="
                f"{effective_num_blocks}"
            )

        else:
            experiment_label = (
                model.__class__.__name__
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
                print(
                    "Early stopping."
                )
                break

    synchronize_device(
        device
    )

    total_training_time = (
        time.perf_counter()
        - training_start
    )

    completed_epochs = len(
        history[
            "epoch_time_seconds"
        ]
    )

    if completed_epochs == 0:
        raise RuntimeError(
            "Nessuna epoca completata."
        )

    average_epoch_time = float(
        np.mean(
            history[
                "epoch_time_seconds"
            ]
        )
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Il checkpoint migliore non è "
            f"stato creato: {checkpoint_path}"
        )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------
    test_mse, test_mae = (
        evaluate_test(
            model=model,
            dataloader=test_loader,
            device=device,
        )
    )

    # --------------------------------------------------------
    # INFERENZA
    # --------------------------------------------------------
    inference_stats = (
        measure_inference_time(
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

    model_class_name = (
        model.__class__.__name__
    )

    # --------------------------------------------------------
    # METRICHE
    # --------------------------------------------------------
    metrics = {
        "model": args.model,
        "model_class": model_class_name,
        "backbone_2d": block_type,

        "config": config_path.stem,
        "dataset": config["dataset_name"],

        "seq_len": int(
            config["seq_len"]
        ),
        "pred_len": int(
            config["pred_len"]
        ),
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

        "num_blocks": (
            int(effective_num_blocks)
            if effective_num_blocks is not None
            else None
        ),

        "use_fft": (
            bool(
                config.get(
                    "use_fft",
                    True,
                )
            )
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

        "kernel_sizes": (
            [
                int(kernel)
                for kernel in config.get(
                    "kernel_sizes",
                    [1, 3, 5],
                )
            ]
            if (
                block_type == "multiscale"
                or args.model
                == "fixed_period_inception"
            )
            else None
        ),

        "kernel_size": (
            int(
                config.get(
                    "kernel_size",
                    3,
                )
            )
            if (
                block_type == "single_kernel"
                or model_class_name
                == "CausalTCN"
            )
            else None
        ),

        "groups": (
            int(
                config.get(
                    "groups",
                    4,
                )
            )
            if block_type == "group"
            else None
        ),

        "num_channels": (
            [
                int(channel)
                for channel in config.get(
                    "num_channels",
                    [32, 64],
                )
            ]
            if model_class_name
            == "CausalTCN"
            else None
        ),

        "device": str(
            device
        ),

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

        "test_mse": float(
            test_mse
        ),

        "test_mae": float(
            test_mae
        ),

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
            float(
                peak_gpu_memory_mb
            )
            if peak_gpu_memory_mb
            is not None
            else None
        ),

        "peak_gpu_reserved_mb": (
            float(
                peak_gpu_reserved_mb
            )
            if peak_gpu_reserved_mb
            is not None
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

    print(
        "\nEsperimento completato"
    )

    print(
        f"Test MSE: "
        f"{test_mse:.6f}"
    )

    print(
        f"Test MAE: "
        f"{test_mae:.6f}"
    )

    print(
        "Tempo medio per epoca: "
        f"{average_epoch_time:.3f}s"
    )

    print(
        "Inferenza: "
        f"{inference_stats['inference_ms_per_sample']:.6f} "
        "ms/campione"
    )

    if peak_gpu_memory_mb is not None:
        print(
            "Peak GPU memory: "
            f"{peak_gpu_memory_mb:.2f} MB"
        )

    print(
        f"Metriche salvate in: "
        f"{metrics_path}"
    )

    return metrics


# ============================================================
# CONFRONTO TIMESNET TOP-K
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

    dataframe[
        comparison_columns
    ].to_csv(
        comparison_directory
        / "comparison_top_k.csv",
        index=False,
    )

    print(
        "\nConfronto top-k:\n"
    )

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
            "trainable_parameters",
            "Parametri allenabili",
            "parameters_top_k.png",
        ),
        (
            "average_epoch_time_seconds",
            "Secondi medi per epoca",
            "epoch_time_top_k.png",
        ),
        (
            "inference_ms_per_sample",
            "Inferenza ms/campione",
            "inference_top_k.png",
        ),
    ]

    for metric, ylabel, filename in plot_specs:
        plot_data = dataframe[
            ["top_k", metric]
        ].dropna()

        if plot_data.empty:
            continue

        plt.figure(
            figsize=(7, 5)
        )

        plt.plot(
            plot_data["top_k"],
            plot_data[metric],
            marker="o",
        )

        plt.xlabel(
            "top_k"
        )

        plt.ylabel(
            ylabel
        )

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
        "Confronto salvato in: "
        f"{comparison_directory}"
    )


# ============================================================
# CONFRONTO NUMERO TIMESBLOCK
# ============================================================

def compare_num_blocks_results(
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
        / (
            "fixed_period_inception_timesblocks_"
            f"{config_path.stem}"
        )
        / "comparison_num_times_blocks"
    )

    comparison_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        results
    ).sort_values(
        by="num_blocks"
    )

    comparison_columns = [
        "num_blocks",
        "fixed_period",
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

    dataframe[
        comparison_columns
    ].to_csv(
        comparison_directory
        / "comparison_num_times_blocks.csv",
        index=False,
    )

    print(
        "\nConfronto numero TimesBlock:\n"
    )

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
            "mse_num_times_blocks.png",
        ),
        (
            "test_mae",
            "Test MAE",
            "mae_num_times_blocks.png",
        ),
        (
            "best_validation_mse",
            "Validation MSE",
            "validation_num_times_blocks.png",
        ),
        (
            "trainable_parameters",
            "Parametri allenabili",
            "parameters_num_times_blocks.png",
        ),
        (
            "average_epoch_time_seconds",
            "Secondi medi per epoca",
            "epoch_time_num_times_blocks.png",
        ),
        (
            "total_training_time_seconds",
            "Tempo totale di training",
            "total_time_num_times_blocks.png",
        ),
        (
            "inference_ms_per_sample",
            "Inferenza ms/campione",
            "inference_num_times_blocks.png",
        ),
        (
            "peak_gpu_memory_mb",
            "Peak GPU memory MB",
            "gpu_memory_num_times_blocks.png",
        ),
    ]

    for metric, ylabel, filename in plot_specs:
        plot_data = dataframe[
            ["num_blocks", metric]
        ].dropna()

        if plot_data.empty:
            continue

        plt.figure(
            figsize=(7, 5)
        )

        plt.plot(
            plot_data["num_blocks"],
            plot_data[metric],
            marker="o",
        )

        plt.xlabel(
            "Numero di TimesBlock"
        )

        plt.ylabel(
            ylabel
        )

        plt.title(
            f"{ylabel} al variare "
            "del numero di TimesBlock"
        )

        plt.xticks(
            plot_data["num_blocks"]
        )

        plt.grid(True)
        plt.tight_layout()

        plt.savefig(
            comparison_directory
            / filename,
            dpi=200,
        )

        plt.close()

    best_row = dataframe.loc[
        dataframe[
            "best_validation_mse"
        ].idxmin()
    ]

    summary = {
        "selection_metric": (
            "best_validation_mse"
        ),
        "best_num_blocks": int(
            best_row["num_blocks"]
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

    with (
        comparison_directory
        / "best_num_times_blocks.json"
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
        "\nMiglior numero di TimesBlock "
        "sulla validation: "
        f"{summary['best_num_blocks']}"
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
        "Confronto salvato in: "
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

    validate_config(
        config
    )

    # --------------------------------------------------------
    # TIMESNET: SWEEP TOP-K
    # --------------------------------------------------------
    if args.model == "timesnet":
        if args.model_class is not None:
            raise ValueError(
                "--model-class non deve essere usato "
                "con --model timesnet."
            )

        top_k_values = sorted(
            set(
                args.top_k_values
            )
        )

        if not top_k_values:
            raise ValueError(
                "Devi specificare almeno "
                "un valore di top_k."
            )

        if any(
            value <= 0
            for value in top_k_values
        ):
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
                num_blocks=None,
            )

            all_results.append(
                metrics
            )

        compare_top_k_results(
            results=all_results,
            config_path=config_path,
        )

        return

    # --------------------------------------------------------
    # FIXED PERIOD INCEPTION:
    # SWEEP 1, 2, 3 TIMESBLOCK
    # --------------------------------------------------------
    if args.model == "fixed_period_inception":
        if args.model_class is not None:
            raise ValueError(
                "--model-class non deve essere usato "
                "con --model fixed_period_inception."
            )

        num_blocks_values = sorted(
            set(
                args.num_blocks_values
            )
        )

        if not num_blocks_values:
            raise ValueError(
                "Devi specificare almeno un valore "
                "di num_blocks."
            )

        if any(
            value <= 0
            for value in num_blocks_values
        ):
            raise ValueError(
                "Tutti i valori di num_blocks "
                "devono essere positivi."
            )

        print(
            "Avvio sweep FixedPeriodInception "
            "per numero di TimesBlock: "
            f"{num_blocks_values}"
        )

        all_results = []

        for num_blocks in num_blocks_values:
            metrics = run_experiment(
                args=args,
                config=config,
                config_path=config_path,
                top_k=None,
                num_blocks=num_blocks,
            )

            all_results.append(
                metrics
            )

        compare_num_blocks_results(
            results=all_results,
            config_path=config_path,
        )

        return

    # --------------------------------------------------------
    # TIMESNET LIGHT
    # --------------------------------------------------------
    if (
        args.model == "timesnet_light"
        and args.model_class is None
    ):
        raise ValueError(
            "Per --model timesnet_light devi "
            "specificare --model-class. "
            "Classi disponibili: "
            "LightTimesNetMultiScale, "
            "LightTimesNetDepthwise, "
            "LightTimesNetGroup, "
            "LightTimesNetSingleKernel."
        )

    # --------------------------------------------------------
    # ALTRI MODELLI: SINGOLO ESPERIMENTO
    # --------------------------------------------------------
    print(
        "Avvio singolo esperimento: "
        f"{args.model}"
    )

    metrics = run_experiment(
        args=args,
        config=config,
        config_path=config_path,
        top_k=None,
        num_blocks=None,
    )

    print(
        "\nRiepilogo:"
    )

    print(
        f"Modello: "
        f"{metrics['model']}"
    )

    print(
        f"Classe: "
        f"{metrics['model_class']}"
    )

    if metrics.get(
        "backbone_2d"
    ) is not None:
        print(
            "Backbone 2D: "
            f"{metrics['backbone_2d']}"
        )

    print(
        f"Test MSE: "
        f"{metrics['test_mse']:.6f}"
    )

    print(
        f"Test MAE: "
        f"{metrics['test_mae']:.6f}"
    )


if __name__ == "__main__":
    main()