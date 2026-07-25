from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import optim

from src.data import build_dataloader


# ============================================================
# ARGOMENTI DA TERMINALE
# ============================================================

def parse_args() -> argparse.Namespace:
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
            "timesnet_mod_2d",
            "timesnet_original",
            "fixed_period_inception",
            "timesnet_light",
            "dlinear",
            "tcn",
        ],
        help="Famiglia di modello da allenare.",
    )

    parser.add_argument(
        "--model-class",
        type=str,
        default=None,
        help=(
            "Classe concreta per timesnet_light. "
            "Esempi: LightTimesNetMultiScale, "
            "LightTimesNetDepthwise, "
            "LightTimesNetGroup, "
            "LightTimesNetSingleKernel."
        ),
    )

    parser.add_argument(
        "--top-k-values",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Valori top_k da provare con TimesNet. "
            "Se omesso, usa top_k dello YAML oppure 3."
        ),
    )

    parser.add_argument(
        "--num-blocks-values",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Numero di TimesBlock da provare con "
            "timesnet_original o fixed_period_inception. "
            "Se omesso, usa il valore dello YAML oppure 1."
        ),
    )

    return parser.parse_args()


# ============================================================
# CONFIGURAZIONE YAML
# ============================================================

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_config_path(
    config_argument: str,
) -> Path:
    config_path = Path(
        config_argument
    ).expanduser()

    project_root = get_project_root()

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

    config_path = config_path.resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            "Configurazione non trovata:\n"
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
            "Configurazione YAML non valida:\n"
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
# SERIALIZZAZIONE JSON ROBUSTA
# ============================================================

def make_json_serializable(
    value: Any,
) -> Any:
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_serializable(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return make_json_serializable(
            value.tolist()
        )

    if isinstance(value, np.generic):
        return value.item()

    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()

        return value.detach().cpu().tolist()

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

    return value


def save_json(
    path: Path,
    content: Any,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_serializable(content),
            file,
            indent=4,
        )


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

    available_names = list_module_model_classes(
        module
    )

    if requested_class is not None:
        if requested_class not in available_names:
            raise AttributeError(
                f"La classe '{requested_class}' non è "
                f"presente in {module_path}. "
                f"Classi disponibili: {available_names}"
            )

        return getattr(
            module,
            requested_class,
        )

    for class_name in preferred_classes or []:
        if class_name in available_names:
            return getattr(
                module,
                class_name,
            )

    if len(available_names) == 1:
        return getattr(
            module,
            available_names[0],
        )

    raise ValueError(
        f"Non posso scegliere automaticamente una classe "
        f"da {module_path}. "
        f"Classi disponibili: {available_names}."
    )


# ============================================================
# COSTRUZIONE DINAMICA
# ============================================================

def instantiate_from_signature(
    model_class,
    config: dict,
    top_k: Optional[int] = None,
    num_blocks: Optional[int] = None,
) -> nn.Module:
    """
    Passa al modello soltanto gli argomenti presenti
    nella firma del suo __init__.

    Gestisce nomi alternativi come:
        num_blocks
        num_times_blocks
        e_layers

    e:
        num_features
        enc_in
        c_in
    """

    seq_len = int(
        config["seq_len"]
    )

    pred_len = int(
        config["pred_len"]
    )

    num_features = int(
        config["num_features"]
    )

    effective_top_k = int(
        top_k
        if top_k is not None
        else config.get(
            "top_k",
            3,
        )
    )

    effective_num_blocks = int(
        num_blocks
        if num_blocks is not None
        else config.get(
            "num_times_blocks",
            config.get(
                "num_blocks",
                config.get(
                    "e_layers",
                    1,
                ),
            ),
        )
    )

    patience_value = config.get(
        "patience",
        5,
    )

    candidate_kwargs = {
        # Lunghezza della sequenza
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
        "num_features": num_features,
        "enc_in": num_features,
        "c_in": num_features,
        "input_size": num_features,
        "input_dim": num_features,
        "n_features": num_features,
        "in_channels": num_features,

        # Output
        "c_out": num_features,
        "output_size": num_features,
        "output_dim": num_features,
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

        # TimesNet
        "top_k": effective_top_k,
        "k": effective_top_k,
        "num_blocks": effective_num_blocks,
        "num_times_blocks": effective_num_blocks,
        "e_layers": effective_num_blocks,

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

        # Inception
        "kernel_sizes": tuple(
            int(kernel)
            for kernel in config.get(
                "kernel_sizes",
                [1, 3, 5],
            )
        ),
        "num_kernels": int(
            config.get(
                "num_kernels",
                len(
                    config.get(
                        "kernel_sizes",
                        [1, 3, 5],
                    )
                ),
            )
        ),

        # TCN / convoluzioni
        "kernel_size": int(
            config.get(
                "kernel_size",
                3,
            )
        ),
        "num_channels": [
            int(channel)
            for channel in config.get(
                "num_channels",
                [32, 64],
            )
        ],
        "groups": int(
            config.get(
                "groups",
                4,
            )
        ),

        # Regolarizzazione
        "dropout": float(
            config.get(
                "dropout",
                0.1,
            )
        ),

        # Eventuali opzioni legacy
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

        # Informativo, usato soltanto se accettato
        "patience": (
            5
            if patience_value is None
            else int(patience_value)
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

    accepted_kwargs = {
        name: value
        for name, value in candidate_kwargs.items()
        if (
            accepts_kwargs
            or name in signature.parameters
        )
    }

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
            f"{model_class.__module__}."
            f"{model_class.__name__}.\n"
            f"Argomenti obbligatori non riconosciuti: "
            f"{missing_required}\n"
            f"Firma: {signature}"
        )

    print(
        "Classe selezionata: "
        f"{model_class.__module__}."
        f"{model_class.__name__}"
    )

    print(
        "Argomenti passati:",
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
    # VECCHIA TIMESNET DI models_2d.py
    # --------------------------------------------------------
    if model_name == "timesnet_mod_2d":
        model_class = find_model_class(
            module_path="src.models_2d",
            preferred_classes=[
                "TimesNet",
            ],
        )

        return instantiate_from_signature(
            model_class=model_class,
            config=config,
            top_k=top_k,
            num_blocks=num_blocks,
        )

    # --------------------------------------------------------
    # TIMESNET ORIGINAL
    # File: src/timesnet_original.py
    # --------------------------------------------------------
    if model_name == "timesnet_original":
        if requested_class is not None:
            raise ValueError(
                "--model-class non deve essere usato "
                "con --model timesnet_original."
            )

        model_class = find_model_class(
            module_path="src.timesnet_original",
            preferred_classes=[
                "TimesNetOriginal",
                "TimesNet",
                "Model",
            ],
        )

        return instantiate_from_signature(
            model_class=model_class,
            config=config,
            top_k=top_k,
            num_blocks=num_blocks,
        )

    # --------------------------------------------------------
    # FIXED PERIOD INCEPTION
    # File: src/fixed_period_inception.py
    # --------------------------------------------------------
    if model_name == "fixed_period_inception":
        model_class = find_model_class(
            module_path="src.fixed_period_inception",
            preferred_classes=[
                "FixedPeriodInception2D",
                "FixedPeriodInception",
            ],
        )

        return instantiate_from_signature(
            model_class=model_class,
            config=config,
            top_k=None,
            num_blocks=num_blocks,
        )

    # --------------------------------------------------------
    # TIMESNET LIGHT
    # File: src/models_light.py
    # --------------------------------------------------------
    if model_name == "timesnet_light":
        allowed_light_classes = {
            "LightTimesNet",
            "LightTimesNetMultiScale",
            "LightTimesNetDepthwise",
            "LightTimesNetGroup",
            "LightTimesNetSingleKernel",
        }

        if requested_class is None:
            raise ValueError(
                "Per --model timesnet_light devi specificare "
                "--model-class."
            )

        if requested_class not in allowed_light_classes:
            raise ValueError(
                "Classe TimesNet Light non supportata: "
                f"{requested_class}. "
                f"Classi disponibili: "
                f"{sorted(allowed_light_classes)}"
            )

        model_class = find_model_class(
            module_path="src.models_light",
            requested_class=requested_class,
        )

        return instantiate_from_signature(
            model_class=model_class,
            config=config,
        )

    # --------------------------------------------------------
    # DLINEAR
    # --------------------------------------------------------
    if model_name == "dlinear":
        model_class = find_model_class(
            module_path="src.models_1d",
            requested_class="DLinear",
        )

        return instantiate_from_signature(
            model_class=model_class,
            config=config,
        )

    # --------------------------------------------------------
    # CAUSAL TCN
    # --------------------------------------------------------
    if model_name == "tcn":
        model_class = find_model_class(
            module_path="src.models_1d",
            requested_class="CausalTCN",
        )

        return instantiate_from_signature(
            model_class=model_class,
            config=config,
        )

    raise ValueError(
        f"Modello non riconosciuto: {model_name}"
    )


# ============================================================
# INFORMAZIONI DEL MODELLO
# ============================================================

def get_model_fixed_period(
    model: nn.Module,
) -> Optional[int]:
    for attribute in (
        "period",
        "fixed_period",
    ):
        if hasattr(model, attribute):
            value = getattr(
                model,
                attribute,
            )

            if value is not None:
                return int(value)

    if hasattr(model, "times_block"):
        for attribute in (
            "period",
            "fixed_period",
        ):
            if hasattr(
                model.times_block,
                attribute,
            ):
                return int(
                    getattr(
                        model.times_block,
                        attribute,
                    )
                )

    return None


def get_model_num_blocks(
    model: nn.Module,
) -> Optional[int]:
    for attribute in (
        "num_times_blocks",
        "num_blocks",
        "e_layers",
    ):
        if hasattr(model, attribute):
            value = getattr(
                model,
                attribute,
            )

            if value is not None:
                return int(value)

    for attribute in (
        "times_blocks",
        "blocks",
        "model",
    ):
        if hasattr(model, attribute):
            value = getattr(
                model,
                attribute,
            )

            if isinstance(
                value,
                nn.ModuleList,
            ):
                return len(value)

    return None


def get_model_top_k(
    model: nn.Module,
) -> Optional[int]:
    for attribute in (
        "top_k",
        "k",
    ):
        if hasattr(model, attribute):
            value = getattr(
                model,
                attribute,
            )

            if value is not None:
                return int(value)

    return None


def get_model_block_type(
    model: nn.Module,
) -> Optional[str]:
    for attribute in (
        "block_type",
        "backbone_2d",
        "block_name",
    ):
        if hasattr(model, attribute):
            value = getattr(
                model,
                attribute,
            )

            if value is not None:
                return str(value)

    if hasattr(model, "times_block"):
        for attribute in (
            "block_type",
            "backbone_2d",
            "block_name",
        ):
            if hasattr(
                model.times_block,
                attribute,
            ):
                return str(
                    getattr(
                        model.times_block,
                        attribute,
                    )
                )

    return None


def sanitize_name(
    value: str,
) -> str:
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )


# ============================================================
# DIRECTORY ESPERIMENTI
# ============================================================

def get_experiments_root() -> Path:
    project_root = get_project_root()

    experiments_root = Path(
        os.environ.get(
            "EXPERIMENTS_DIR",
            os.environ.get(
                "EXPERIMENTS_ROOT",
                project_root / "experiments",
            ),
        )
    ).expanduser().resolve()

    experiments_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    return experiments_root


def create_experiment_directory(
    model_name: str,
    config_path: Path,
    model: nn.Module,
    config: dict,
    top_k: Optional[int] = None,
    requested_class: Optional[str] = None,
    num_blocks: Optional[int] = None,
) -> Path:
    """
    Crea una cartella piatta e auto-descrittiva per ogni run.

    Esempi:

    fixed_period_inception_etth1_seq_96_pred_24_period_24_tb_1

    timesnet_original_etth1_seq_96_pred_24_top_3_tb_2

    timesnet_light_multiscale_etth1_seq_96_pred_24_period_24_tb_1

    dlinear_etth1_seq_96_pred_24
    """

    experiments_root = get_experiments_root()

    dataset_name = sanitize_name(
        str(
            config.get(
                "dataset_name",
                config_path.stem,
            )
        ).lower()
    )

    seq_len = int(
        config["seq_len"]
    )

    pred_len = int(
        config["pred_len"]
    )

    effective_num_blocks = get_model_num_blocks(
        model
    )

    if effective_num_blocks is None:
        effective_num_blocks = (
            int(num_blocks)
            if num_blocks is not None
            else None
        )

    effective_period = get_model_fixed_period(
        model
    )

    effective_top_k = get_model_top_k(
        model
    )

    if effective_top_k is None and top_k is not None:
        effective_top_k = int(top_k)

    name_parts = [
        sanitize_name(
            model_name.lower()
        ),
    ]

    # Per timesnet_light inserisce anche il nome del backbone.
    if model_name == "timesnet_light":
        class_name = (
            requested_class
            if requested_class is not None
            else model.__class__.__name__
        )

        normalized_class_name = (
            str(class_name)
            .replace("LightTimesNet", "")
            .strip("_")
            .lower()
        )

        if not normalized_class_name:
            normalized_class_name = "base"

        name_parts.append(
            sanitize_name(
                normalized_class_name
            )
        )

    name_parts.extend(
        [
            dataset_name,
            "seq",
            str(seq_len),
            "pred",
            str(pred_len),
        ]
    )

    if model_name == "timesnet_original":
        if effective_top_k is None:
            raise ValueError(
                "Impossibile determinare top_k "
                "per TimesNetOriginal."
            )

        name_parts.extend(
            [
                "top",
                str(effective_top_k),
            ]
        )

        if effective_num_blocks is not None:
            name_parts.extend(
                [
                    "tb",
                    str(
                        effective_num_blocks
                    ),
                ]
            )

    elif model_name == "fixed_period_inception":
        if effective_period is None:
            raise ValueError(
                "Impossibile determinare fixed_period."
            )

        name_parts.extend(
            [
                "period",
                str(effective_period),
            ]
        )

        if effective_num_blocks is not None:
            name_parts.extend(
                [
                    "tb",
                    str(
                        effective_num_blocks
                    ),
                ]
            )

    elif model_name == "timesnet_light":
        if effective_period is not None:
            name_parts.extend(
                [
                    "period",
                    str(effective_period),
                ]
            )

        if effective_num_blocks is not None:
            name_parts.extend(
                [
                    "tb",
                    str(
                        effective_num_blocks
                    ),
                ]
            )

    elif model_name == "timesnet_mod_2d":
        if effective_top_k is not None:
            name_parts.extend(
                [
                    "top",
                    str(effective_top_k),
                ]
            )

        if effective_num_blocks is not None:
            name_parts.extend(
                [
                    "tb",
                    str(
                        effective_num_blocks
                    ),
                ]
            )

    experiment_name = "_".join(
        name_parts
    )

    experiment_directory = (
        experiments_root
        / experiment_name
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
                        float(gradient_clip),
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

    return total_loss / total_samples


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

            error = predictions - batch_y

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

            element_count += batch_y.numel()

    if element_count == 0:
        raise RuntimeError(
            "Il test DataLoader non contiene elementi."
        )

    return (
        squared_error_sum / element_count,
        absolute_error_sum / element_count,
    )


# ============================================================
# TEMPO DI INFERENZA
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
            total_samples += batch_x.size(0)

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
            total_samples
            / elapsed_seconds
        ),
    }


# ============================================================
# SINGO ESPERIMENTO
# ============================================================

def run_experiment(
    args: argparse.Namespace,
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
            config=config,
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

    effective_period = get_model_fixed_period(
        model
    )

    effective_num_blocks = (
        get_model_num_blocks(
            model
        )
    )

    if (
        effective_num_blocks is None
        and num_blocks is not None
    ):
        effective_num_blocks = int(
            num_blocks
        )

    effective_top_k = get_model_top_k(
        model
    )

    if (
        effective_top_k is None
        and top_k is not None
    ):
        effective_top_k = int(
            top_k
        )

    block_type = get_model_block_type(
        model
    )

    kernel_sizes = [
        int(kernel)
        for kernel in config.get(
            "kernel_sizes",
            [1, 3, 5],
        )
    ]

    config_used = dict(
        config
    )

    config_used.update(
        {
            "selected_model": args.model,
            "selected_model_class": (
                model.__class__.__name__
            ),
            "top_k": effective_top_k,
            "num_blocks": effective_num_blocks,
            "num_times_blocks": (
                effective_num_blocks
            ),
            "fixed_period": effective_period,
            "backbone_2d": block_type,
            "kernel_sizes": kernel_sizes,
        }
    )

    with (
        experiment_directory
        / "config_used.yaml"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            make_json_serializable(
                config_used
            ),
            file,
            sort_keys=False,
        )

    print("\n" + "=" * 72)
    print(f"Modello: {args.model}")
    print(
        "Classe: "
        f"{model.__class__.__name__}"
    )
    print(
        "Dataset: "
        f"{config['dataset_name']}"
    )
    print(f"seq_len: {config['seq_len']}")
    print(f"pred_len: {config['pred_len']}")

    if effective_top_k is not None:
        print(f"top_k: {effective_top_k}")

    if effective_num_blocks is not None:
        print(
            "Numero TimesBlock: "
            f"{effective_num_blocks}"
        )

    if effective_period is not None:
        print(
            "fixed_period: "
            f"{effective_period}"
        )

    print(f"kernel_sizes: {kernel_sizes}")
    print(f"Device: {device}")
    print(
        "Parametri allenabili: "
        f"{parameter_count:,}"
    )
    print(
        f"Output: {experiment_directory}"
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
            "Shape non compatibili prima del training: "
            f"prediction={sample_prediction.shape}, "
            f"target={sample_y.shape}."
        )

    patience_value = config.get(
        "patience",
        5,
    )

    patience = (
        5
        if patience_value is None
        else int(patience_value)
    )

    best_validation_loss = float(
        "inf"
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

        label_parts = []

        if effective_top_k is not None:
            label_parts.append(
                f"top_k={effective_top_k}"
            )

        if effective_num_blocks is not None:
            label_parts.append(
                "num_times_blocks="
                f"{effective_num_blocks}"
            )

        experiment_label = (
            " | ".join(label_parts)
            if label_parts
            else model.__class__.__name__
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
            "Il checkpoint migliore non è stato creato:\n"
            f"{checkpoint_path}"
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
        "seq_len": int(
            config["seq_len"]
        ),
        "pred_len": int(
            config["pred_len"]
        ),
        "batch_size": int(
            config["batch_size"]
        ),

        "top_k": effective_top_k,
        "num_blocks": effective_num_blocks,
        "num_times_blocks": (
            effective_num_blocks
        ),
        "fixed_period": effective_period,

        "backbone_2d": block_type,
        "kernel_sizes": kernel_sizes,

        "use_fft": (
            True
            if args.model
            == "timesnet_original"
            else config.get(
                "use_fft"
            )
        ),

        "use_inception": (
            True
            if args.model
            == "timesnet_original"
            else config.get(
                "use_inception"
            )
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
            if peak_gpu_memory_mb
            is not None
            else None
        ),
        "peak_gpu_reserved_mb": (
            float(peak_gpu_reserved_mb)
            if peak_gpu_reserved_mb
            is not None
            else None
        ),

        "experiment_directory": str(
            experiment_directory
        ),
    }

    save_json(
        experiment_directory
        / "metrics.json",
        metrics,
    )

    save_json(
        experiment_directory
        / "history.json",
        history,
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
        "Metriche salvate in: "
        f"{experiment_directory / 'metrics.json'}"
    )

    return metrics


# ============================================================
# CONFRONTI INTERNI
# ============================================================

def save_sweep_comparison(
    results: list[dict],
    output_directory: Path,
    filename: str,
    sort_columns: list[str],
) -> None:
    import pandas as pd

    if not results:
        return

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.DataFrame(
        results
    )

    existing_sort_columns = [
        column
        for column in sort_columns
        if column in dataframe.columns
    ]

    if existing_sort_columns:
        dataframe = dataframe.sort_values(
            existing_sort_columns
        )

    dataframe.to_csv(
        output_directory / filename,
        index=False,
    )

    print(
        "\nConfronto salvato in:"
    )

    print(
        output_directory / filename
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    config, config_path = load_config(
        args.config
    )

    validate_config(
        config
    )

    # --------------------------------------------------------
    # TIMESNET ORIGINAL:
    # TOP-K × NUMERO DI TIMESBLOCK
    # --------------------------------------------------------
    if args.model == "timesnet_original":
        if args.model_class is not None:
            raise ValueError(
                "--model-class non deve essere usato "
                "con timesnet_original."
            )

        top_k_values = (
            sorted(
                set(args.top_k_values)
            )
            if args.top_k_values
            is not None
            else [
                int(
                    config.get(
                        "top_k",
                        3,
                    )
                )
            ]
        )

        num_blocks_values = (
            sorted(
                set(
                    args.num_blocks_values
                )
            )
            if args.num_blocks_values
            is not None
            else [
                int(
                    config.get(
                        "num_times_blocks",
                        config.get(
                            "num_blocks",
                            1,
                        ),
                    )
                )
            ]
        )

        if any(
            value <= 0
            for value in top_k_values
        ):
            raise ValueError(
                "I valori top_k devono essere positivi."
            )

        if any(
            value <= 0
            for value in num_blocks_values
        ):
            raise ValueError(
                "I valori num_blocks devono essere positivi."
            )

        print(
            "Avvio sweep TimesNetOriginal:"
        )

        print(
            f"top_k={top_k_values}"
        )

        print(
            "num_times_blocks="
            f"{num_blocks_values}"
        )

        all_results = []

        for top_k in top_k_values:
            for num_blocks in (
                num_blocks_values
            ):
                metrics = run_experiment(
                    args=args,
                    config=config,
                    config_path=config_path,
                    top_k=top_k,
                    num_blocks=num_blocks,
                )

                all_results.append(
                    metrics
                )

        comparison_directory = (
            get_experiments_root()
            / (
                "timesnet_original_"
                f"{config_path.stem}"
            )
            / "comparison"
        )

        save_sweep_comparison(
            results=all_results,
            output_directory=(
                comparison_directory
            ),
            filename=(
                "comparison_top_k_"
                "num_times_blocks.csv"
            ),
            sort_columns=[
                "top_k",
                "num_times_blocks",
            ],
        )

        return

    # --------------------------------------------------------
    # VECCHIA TIMESNET models_2d:
    # SWEEP TOP-K
    # --------------------------------------------------------
    if args.model == "timesnet_mod_2d":
        top_k_values = (
            sorted(
                set(args.top_k_values)
            )
            if args.top_k_values
            is not None
            else [
                int(
                    config.get(
                        "top_k",
                        3,
                    )
                )
            ]
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

        comparison_directory = (
            get_experiments_root()
            / (
                "timesnet_mod_2d_"
                f"{config_path.stem}"
            )
            / "comparison"
        )

        save_sweep_comparison(
            results=all_results,
            output_directory=(
                comparison_directory
            ),
            filename="comparison_top_k.csv",
            sort_columns=["top_k"],
        )

        return

    # --------------------------------------------------------
    # FIXED PERIOD:
    # SWEEP NUMERO TIMESBLOCK
    # --------------------------------------------------------
    if args.model == "fixed_period_inception":
        num_blocks_values = (
            sorted(
                set(
                    args.num_blocks_values
                )
            )
            if args.num_blocks_values
            is not None
            else [
                int(
                    config.get(
                        "num_blocks",
                        config.get(
                            "num_times_blocks",
                            1,
                        ),
                    )
                )
            ]
        )

        if any(
            value <= 0
            for value in num_blocks_values
        ):
            raise ValueError(
                "I valori num_blocks devono essere positivi."
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

        comparison_directory = (
            get_experiments_root()
            / (
                "fixed_period_inception_"
                "timesblocks_"
                f"{config_path.stem}"
            )
            / "comparison_num_times_blocks"
        )

        save_sweep_comparison(
            results=all_results,
            output_directory=(
                comparison_directory
            ),
            filename=(
                "comparison_num_times_blocks.csv"
            ),
            sort_columns=[
                "num_times_blocks"
            ],
        )

        return

    # --------------------------------------------------------
    # ALTRI MODELLI:
    # SINGO ESPERIMENTO
    # --------------------------------------------------------

    metrics = run_experiment(
        args=args,
        config=config,
        config_path=config_path,
        top_k=None,
        num_blocks=None,
    )

    print("\nRiepilogo")
    print(
        f"Modello: {metrics['model']}"
    )
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