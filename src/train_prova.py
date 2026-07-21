import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import optim

from src.data import build_dataloader


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
            "top_k_inception",
        ],
        help="Architettura 2D da allenare.",
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


def build_model(
    model_name: str,
    config: dict,
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

        return TimesNet(
            seq_len=config["seq_len"],
            pred_len=config["pred_len"],
            enc_in=config["num_features"],
            d_model=config.get(
                "d_model",
                32,
            ),
            top_k=config.get(
                "top_k",
                3,
            ),
            use_fft=config.get(
                "use_fft",
                True,
            ),
            fixed_period=config.get(
                "fixed_period",
                24,
            ),
            use_inception=config.get(
                "use_inception",
                True,
            ),
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

    raise ValueError(
        f"Modello non riconosciuto: {model_name}"
    )


def create_experiment_directory(
    model_name: str,
    config_name: str,
    config: dict,
) -> Path:
    project_root = (
        Path(__file__).resolve().parent.parent
    )

    experiment_name = (
        f"{model_name}_"
        f"{config['dataset_name']}_"
        f"seq{config['seq_len']}_"
        f"pred{config['pred_len']}_"
        f"{config_name}"
    )

    experiment_directory = (
        project_root
        / "experiments"
        / experiment_name
    )

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


def main():
    args = parse_args()

    config, config_path = load_config(
        args.config
    )

    validate_config(config)
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

    experiment_directory = (
        create_experiment_directory(
            model_name=args.model,
            config_name=config_path.stem,
            config=config,
        )
    )

    checkpoint_path = (
        experiment_directory
        / "best_model.pth"
    )

    shutil.copy2(
        config_path,
        experiment_directory
        / "config_used.yaml",
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("=" * 60)
    print(f"Modello: {args.model}")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Device: {device}")
    print(f"seq_len: {config['seq_len']}")
    print(f"pred_len: {config['pred_len']}")
    print(f"Parametri: {parameter_count:,}")

    if args.model == "fixed_period_inception":
        print(
            "Periodo fixed usato dal modello: "
            f"{config['fixed_period']}"
        )

    print("=" * 60)

    # Controllo iniziale delle shape.
    sample_x, sample_y = next(
        iter(train_loader)
    )

    with torch.no_grad():
        sample_prediction = model(
            sample_x.to(device)
        ).cpu()

    print(
        f"Input:      {tuple(sample_x.shape)}"
    )
    print(
        f"Target:     {tuple(sample_y.shape)}"
    )
    print(
        "Prediction: "
        f"{tuple(sample_prediction.shape)}"
    )

    if sample_prediction.shape != sample_y.shape:
        raise RuntimeError(
            "Il modello non produce la stessa "
            "shape del target."
        )

    best_validation_loss = float("inf")
    patience = config.get("patience", 5)
    epochs_without_improvement = 0

    history = {
        "train_mse": [],
        "val_mse": [],
    }

    for epoch in range(
        1,
        config["epochs"] + 1,
    ):
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

        history["train_mse"].append(
            train_mse
        )

        history["val_mse"].append(
            val_mse
        )

        print(
            f"Epoch {epoch:03d}/"
            f"{config['epochs']} | "
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

    metrics = {
        "model": args.model,
        "config": config_path.stem,
        "dataset": config["dataset_name"],
        "seq_len": config["seq_len"],
        "pred_len": config["pred_len"],
        "fixed_period": (
            model.period
            if args.model
            == "fixed_period_inception"
            else None
        ),
        "trainable_parameters": parameter_count,
        "best_validation_mse": (
            best_validation_loss
        ),
        "test_mse": test_mse,
        "test_mae": test_mae,
    }

    with (
        experiment_directory
        / "metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    with (
        experiment_directory
        / "history.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=4,
        )

    print("=" * 60)
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Risultati: {experiment_directory}")
    print("=" * 60)


if __name__ == "__main__":
    main()