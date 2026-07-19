from pathlib import Path

import yaml

from src.data import build_dataloader


CONFIG_PATHS = [
    "configs/etth1_24.yaml",
    "configs/etth1_48.yaml",
    "configs/etth1_96.yaml",
    "configs/ettm1_24.yaml",
    "configs/ettm1_48.yaml",
    "configs/ettm1_96.yaml",
]


def load_config(path: str) -> dict:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config non trovato: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config YAML non valido: {config_path}")

    return config


def test_config(config_path: str):
    config = load_config(config_path)

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

    x, y = next(iter(train_loader))

    expected_x = (
        config["batch_size"],
        config["seq_len"],
        config["num_features"],
    )

    expected_y = (
        config["batch_size"],
        config["pred_len"],
        config["num_features"],
    )

    assert tuple(x.shape) == expected_x, (
        f"Shape input errata: {tuple(x.shape)}, attesa: {expected_x}"
    )

    assert tuple(y.shape) == expected_y, (
        f"Shape target errata: {tuple(y.shape)}, attesa: {expected_y}"
    )

    print(
        f"{config['dataset_name']} | "
        f"pred_len={config['pred_len']} | "
        f"x={tuple(x.shape)} | "
        f"y={tuple(y.shape)}"
    )


if __name__ == "__main__":
    for config_path in CONFIG_PATHS:
        test_config(config_path)