from pathlib import Path

import yaml

from src.data import create_dataloaders


# File YAML da controllare
CONFIG_PATHS = [
    "configs/etth1.yaml",
    "configs/ettm1.yaml",
]


def load_config(path: str) -> dict:
    """Carica un file YAML e restituisce un dizionario Python."""

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config non trovato: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config YAML non valido: {config_path}")

    return config


def test_config(config_path: str):
    """
    Controlla tutti i pred_len presenti nel file YAML.

    Verifica che le shape dei batch siano corrette.
    """

    config = load_config(config_path)

    # Testiamo ogni orizzonte definito nel YAML
    for pred_len in config["time"]["pred_lens"]:
        train_loader, _, _ = create_dataloaders(
            config=config,
            pred_len=pred_len,
        )

        # Estraiamo il primo batch
        x, y = next(iter(train_loader))

        expected_x = (
            config["dataloader"]["batch_size"],
            config["time"]["seq_len"],
            config["model"]["num_features"],
        )

        expected_y = (
            config["dataloader"]["batch_size"],
            pred_len,
            config["model"]["num_features"],
        )

        # x deve avere shape: batch, seq_len, feature
        assert tuple(x.shape) == expected_x, (
            f"Shape input errata: {tuple(x.shape)}, "
            f"attesa: {expected_x}"
        )

        # y deve avere shape: batch, pred_len, feature
        assert tuple(y.shape) == expected_y, (
            f"Shape target errata: {tuple(y.shape)}, "
            f"attesa: {expected_y}"
        )

        print(
            f"{config['dataset']['name']} | "
            f"pred_len={pred_len} | "
            f"x={tuple(x.shape)} | "
            f"y={tuple(y.shape)}"
        )


if __name__ == "__main__":
    # Controllo di tutti i file YAML
    for config_path in CONFIG_PATHS:
        test_config(config_path)