import argparse
import importlib

from data import build_dataloader


def load_config(config_name):
    module = importlib.import_module(f"configs.{config_name}")

    if not hasattr(module, "CONFIG"):
        raise AttributeError(
            f"Il file configs/{config_name}.py "
            "non contiene la variabile CONFIG."
        )

    return module.CONFIG


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Nome del file di configurazione senza .py, "
            "ad esempio ettm1_24."
        ),
    )

    args = parser.parse_args()
    config = load_config(args.config)

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

    print("Configurazione caricata correttamente")
    print(f"Dataset: {config['dataset_name']}")
    print(f"CSV: {train_dataset.csv_path}")
    print(f"Feature: {train_dataset.feature_names}")
    print(f"Numero feature: {train_dataset.num_features}")
    print(f"seq_len: {train_dataset.seq_len}")
    print(f"pred_len: {train_dataset.pred_len}")
    print()

    print(f"Finestre train: {len(train_dataset)}")
    print(f"Finestre validation: {len(val_dataset)}")
    print(f"Finestre test: {len(test_dataset)}")
    print()

    print(f"Input [B, T, C]: {x.shape}")
    print(f"Target [B, H, C]: {y.shape}")


if __name__ == "__main__":
    main()