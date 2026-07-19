from torch.utils.data import DataLoader

from src.ett_dataset import TimeSeriesDataset


def create_dataloaders(config: dict, pred_len: int):
    """
    Crea i DataLoader di train, validation e test.

    Args:
        config: configurazione letta dal file YAML
        pred_len: singolo orizzonte scelto per il training
    """

    # Il pred_len scelto deve essere presente nella lista del YAML
    allowed_pred_lens = config["time"]["pred_lens"]

    if pred_len not in allowed_pred_lens:
        raise ValueError(
            f"pred_len={pred_len} non ammesso. "
            f"Valori disponibili: {allowed_pred_lens}"
        )

    # Parametri comuni ai tre split
    dataset_args = {
        "csv_path": config["dataset"]["csv_path"],
        "seq_len": config["time"]["seq_len"],
        "pred_len": pred_len,
    }

    train_dataset = TimeSeriesDataset(
        flag="train",
        **dataset_args,
    )

    val_dataset = TimeSeriesDataset(
        flag="val",
        **dataset_args,
    )

    test_dataset = TimeSeriesDataset(
        flag="test",
        **dataset_args,
    )

    loader_cfg = config["dataloader"]

    # Il training viene mescolato
    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=loader_cfg["shuffle_train"],
        num_workers=loader_cfg["num_workers"],
        drop_last=loader_cfg["drop_last_train"],
    )

    # Validation e test non devono essere mescolati
    val_loader = DataLoader(
        val_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        num_workers=loader_cfg["num_workers"],
        drop_last=False,
    )

    return train_loader, val_loader, test_loader