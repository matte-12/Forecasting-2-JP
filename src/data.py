from __future__ import annotations

from torch.utils.data import DataLoader

from src.ett_dataset import TimeSeriesDataset
from src.runners.paths import resolve_csv_path


def build_dataloader(
    config: dict,
    flag: str,
):
    """
    Crea il Dataset e il DataLoader per train, validation o test.

    Il percorso CSV viene risolto tramite:

        src.runners.paths.resolve_csv_path

    In questo modo:

    - se csv_path è assoluto, viene usato direttamente;
    - se csv_path è relativo, viene risolto rispetto a DATA_ROOT;
    - se DATA_ROOT non è definita, viene usata la cartella
      <project_root>/data;
    - il prefisso storico "data/" può essere gestito da paths.py.

    Args:
        config:
            Dizionario caricato dal file YAML.

        flag:
            Uno tra "train", "val" e "test".

    Returns:
        tuple:
            dataset, dataloader
    """

    valid_flags = {
        "train",
        "val",
        "test",
    }

    if flag not in valid_flags:
        raise ValueError(
            f"flag non valido: {flag}. "
            f"Valori ammessi: {sorted(valid_flags)}"
        )

    required_keys = [
        "csv_path",
        "seq_len",
        "pred_len",
        "batch_size",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in config
    ]

    if missing_keys:
        raise KeyError(
            "Chiavi mancanti nella configurazione: "
            f"{missing_keys}"
        )

    # ========================================================
    # RISOLUZIONE DEL PERCORSO CSV
    # ========================================================

    csv_path = resolve_csv_path(
        config["csv_path"],
        must_exist=True,
    )

    print(
        f"[{flag}] CSV utilizzato: {csv_path}"
    )

    # ========================================================
    # DATASET
    # ========================================================

    dataset = TimeSeriesDataset(
        csv_path=csv_path,
        flag=flag,
        seq_len=int(
            config["seq_len"]
        ),
        pred_len=int(
            config["pred_len"]
        ),
    )

    # ========================================================
    # PARAMETRI DEL DATALOADER
    # ========================================================

    is_train = flag == "train"

    batch_size = int(
        config["batch_size"]
    )

    num_workers = int(
        config.get(
            "num_workers",
            0,
        )
    )

    shuffle = (
        bool(
            config.get(
                "shuffle_train",
                True,
            )
        )
        if is_train
        else False
    )

    drop_last = (
        bool(
            config.get(
                "drop_last_train",
                True,
            )
        )
        if is_train
        else False
    )

    pin_memory = bool(
        config.get(
            "pin_memory",
            False,
        )
    )

    persistent_workers = bool(
        config.get(
            "persistent_workers",
            False,
        )
    )

    # PyTorch non permette persistent_workers=True
    # quando num_workers=0.
    if num_workers == 0:
        persistent_workers = False

    # ========================================================
    # DATALOADER
    # ========================================================

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    print(
        f"[{flag}] Campioni: {len(dataset)} | "
        f"Batch: {len(dataloader)} | "
        f"batch_size: {batch_size} | "
        f"shuffle: {shuffle} | "
        f"drop_last: {drop_last}"
    )

    return dataset, dataloader