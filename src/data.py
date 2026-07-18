from pathlib import Path

from torch.utils.data import DataLoader

from src.ett_dataset import TimeSeriesDataset

def resolve_csv_path(csv_path):
    """
    Risolve un percorso CSV assoluto o relativo.

    - Se è assoluto, viene usato direttamente.
    - Se è relativo, viene interpretato rispetto alla root del progetto.
    """
    csv_path = Path(csv_path).expanduser()

    if csv_path.is_absolute():
        return csv_path

    project_root = Path(__file__).resolve().parent
    return project_root / csv_path


def build_dataloader(config, flag):
    """
    Crea Dataset e DataLoader usando i parametri della configurazione.
    """
    csv_path = resolve_csv_path(config["csv_path"])

    dataset = TimeSeriesDataset(
        csv_path=csv_path,
        flag=flag,
        seq_len=config["seq_len"],
        pred_len=config["pred_len"],
    )

    is_train = flag == "train"

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=(
            config.get("shuffle_train", True)
            if is_train
            else False
        ),
        num_workers=config.get("num_workers", 0),
        drop_last=(
            config.get("drop_last_train", True)
            if is_train
            else False
        ),
        pin_memory=config.get("pin_memory", False),
    )

    return dataset, dataloader