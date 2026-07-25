"""
Gestione centralizzata dei percorsi.

Il progetto può funzionare:

- su Google Colab con dati e risultati su Drive;
- in locale usando le cartelle data/ ed experiments/;
- su server con percorsi personalizzati.

Variabili d'ambiente:

DATA_ROOT
    Cartella principale dei dataset.

EXPERIMENTS_ROOT
    Cartella principale di tutti gli esperimenti.

EXPERIMENTS_DIR
    Cartella specifica assegnata al runner corrente.
"""

from __future__ import annotations

import os
from pathlib import Path


# Il file si trova in:
# <project_root>/src/runners/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

SRC_DIR = PROJECT_ROOT / "src"
CONFIGS_DIR = PROJECT_ROOT / "configs"

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"


def resolve_path(
    value: str | Path,
) -> Path:
    """
    Converte un percorso in Path assoluto.
    """

    return (
        Path(value)
        .expanduser()
        .resolve()
    )


def ensure_directory(
    directory: Path,
) -> Path:
    """
    Crea una cartella se non esiste.
    """

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory


def get_data_root(
    create: bool = False,
) -> Path:
    """
    Priorità:

    1. DATA_ROOT;
    2. <project_root>/data.
    """

    value = os.environ.get(
        "DATA_ROOT"
    )

    if value:
        data_root = resolve_path(
            value
        )
    else:
        data_root = (
            DEFAULT_DATA_ROOT.resolve()
        )

    if create:
        ensure_directory(
            data_root
        )

    return data_root


def get_experiments_root(
    create: bool = True,
) -> Path:
    """
    Priorità:

    1. EXPERIMENTS_ROOT;
    2. <project_root>/experiments.
    """

    value = os.environ.get(
        "EXPERIMENTS_ROOT"
    )

    if value:
        experiments_root = resolve_path(
            value
        )
    else:
        experiments_root = (
            DEFAULT_EXPERIMENTS_ROOT.resolve()
        )

    if create:
        ensure_directory(
            experiments_root
        )

    return experiments_root


def get_experiments_dir(
    create: bool = True,
) -> Path:
    """
    Restituisce la cartella specifica del runner.

    Priorità:

    1. EXPERIMENTS_DIR;
    2. EXPERIMENTS_ROOT;
    3. <project_root>/experiments.
    """

    value = os.environ.get(
        "EXPERIMENTS_DIR"
    )

    if value:
        experiments_dir = resolve_path(
            value
        )
    else:
        experiments_dir = get_experiments_root(
            create=create
        )

    if create:
        ensure_directory(
            experiments_dir
        )

    return experiments_dir


def get_group_directory(
    group_name: str,
    create: bool = True,
) -> Path:
    """
    Restituisce:

    EXPERIMENTS_ROOT/<group_name>
    """

    group_name = str(
        group_name
    ).strip()

    if not group_name:
        raise ValueError(
            "group_name non può essere vuoto."
        )

    directory = (
        get_experiments_root(
            create=create
        )
        / group_name
    )

    if create:
        ensure_directory(
            directory
        )

    return directory


def get_configs_dir() -> Path:
    """
    Restituisce la cartella configs/.
    """

    directory = CONFIGS_DIR.resolve()

    if not directory.exists():
        raise FileNotFoundError(
            "Cartella configs non trovata:\n"
            f"{directory}"
        )

    return directory


def resolve_csv_path(
    csv_path: str | Path,
    must_exist: bool = True,
) -> Path:
    """
    Risolve il percorso di un dataset CSV.

    Casi supportati:

    1. Percorso assoluto:
       /content/drive/MyDrive/.../ETTh1.csv

    2. Percorso relativo:
       ETT-small/ETTh1.csv

    3. Percorso relativo storico:
       data/ETT-small/ETTh1.csv

    Se DATA_ROOT è definita, viene usata come radice.
    Altrimenti viene usata <project_root>/data.
    """

    csv_path = Path(
        csv_path
    ).expanduser()

    # Un percorso assoluto viene mantenuto.
    if csv_path.is_absolute():
        resolved_path = csv_path.resolve()

    else:
        path_parts = csv_path.parts

        # Evita di ottenere:
        #
        # DATA_ROOT/data/ETT-small/ETTh1.csv
        #
        # quando lo YAML contiene già il prefisso "data/".
        if (
            path_parts
            and path_parts[0].lower() == "data"
        ):
            csv_path = Path(
                *path_parts[1:]
            )

        resolved_path = (
            get_data_root()
            / csv_path
        ).resolve()

    if (
        must_exist
        and not resolved_path.exists()
    ):
        raise FileNotFoundError(
            "Dataset CSV non trovato.\n\n"
            f"csv_path originale: {csv_path}\n"
            f"DATA_ROOT: {get_data_root()}\n"
            f"Percorso risolto: {resolved_path}\n\n"
            "Controlla la struttura della cartella "
            "dataset_forecast e il valore di csv_path "
            "nel file YAML."
        )

    return resolved_path

def get_comparison_directory(
    comparison_name: str | None = None,
    create: bool = True,
) -> Path:
    """
    Restituisce:

    experiments/comparison/

    oppure:

    experiments/comparison/<comparison_name>/
    """

    directory = (
        get_experiments_root(
            create=create
        )
        / "comparison"
    )

    if comparison_name:
        directory = (
            directory
            / comparison_name.strip()
        )

    if create:
        ensure_directory(
            directory
        )

    return directory


def describe_paths() -> dict[str, str]:
    """
    Restituisce i percorsi correntemente utilizzati.
    """

    return {
        "project_root": str(
            PROJECT_ROOT
        ),
        "configs_dir": str(
            CONFIGS_DIR.resolve()
        ),
        "data_root": str(
            get_data_root()
        ),
        "experiments_root": str(
            get_experiments_root()
        ),
        "experiments_dir": str(
            get_experiments_dir()
        ),
    }


def print_paths() -> None:
    """
    Stampa i percorsi correnti.
    """

    print("=" * 80)
    print("PERCORSI DEL PROGETTO")
    print("=" * 80)

    for name, value in describe_paths().items():
        print(
            f"{name}: {value}"
        )