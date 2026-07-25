"""
Funzioni condivise dai runner.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator
import json
import os
import subprocess
import sys
import time

import yaml

from src.runners.paths import (
    PROJECT_ROOT,
    get_configs_dir,
    get_experiments_root,
)


DEFAULT_DATASETS = (
    "etth1",
    "ettm1",
    "electricity",
)


def get_runner_output_dir(
    group_name: str,
) -> Path:
    """
    Priorità:

    1. EXPERIMENTS_DIR impostata dallo script .sh;
    2. EXPERIMENTS_ROOT/<group_name>.
    """

    environment_dir = os.environ.get(
        "EXPERIMENTS_DIR"
    )

    if environment_dir:
        output_dir = (
            Path(environment_dir)
            .expanduser()
            .resolve()
        )
    else:
        output_dir = (
            get_experiments_root()
            / group_name
        ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


def find_config_files(
    dataset_names: Iterable[str] = DEFAULT_DATASETS,
) -> list[Path]:
    """
    Cerca i file YAML per i dataset richiesti.
    """

    normalized_names = tuple(
        str(name).lower()
        for name in dataset_names
    )

    configs_dir = get_configs_dir()

    yaml_files = sorted(
        list(
            configs_dir.glob("*.yaml")
        )
        +
        list(
            configs_dir.glob("*.yml")
        )
    )

    selected = [
        yaml_file
        for yaml_file in yaml_files
        if any(
            dataset_name
            in yaml_file.stem.lower()
            for dataset_name
            in normalized_names
        )
    ]

    if not selected:
        raise FileNotFoundError(
            "Nessuna configurazione trovata per: "
            f"{', '.join(normalized_names)}\n"
            f"Cartella: {configs_dir}"
        )

    return selected


def load_yaml(
    yaml_path: Path,
) -> dict:
    """
    Carica un file YAML.
    """

    with yaml_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    if not isinstance(config, dict):
        raise ValueError(
            f"YAML non valido: {yaml_path}"
        )

    return config


def save_yaml(
    yaml_path: Path,
    config: dict,
) -> None:
    """
    Salva un file YAML.
    """

    with yaml_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )


@contextmanager
def temporary_yaml_values(
    yaml_path: Path,
    updates: dict,
) -> Iterator[dict]:
    """
    Modifica temporaneamente alcuni valori del file YAML.

    Il file originale viene sempre ripristinato,
    anche in caso di errore.
    """

    original_config = load_yaml(
        yaml_path
    )

    temporary_config = dict(
        original_config
    )

    temporary_config.update(
        updates
    )

    save_yaml(
        yaml_path,
        temporary_config,
    )

    try:
        yield temporary_config
    finally:
        save_yaml(
            yaml_path,
            original_config,
        )


def build_environment(
    output_dir: Path,
) -> dict[str, str]:
    """
    Ambiente passato a train_prova.py.
    """

    environment = os.environ.copy()

    environment["PYTHONPATH"] = str(
        PROJECT_ROOT
    )

    environment["EXPERIMENTS_DIR"] = str(
        output_dir
    )

    return environment


def build_train_command(
    config_name: str,
    model_name: str,
    model_class: str | None = None,
    top_k_values: list[int] | None = None,
    num_blocks_values: list[int] | None = None,
) -> list[str]:
    """
    Costruisce il comando per src.train_prova.
    """

    command = [
        sys.executable,
        "-m",
        "src.train_prova",
        "--config",
        config_name,
        "--model",
        model_name,
    ]

    if model_class is not None:
        command.extend(
            [
                "--model-class",
                model_class,
            ]
        )

    if top_k_values is not None:
        command.append(
            "--top-k-values"
        )

        command.extend(
            str(value)
            for value in top_k_values
        )

    if num_blocks_values is not None:
        command.append(
            "--num-blocks-values"
        )

        command.extend(
            str(value)
            for value in num_blocks_values
        )

    return command


def run_training_command(
    command: list[str],
    output_dir: Path,
) -> subprocess.CompletedProcess:
    """
    Esegue un comando di training.
    """

    print(
        "Comando:",
        " ".join(command),
    )

    start_time = time.perf_counter()

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=build_environment(
            output_dir
        ),
        text=True,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print(
        f"Tempo comando: {elapsed:.2f} secondi"
    )

    return result


def write_runner_summary(
    output_dir: Path,
    runner_name: str,
    completed: list,
    skipped: list,
    failed: list,
) -> Path:
    """
    Salva il riepilogo JSON del runner.
    """

    summary = {
        "runner": runner_name,
        "output_directory": str(
            output_dir
        ),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
    }

    summary_path = (
        output_dir
        / f"{runner_name}_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    return summary_path


def print_runner_header(
    title: str,
    output_dir: Path,
    yaml_files: list[Path],
) -> None:
    """
    Stampa intestazione del runner.
    """

    print("=" * 80)
    print(title)
    print("=" * 80)

    print(
        "Repository:",
        PROJECT_ROOT,
    )

    print(
        "Output:",
        output_dir,
    )

    print(
        "Configurazioni:",
        len(yaml_files),
    )

    print()

    for yaml_file in yaml_files:
        print(
            "-",
            yaml_file.name,
        )