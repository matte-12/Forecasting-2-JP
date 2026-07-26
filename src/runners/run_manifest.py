"""
Runner unico basato su experiments_manifest.json.

Funzioni principali:

- seleziona i gruppi abilitati;
- seleziona i dataset;
- trova gli YAML esistenti;
- modifica temporaneamente i parametri dello YAML;
- esegue train_prova.py;
- salva tutti i risultati direttamente in EXPERIMENTS_ROOT;
- salva un riepilogo JSON dei comandi eseguiti.

Esempi:

    python -m src.runners.run_manifest

    python -m src.runners.run_manifest --group cicli

    python -m src.runners.run_manifest --group fixed_period

    python -m src.runners.run_manifest --dataset etth1

    python -m src.runners.run_manifest \
        --group times_block \
        --dataset electricity

    python -m src.runners.run_manifest --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import yaml

from src.runners.paths import (
    PROJECT_ROOT,
    get_configs_dir,
    get_experiments_root,
)


MANIFEST_PATH = (
    Path(__file__).resolve().parent
    / "experiments_manifest.json"
)

SUPPORTED_DATASETS = {
    "etth1",
    "ettm1",
    "electricity",
}


# ============================================================
# ARGOMENTI CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Esegue gli esperimenti definiti nel manifest JSON."
        )
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=str(MANIFEST_PATH),
        help="Percorso del manifest JSON.",
    )

    parser.add_argument(
        "--group",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Esegue soltanto i gruppi indicati. "
            "Esempio: --group cicli fixed_period"
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Esegue soltanto i dataset indicati. "
            "Esempio: --dataset etth1 electricity"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Mostra i comandi senza eseguire il training."
        ),
    )

    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help=(
            "Interrompe tutto al primo comando fallito."
        ),
    )

    return parser.parse_args()


# ============================================================
# LETTURA FILE
# ============================================================

def load_json(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest non trovato:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        content = json.load(file)

    if not isinstance(content, dict):
        raise ValueError(
            "Il manifest deve contenere un oggetto JSON."
        )

    return content


def load_yaml(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise ValueError(
            f"YAML non valido: {path}"
        )

    return content


def save_yaml(
    path: Path,
    content: dict,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            content,
            file,
            sort_keys=False,
        )


@contextmanager
def temporary_yaml_updates(
    yaml_path: Path,
    updates: dict[str, Any],
) -> Iterator[None]:
    """
    Applica temporaneamente modifiche a un file YAML.

    Il contenuto originale viene sempre ripristinato,
    anche se il training fallisce.
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
        yield
    finally:
        save_yaml(
            yaml_path,
            original_config,
        )


# ============================================================
# CONFIGURAZIONI YAML
# ============================================================

def identify_dataset(
    yaml_path: Path,
    config: dict,
) -> str | None:
    """
    Identifica il dataset usando dataset_name,
    csv_path oppure il nome del file YAML.
    """

    searchable_text = " ".join(
        [
            str(
                config.get(
                    "dataset_name",
                    "",
                )
            ),
            str(
                config.get(
                    "csv_path",
                    "",
                )
            ),
            yaml_path.stem,
        ]
    ).lower()

    if "etth1" in searchable_text:
        return "etth1"

    if "ettm1" in searchable_text:
        return "ettm1"

    if "electricity" in searchable_text:
        return "electricity"

    return None


def find_yaml_files(
    enabled_datasets: set[str],
) -> list[tuple[Path, str]]:
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

    selected_files = []

    for yaml_path in yaml_files:
        config = load_yaml(
            yaml_path
        )

        dataset = identify_dataset(
            yaml_path=yaml_path,
            config=config,
        )

        if dataset is None:
            print(
                "Configurazione ignorata, "
                "dataset non riconosciuto:",
                yaml_path.name,
            )

            continue

        if dataset not in enabled_datasets:
            continue

        selected_files.append(
            (
                yaml_path,
                dataset,
            )
        )

    return selected_files


# ============================================================
# GENERAZIONE COMBINAZIONI
# ============================================================

def normalize_int_values(
    value: Any,
    default: list[int | None],
) -> list[int | None]:
    if value is None:
        return default

    if not isinstance(value, list):
        raise TypeError(
            "I valori sweep devono essere liste."
        )

    return [
        int(item)
        if item is not None
        else None
        for item in value
    ]


def normalize_string_values(
    value: Any,
    default: list[str | None],
) -> list[str | None]:
    if value is None:
        return default

    if not isinstance(value, list):
        raise TypeError(
            "model_classes deve essere una lista."
        )

    return [
        str(item)
        if item is not None
        else None
        for item in value
    ]


def generate_jobs(
    group: dict,
    yaml_path: Path,
    dataset: str,
) -> list[dict]:
    """
    Genera il prodotto cartesiano degli sweep.
    """

    group_seq_lengths = normalize_int_values(
        group.get("seq_len_values"),
        default=[None],
    )

    jobs = []

    for model_config in group.get(
        "models",
        [],
    ):
        model_name = str(
            model_config["model"]
        )

        top_k_values = normalize_int_values(
            model_config.get(
                "top_k_values"
            ),
            default=[None],
        )

        num_blocks_values = normalize_int_values(
            model_config.get(
                "num_blocks_values"
            ),
            default=[None],
        )

        fixed_period_values = normalize_int_values(
            model_config.get(
                "fixed_period_values"
            ),
            default=[None],
        )

        model_classes = normalize_string_values(
            model_config.get(
                "model_classes"
            ),
            default=[None],
        )

        combinations = itertools.product(
            group_seq_lengths,
            top_k_values,
            num_blocks_values,
            fixed_period_values,
            model_classes,
        )

        for (
            seq_len,
            top_k,
            num_blocks,
            fixed_period,
            model_class,
        ) in combinations:
            jobs.append(
                {
                    "group": group["name"],
                    "dataset": dataset,
                    "config_path": yaml_path,
                    "config_name": yaml_path.stem,
                    "model": model_name,
                    "seq_len": seq_len,
                    "top_k": top_k,
                    "num_blocks": num_blocks,
                    "fixed_period": fixed_period,
                    "model_class": model_class,
                }
            )

    return jobs


# ============================================================
# COSTRUZIONE COMANDO
# ============================================================

def build_command(
    job: dict,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.train_prova",
        "--config",
        job["config_name"],
        "--model",
        job["model"],
    ]

    if job["model_class"] is not None:
        command.extend(
            [
                "--model-class",
                str(
                    job["model_class"]
                ),
            ]
        )

    if job["top_k"] is not None:
        command.extend(
            [
                "--top-k-values",
                str(
                    job["top_k"]
                ),
            ]
        )

    if job["num_blocks"] is not None:
        command.extend(
            [
                "--num-blocks-values",
                str(
                    job["num_blocks"]
                ),
            ]
        )

    return command


def build_yaml_updates(
    job: dict,
) -> dict:
    updates = {}

    if job["seq_len"] is not None:
        updates["seq_len"] = int(
            job["seq_len"]
        )

    if job["fixed_period"] is not None:
        updates["fixed_period"] = int(
            job["fixed_period"]
        )

    if job["num_blocks"] is not None:
        updates["num_blocks"] = int(
            job["num_blocks"]
        )

        updates["num_times_blocks"] = int(
            job["num_blocks"]
        )

    return updates


# ============================================================
# OUTPUT PREVISTO
# ============================================================

def sanitize_name(
    value: str,
) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def get_expected_output_directory(
    job: dict,
    original_config: dict,
    experiments_root: Path,
) -> Path:
    """
    Ricostruisce il nome piatto previsto da train_prova.py.

    Serve solo per skip_existing.
    """

    seq_len = int(
        job["seq_len"]
        if job["seq_len"] is not None
        else original_config["seq_len"]
    )

    pred_len = int(
        original_config["pred_len"]
    )

    name_parts = [
        sanitize_name(
            job["model"]
        ),
    ]

    if (
        job["model"]
        == "timesnet_light"
        and job["model_class"]
        is not None
    ):
        backbone_name = (
            str(
                job["model_class"]
            )
            .replace(
                "LightTimesNet",
                "",
            )
            .strip("_")
            .lower()
        )

        name_parts.append(
            sanitize_name(
                backbone_name or "base"
            )
        )

    name_parts.extend(
        [
            sanitize_name(
                job["dataset"]
            ),
            "seq",
            str(seq_len),
            "pred",
            str(pred_len),
        ]
    )

    if job["top_k"] is not None:
        name_parts.extend(
            [
                "top",
                str(
                    job["top_k"]
                ),
            ]
        )

    if job["fixed_period"] is not None:
        name_parts.extend(
            [
                "period",
                str(
                    job["fixed_period"]
                ),
            ]
        )

    if job["num_blocks"] is not None:
        name_parts.extend(
            [
                "tb",
                str(
                    job["num_blocks"]
                ),
            ]
        )

    return (
        experiments_root
        / "_".join(name_parts)
    )


# ============================================================
# ESECUZIONE
# ============================================================

def run_job(
    job: dict,
    experiments_root: Path,
    dry_run: bool,
    skip_existing: bool,
) -> dict:
    original_config = load_yaml(
        job["config_path"]
    )

    expected_output = (
        get_expected_output_directory(
            job=job,
            original_config=original_config,
            experiments_root=(
                experiments_root
            ),
        )
    )

    metrics_path = (
        expected_output
        / "metrics.json"
    )

    command = build_command(
        job
    )

    yaml_updates = build_yaml_updates(
        job
    )

    command_text = " ".join(
        command
    )

    print("\n" + "=" * 80)

    print(
        "Gruppo:",
        job["group"],
    )

    print(
        "Dataset:",
        job["dataset"],
    )

    print(
        "Config:",
        job["config_name"],
    )

    print(
        "Modello:",
        job["model"],
    )

    print(
        "Aggiornamenti YAML:",
        yaml_updates,
    )

    print(
        "Comando:",
        command_text,
    )

    print(
        "Output previsto:",
        expected_output,
    )

    print("=" * 80)

    record = {
        "group": job["group"],
        "dataset": job["dataset"],
        "config": job["config_name"],
        "model": job["model"],
        "model_class": job["model_class"],
        "seq_len": job["seq_len"],
        "top_k": job["top_k"],
        "num_blocks": job["num_blocks"],
        "fixed_period": job["fixed_period"],
        "command": command_text,
        "yaml_updates": yaml_updates,
        "expected_output": str(
            expected_output
        ),
    }

    if (
        skip_existing
        and metrics_path.exists()
    ):
        print(
            "SKIP: metrics.json già presente."
        )

        record["status"] = "skipped"

        return record

    if dry_run:
        print(
            "DRY RUN: comando non eseguito."
        )

        record["status"] = "dry_run"

        return record

    environment = os.environ.copy()

    environment["PYTHONPATH"] = str(
        PROJECT_ROOT
    )

    environment["EXPERIMENTS_DIR"] = str(
        experiments_root
    )

    start_time = time.perf_counter()

    with temporary_yaml_updates(
        yaml_path=job["config_path"],
        updates=yaml_updates,
    ):
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            text=True,
        )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    record["returncode"] = int(
        result.returncode
    )

    record["elapsed_seconds"] = float(
        elapsed_seconds
    )

    record["status"] = (
        "completed"
        if result.returncode == 0
        else "failed"
    )

    return record


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    manifest_path = Path(
        args.manifest
    ).expanduser().resolve()

    manifest = load_json(
        manifest_path
    )

    settings = manifest.get(
        "settings",
        {},
    )

    stop_on_error = bool(
        args.stop_on_error
        or settings.get(
            "stop_on_error",
            False,
        )
    )

    skip_existing = bool(
        settings.get(
            "skip_existing",
            True,
        )
    )

    requested_groups = (
        {
            str(name).strip().lower()
            for name in args.group
        }
        if args.group
        else None
    )

    requested_datasets = (
        {
            str(name).strip().lower()
            for name in args.dataset
        }
        if args.dataset
        else None
    )

    if requested_datasets:
        invalid_datasets = (
            requested_datasets
            - SUPPORTED_DATASETS
        )

        if invalid_datasets:
            raise ValueError(
                "Dataset non supportati: "
                f"{sorted(invalid_datasets)}"
            )

    experiments_root = get_experiments_root(
        create=True
    )

    selected_groups = []

    for group in manifest.get(
        "groups",
        [],
    ):
        group_name = str(
            group.get(
                "name",
                "",
            )
        ).strip().lower()

        if not group_name:
            raise ValueError(
                "Ogni gruppo deve avere un nome."
            )

        if not group.get(
            "enabled",
            True,
        ):
            continue

        if (
            requested_groups is not None
            and group_name
            not in requested_groups
        ):
            continue

        selected_groups.append(
            group
        )

    if not selected_groups:
        raise ValueError(
            "Nessun gruppo selezionato."
        )

    jobs = []

    for group in selected_groups:
        group_datasets = {
            str(dataset).strip().lower()
            for dataset in group.get(
                "datasets",
                [],
            )
        }

        if requested_datasets is not None:
            group_datasets &= (
                requested_datasets
            )

        if not group_datasets:
            print(
                "Gruppo senza dataset selezionati:",
                group["name"],
            )

            continue

        yaml_files = find_yaml_files(
            enabled_datasets=group_datasets
        )

        for yaml_path, dataset in yaml_files:
            jobs.extend(
                generate_jobs(
                    group=group,
                    yaml_path=yaml_path,
                    dataset=dataset,
                )
            )

    if not jobs:
        raise ValueError(
            "Nessun esperimento generato."
        )

    print("=" * 80)
    print("RUN MANIFEST")
    print("=" * 80)

    print(
        "Manifest:",
        manifest_path,
    )

    print(
        "Experiments root:",
        experiments_root,
    )

    print(
        "Esperimenti generati:",
        len(jobs),
    )

    print(
        "Dry run:",
        args.dry_run,
    )

    print(
        "Skip existing:",
        skip_existing,
    )

    records = []

    for index, job in enumerate(
        jobs,
        start=1,
    ):
        print(
            f"\nEsperimento {index}/{len(jobs)}"
        )

        record = run_job(
            job=job,
            experiments_root=(
                experiments_root
            ),
            dry_run=args.dry_run,
            skip_existing=(
                skip_existing
            ),
        )

        records.append(
            record
        )

        if (
            record["status"] == "failed"
            and stop_on_error
        ):
            print(
                "Interruzione al primo errore."
            )

            break

    summary = {
        "manifest": str(
            manifest_path
        ),
        "experiments_root": str(
            experiments_root
        ),
        "dry_run": bool(
            args.dry_run
        ),
        "total_generated": len(
            jobs
        ),
        "total_processed": len(
            records
        ),
        "completed": sum(
            record["status"]
            == "completed"
            for record in records
        ),
        "skipped": sum(
            record["status"]
            == "skipped"
            for record in records
        ),
        "failed": sum(
            record["status"]
            == "failed"
            for record in records
        ),
        "dry_run_count": sum(
            record["status"]
            == "dry_run"
            for record in records
        ),
        "runs": records,
    }

    summary_path = (
        experiments_root
        / "run_manifest_summary.json"
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

    print("\n" + "=" * 80)
    print("MANIFEST TERMINATO")
    print("=" * 80)

    print(
        "Completati:",
        summary["completed"],
    )

    print(
        "Saltati:",
        summary["skipped"],
    )

    print(
        "Falliti:",
        summary["failed"],
    )

    print(
        "Dry run:",
        summary["dry_run_count"],
    )

    print(
        "Riepilogo:",
        summary_path,
    )

    if summary["failed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()