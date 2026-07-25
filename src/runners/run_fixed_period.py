"""
Runner per il periodo di controllo 17.

I periodi 24 e 48 sono già presenti.

Esegue:

    fixed_period = 17
    num_times_blocks = 1

Struttura finale:

fixed_period_inception_period24 vs 48 vs 17/
└── fixed_period_inception_etth1_24/
    ├── period_24/
    ├── period_48/
    └── period_17/
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil

from src.runners.runner_utils import (
    build_train_command,
    find_config_files,
    get_runner_output_dir,
    print_runner_header,
    run_training_command,
    temporary_yaml_values,
    write_runner_summary,
)


GROUP_NAME = (
    "fixed_period_inception_"
    "period24 vs 48 vs 17"
)

FIXED_PERIOD = 17
NUM_TIMES_BLOCKS = 1

SKIP_COMPLETED = True
DELETE_TEMPORARY_DIRECTORY = True


def get_final_directory(
    output_dir: Path,
    config_name: str,
) -> Path:
    return (
        output_dir
        / f"fixed_period_inception_{config_name}"
        / f"period_{FIXED_PERIOD}"
    )


def find_generated_directory(
    output_dir: Path,
    config_name: str,
) -> Path | None:
    """
    Cerca la cartella generata da train_prova.py.
    """

    candidates = [
        (
            output_dir
            / (
                "fixed_period_inception_timesblocks_"
                f"{config_name}"
            )
            / f"period_{FIXED_PERIOD}"
            / (
                "num_times_blocks_"
                f"{NUM_TIMES_BLOCKS}"
            )
        ),
        (
            output_dir
            / (
                "fixed_period_inception_timesblocks_"
                f"{config_name}"
            )
            / f"period_{FIXED_PERIOD}"
        ),
        (
            output_dir
            / (
                "fixed_period_inception_"
                f"{config_name}"
            )
            / f"period_{FIXED_PERIOD}"
            / (
                "num_times_blocks_"
                f"{NUM_TIMES_BLOCKS}"
            )
        ),
        (
            output_dir
            / (
                "fixed_period_inception_"
                f"{config_name}"
            )
            / f"period_{FIXED_PERIOD}"
        ),
    ]

    for candidate in candidates:
        if (
            candidate.exists()
            and (
                candidate
                / "metrics.json"
            ).exists()
        ):
            return candidate

    for metrics_path in output_dir.rglob(
        "metrics.json"
    ):
        path_text = str(
            metrics_path
        ).lower()

        if (
            config_name.lower()
            in path_text
            and f"period_{FIXED_PERIOD}"
            in path_text
        ):
            return metrics_path.parent

    return None


def copy_results(
    source_dir: Path,
    destination_dir: Path,
) -> None:
    """
    Copia i risultati nella cartella finale period_17.
    """

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        source_dir.resolve()
        == destination_dir.resolve()
    ):
        return

    for source_item in source_dir.iterdir():
        destination_item = (
            destination_dir
            / source_item.name
        )

        if source_item.is_file():
            shutil.copy2(
                source_item,
                destination_item,
            )

        elif source_item.is_dir():
            shutil.copytree(
                source_item,
                destination_item,
                dirs_exist_ok=True,
            )

    if not (
        destination_dir
        / "metrics.json"
    ).exists():
        raise FileNotFoundError(
            "metrics.json non trovato dopo la copia:\n"
            f"{destination_dir}"
        )


def update_metrics(
    destination_dir: Path,
    config_name: str,
) -> None:
    """
    Aggiorna i metadati del risultato.
    """

    metrics_path = (
        destination_dir
        / "metrics.json"
    )

    if not metrics_path.exists():
        return

    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(
            file
        )

    metrics["config"] = config_name
    metrics["fixed_period"] = FIXED_PERIOD
    metrics["num_blocks"] = NUM_TIMES_BLOCKS
    metrics[
        "num_times_blocks"
    ] = NUM_TIMES_BLOCKS
    metrics[
        "experiment_directory"
    ] = str(destination_dir)

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )


def remove_temporary_directory(
    output_dir: Path,
    config_name: str,
) -> None:
    """
    Elimina:

    fixed_period_inception_timesblocks_<config>
    """

    temporary_root = (
        output_dir
        / (
            "fixed_period_inception_timesblocks_"
            f"{config_name}"
        )
    )

    if temporary_root.exists():
        shutil.rmtree(
            temporary_root
        )

        print(
            "Cartella temporanea eliminata:",
            temporary_root,
        )


def main() -> None:
    output_dir = get_runner_output_dir(
        GROUP_NAME
    )

    yaml_files = find_config_files()

    print_runner_header(
        title="RUNNER FIXED PERIOD 17",
        output_dir=output_dir,
        yaml_files=yaml_files,
    )

    completed = []
    skipped = []
    failed = []

    for index, yaml_file in enumerate(
        yaml_files,
        start=1,
    ):
        config_name = yaml_file.stem

        final_directory = get_final_directory(
            output_dir=output_dir,
            config_name=config_name,
        )

        final_metrics = (
            final_directory
            / "metrics.json"
        )

        print("\n" + "=" * 80)

        print(
            f"[{index}/{len(yaml_files)}] "
            f"Configurazione: {config_name}"
        )

        print(
            "Destinazione:",
            final_directory,
        )

        print("=" * 80)

        if (
            SKIP_COMPLETED
            and final_metrics.exists()
        ):
            print(
                "period_17 già presente. "
                "Training saltato."
            )

            skipped.append(
                {
                    "config": config_name,
                    "reason": (
                        "metrics.json già presente"
                    ),
                }
            )

            continue

        try:
            with temporary_yaml_values(
                yaml_path=yaml_file,
                updates={
                    "fixed_period": (
                        FIXED_PERIOD
                    ),
                },
            ):
                command = build_train_command(
                    config_name=config_name,
                    model_name=(
                        "fixed_period_inception"
                    ),
                    num_blocks_values=[
                        NUM_TIMES_BLOCKS
                    ],
                )

                result = run_training_command(
                    command=command,
                    output_dir=output_dir,
                )

            if result.returncode != 0:
                failed.append(
                    {
                        "config": config_name,
                        "stage": "training",
                        "returncode": (
                            result.returncode
                        ),
                    }
                )

                continue

            generated_directory = (
                find_generated_directory(
                    output_dir=output_dir,
                    config_name=config_name,
                )
            )

            if generated_directory is None:
                failed.append(
                    {
                        "config": config_name,
                        "stage": (
                            "find_generated_directory"
                        ),
                    }
                )

                continue

            print(
                "Risultati trovati in:",
                generated_directory,
            )

            copy_results(
                source_dir=generated_directory,
                destination_dir=final_directory,
            )

            update_metrics(
                destination_dir=final_directory,
                config_name=config_name,
            )

            if DELETE_TEMPORARY_DIRECTORY:
                remove_temporary_directory(
                    output_dir=output_dir,
                    config_name=config_name,
                )

            completed.append(
                {
                    "config": config_name,
                    "fixed_period": (
                        FIXED_PERIOD
                    ),
                    "num_times_blocks": (
                        NUM_TIMES_BLOCKS
                    ),
                    "directory": str(
                        final_directory
                    ),
                }
            )

            print(
                "Risultati finali:",
                final_directory,
            )

        except Exception as error:
            failed.append(
                {
                    "config": config_name,
                    "stage": "exception",
                    "error": repr(error),
                }
            )

    summary_path = write_runner_summary(
        output_dir=output_dir,
        runner_name="run_fixed_period",
        completed=completed,
        skipped=skipped,
        failed=failed,
    )

    print("\n" + "=" * 80)
    print("RUNNER FIXED PERIOD TERMINATO")
    print("=" * 80)

    print(
        "Completati:",
        len(completed),
    )

    print(
        "Saltati:",
        len(skipped),
    )

    print(
        "Falliti:",
        len(failed),
    )

    print(
        "Riepilogo:",
        summary_path,
    )

    if failed:
        print(
            "\nConfigurazioni fallite:"
        )

        for item in failed:
            print(
                "-",
                item,
            )

        raise SystemExit(1)


if __name__ == "__main__":
    main()