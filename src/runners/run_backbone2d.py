"""
Runner per confrontare i backbone 2D leggeri.

Output:

experiments/backbone2d/
"""

from __future__ import annotations
from src.runners.paths import get_experiments_root

from src.runners.runner_utils import (
    build_train_command,
    find_config_files,
    get_runner_output_dir,
    print_runner_header,
    run_training_command,
    write_runner_summary,
)


# GROUP_NAME = "backbone2d"


# Questi nomi devono corrispondere alle classi
# effettivamente disponibili in models_light.py.
BACKBONE_CLASSES = [
    "LightTimesNetMultiScale",
    "LightTimesNetDepthwise",
    "LightTimesNetGroup",
    "LightTimesNetSingleKernel",
]


def main() -> None:
    """
        output_dir = get_runner_output_dir(
        GROUP_NAME
    )
    """
    output_dir = get_experiments_root(
        create=True
    )

    yaml_files = find_config_files()

    print_runner_header(
        title="RUNNER BACKBONE 2D",
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

        print("\n" + "=" * 80)

        print(
            f"[{index}/{len(yaml_files)}] "
            f"Configurazione: {config_name}"
        )

        print("=" * 80)

        for model_class in BACKBONE_CLASSES:
            print(
                "\nBackbone:",
                model_class,
            )

            command = build_train_command(
                config_name=config_name,
                model_name="timesnet_light",
                model_class=model_class,
            )

            result = run_training_command(
                command=command,
                output_dir=output_dir,
            )

            record = {
                "config": config_name,
                "model": "timesnet_light",
                "model_class": model_class,
            }

            if result.returncode == 0:
                completed.append(
                    record
                )
            else:
                record[
                    "returncode"
                ] = result.returncode

                failed.append(
                    record
                )

    summary_path = write_runner_summary(
        output_dir=output_dir,
        runner_name="run_backbone2d",
        completed=completed,
        skipped=skipped,
        failed=failed,
    )

    print("\n" + "=" * 80)
    print("RUNNER BACKBONE 2D TERMINATO")
    print("=" * 80)

    print(
        "Completati:",
        len(completed),
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
        raise SystemExit(1)


if __name__ == "__main__":
    main()