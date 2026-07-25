"""
Runner per il confronto tra:

- DLinear
- CausalTCN
- FixedPeriodInception con periodo 24

Output:

experiments/cicli/
"""

from __future__ import annotations
from src.runners.paths import get_experiments_root
from src.runners.runner_utils import (
    build_train_command,
    find_config_files,
    get_runner_output_dir,
    print_runner_header,
    run_training_command,
    temporary_yaml_values,
    write_runner_summary,
)




FIXED_PERIOD = 24
NUM_TIMES_BLOCKS = 1


def main() -> None:
    output_dir = get_experiments_root(
        create=True
    )

    yaml_files = find_config_files()

    print_runner_header(
        title="RUNNER CICLI",
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

        # DLinear e TCN non dipendono dal fixed_period.
        experiments = [
            {
                "model": "dlinear",
                "num_blocks": None,
            },
            {
                "model": "tcn",
                "num_blocks": None,
            },
        ]

        for experiment in experiments:
            model_name = experiment[
                "model"
            ]

            print(
                f"\nModello: {model_name}"
            )

            command = build_train_command(
                config_name=config_name,
                model_name=model_name,
            )

            result = run_training_command(
                command=command,
                output_dir=output_dir,
            )

            record = {
                "config": config_name,
                "model": model_name,
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

        # FixedPeriodInception con periodo 24.
        with temporary_yaml_values(
            yaml_path=yaml_file,
            updates={
                "fixed_period": (
                    FIXED_PERIOD
                ),
            },
        ):
            print(
                "\nModello: fixed_period_inception"
            )

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

            record = {
                "config": config_name,
                "model": (
                    "fixed_period_inception"
                ),
                "fixed_period": (
                    FIXED_PERIOD
                ),
                "num_times_blocks": (
                    NUM_TIMES_BLOCKS
                ),
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
        runner_name="run_cicli",
        completed=completed,
        skipped=skipped,
        failed=failed,
    )

    print("\n" + "=" * 80)
    print("RUNNER CICLI TERMINATO")
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