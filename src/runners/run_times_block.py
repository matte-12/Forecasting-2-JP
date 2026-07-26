"""
Runner per analizzare:

TimesNet:
    top_k = 1, 2, 3
    num_times_blocks = 1, 2, 3

FixedPeriodInception:
    fixed_period = 24
    num_times_blocks = 1, 2, 3

Output:

experiments/times_block/
"""
from src.runners.paths import get_experiments_root
from __future__ import annotations
from src.runners.dataset_selection import (
    get_enabled_datasets,
    print_enabled_datasets,
)
from src.runners.runner_utils import (
    build_train_command,
    find_config_files,
    get_runner_output_dir,
    print_runner_header,
    run_training_command,
    temporary_yaml_values,
    write_runner_summary,
)




TOP_K_VALUES = [
    1,
    2,
    3,
]

NUM_BLOCKS_VALUES = [
    1,
    2,
    3,
]

FIXED_PERIOD = 24


def main() -> None:
    output_dir = get_experiments_root(
        create=True
    )

    runner_name = "run_times_block"

    enabled_datasets = get_enabled_datasets(
        runner_name
    )

    yaml_files = find_config_files(
        dataset_names=enabled_datasets
    )

    print_enabled_datasets(
        runner_name
    )

    print_runner_header(
        title="RUNNER TIMES BLOCK",
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

        # ====================================================
        # TIMESNET FFT
        # ====================================================

        print(
            "\nTimesNet FFT"
        )

        command = build_train_command(
            config_name=config_name,
            model_name="timesnet_original",
            top_k_values=TOP_K_VALUES,
            num_blocks_values=(
                NUM_BLOCKS_VALUES
            ),
        )

        result = run_training_command(
            command=command,
            output_dir=output_dir,
        )

        record = {
            "config": config_name,
            "model": "timesnet",
            "top_k_values": TOP_K_VALUES,
            "num_blocks_values": (
                NUM_BLOCKS_VALUES
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

        # ====================================================
        # FIXED PERIOD
        # ====================================================

        with temporary_yaml_values(
            yaml_path=yaml_file,
            updates={
                "fixed_period": (
                    FIXED_PERIOD
                ),
            },
        ):
            print(
                "\nFixedPeriodInception"
            )

            command = build_train_command(
                config_name=config_name,
                model_name=(
                    "fixed_period_inception"
                ),
                num_blocks_values=(
                    NUM_BLOCKS_VALUES
                ),
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
                "num_blocks_values": (
                    NUM_BLOCKS_VALUES
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
        runner_name="run_times_block",
        completed=completed,
        skipped=skipped,
        failed=failed,
    )

    print("\n" + "=" * 80)
    print("RUNNER TIMES BLOCK TERMINATO")
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