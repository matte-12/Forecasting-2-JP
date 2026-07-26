"""
Runner per studiare l'effetto del numero di cicli osservati.

Il runner usa i file YAML già presenti e modifica
temporaneamente seq_len con i valori:

    96, 192, 384

Per ogni seq_len esegue:

    - DLinear
    - TCN
    - FixedPeriodInception

FixedPeriodInception usa:

    fixed_period = 24
    num_times_blocks = 1

I file YAML originali vengono sempre ripristinati.

I risultati vengono salvati direttamente in EXPERIMENTS_ROOT,
con nomi piatti, per esempio:

    dlinear_etth1_seq_96_pred_24/
    dlinear_etth1_seq_192_pred_24/
    dlinear_etth1_seq_384_pred_24/

    tcn_etth1_seq_96_pred_24/
    tcn_etth1_seq_192_pred_24/
    tcn_etth1_seq_384_pred_24/

    fixed_period_inception_etth1_seq_96_pred_24_period_24_tb_1/
    fixed_period_inception_etth1_seq_192_pred_24_period_24_tb_1/
    fixed_period_inception_etth1_seq_384_pred_24_period_24_tb_1/
"""

from __future__ import annotations
from src.runners.dataset_selection import (
    get_enabled_datasets,
    print_enabled_datasets,
)

from src.runners.paths import get_experiments_root
from src.runners.runner_utils import (
    build_train_command,
    find_config_files,
    load_yaml,
    print_runner_header,
    run_training_command,
    temporary_yaml_values,
    write_runner_summary,
)


# ============================================================
# CONFIGURAZIONE DEL RUNNER
# ============================================================

SEQ_LENGTHS = [
    96,
    192,
    384,
]

FIXED_PERIOD = 24

NUM_TIMES_BLOCKS = 1


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    # Tutti i risultati vengono salvati direttamente
    # dentro EXPERIMENTS_ROOT.
    output_dir = get_experiments_root(
        create=True
    )

    runner_name = "run_cicli"

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
        title="RUNNER CICLI: SEQ_LEN 96 VS 192 VS 384",
        output_dir=output_dir,
        yaml_files=yaml_files,
    )

    print(
        "\nSeq_len da eseguire:",
        SEQ_LENGTHS,
    )

    print(
        "Periodo FixedPeriodInception:",
        FIXED_PERIOD,
    )

    print(
        "Numero TimesBlock:",
        NUM_TIMES_BLOCKS,
    )

    completed = []
    skipped = []
    failed = []

    models_per_seq_len = 3

    total_experiments = (
        len(yaml_files)
        * len(SEQ_LENGTHS)
        * models_per_seq_len
    )

    experiment_index = 0

    # ========================================================
    # CICLO SULLE CONFIGURAZIONI YAML ESISTENTI
    # ========================================================

    for yaml_file in yaml_files:
        config_name = yaml_file.stem

        original_config = load_yaml(
            yaml_file
        )

        pred_len = int(
            original_config["pred_len"]
        )

        dataset_name = original_config.get(
            "dataset_name",
            config_name,
        )

        print("\n" + "#" * 80)

        print(
            "Configurazione base:",
            config_name,
        )

        print(
            "Dataset:",
            dataset_name,
        )

        print(
            "Pred_len:",
            pred_len,
        )

        print("#" * 80)

        # ====================================================
        # CICLO SU SEQ_LEN
        # ====================================================

        for seq_len in SEQ_LENGTHS:
            print("\n" + "=" * 80)

            print(
                "Configurazione:",
                config_name,
            )

            print(
                "Dataset:",
                dataset_name,
            )

            print(
                "Seq_len temporanea:",
                seq_len,
            )

            print(
                "Pred_len:",
                pred_len,
            )

            print("=" * 80)

            # Tutti i modelli di questo gruppo vedranno
            # temporaneamente la stessa seq_len.
            with temporary_yaml_values(
                yaml_path=yaml_file,
                updates={
                    "seq_len": int(
                        seq_len
                    ),
                },
            ):

                # ============================================
                # DLINEAR
                # ============================================

                experiment_index += 1

                print("\n" + "-" * 80)

                print(
                    f"[{experiment_index}/"
                    f"{total_experiments}]"
                )

                print(
                    "Modello: DLinear"
                )

                print(
                    f"seq_len={seq_len} | "
                    f"pred_len={pred_len}"
                )

                print("-" * 80)

                command = build_train_command(
                    config_name=config_name,
                    model_name="dlinear",
                )

                result = run_training_command(
                    command=command,
                    output_dir=output_dir,
                )

                record = {
                    "config": config_name,
                    "dataset": dataset_name,
                    "model": "dlinear",
                    "seq_len": int(
                        seq_len
                    ),
                    "pred_len": int(
                        pred_len
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

                # ============================================
                # TCN
                # ============================================

                experiment_index += 1

                print("\n" + "-" * 80)

                print(
                    f"[{experiment_index}/"
                    f"{total_experiments}]"
                )

                print(
                    "Modello: TCN"
                )

                print(
                    f"seq_len={seq_len} | "
                    f"pred_len={pred_len}"
                )

                print("-" * 80)

                command = build_train_command(
                    config_name=config_name,
                    model_name="tcn",
                )

                result = run_training_command(
                    command=command,
                    output_dir=output_dir,
                )

                record = {
                    "config": config_name,
                    "dataset": dataset_name,
                    "model": "tcn",
                    "seq_len": int(
                        seq_len
                    ),
                    "pred_len": int(
                        pred_len
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

                # ============================================
                # FIXED PERIOD INCEPTION
                # ============================================

                experiment_index += 1

                print("\n" + "-" * 80)

                print(
                    f"[{experiment_index}/"
                    f"{total_experiments}]"
                )

                print(
                    "Modello: FixedPeriodInception"
                )

                print(
                    f"seq_len={seq_len} | "
                    f"pred_len={pred_len} | "
                    f"period={FIXED_PERIOD} | "
                    f"tb={NUM_TIMES_BLOCKS}"
                )

                print("-" * 80)

                # Viene applicato un secondo aggiornamento
                # temporaneo allo stesso YAML:
                #
                # seq_len resta quella del ciclo esterno;
                # fixed_period e numero di blocchi vengono
                # impostati per FixedPeriodInception.
                with temporary_yaml_values(
                    yaml_path=yaml_file,
                    updates={
                        "seq_len": int(
                            seq_len
                        ),
                        "fixed_period": int(
                            FIXED_PERIOD
                        ),
                        "num_times_blocks": int(
                            NUM_TIMES_BLOCKS
                        ),
                        "num_blocks": int(
                            NUM_TIMES_BLOCKS
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

                record = {
                    "config": config_name,
                    "dataset": dataset_name,
                    "model": (
                        "fixed_period_inception"
                    ),
                    "seq_len": int(
                        seq_len
                    ),
                    "pred_len": int(
                        pred_len
                    ),
                    "fixed_period": int(
                        FIXED_PERIOD
                    ),
                    "num_times_blocks": int(
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

    # ========================================================
    # RIEPILOGO
    # ========================================================

    summary_path = write_runner_summary(
        output_dir=output_dir,
        runner_name="run_cicli",
        completed=completed,
        skipped=skipped,
        failed=failed,
    )

    print("\n" + "=" * 80)

    print(
        "RUNNER CICLI TERMINATO"
    )

    print("=" * 80)

    print(
        "Esperimenti previsti:",
        total_experiments,
    )

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
        "Output:",
        output_dir,
    )

    print(
        "Riepilogo:",
        summary_path,
    )

    if failed:
        print(
            "\nEsperimenti falliti:"
        )

        for item in failed:
            print(
                "-",
                item,
            )

        raise SystemExit(1)


if __name__ == "__main__":
    main()