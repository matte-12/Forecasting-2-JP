"""
Runner per il confronto tra periodi fissi:

    24, 48 e 17

Modello:

    fixed_period_inception

Numero di TimesBlock:

    1

Struttura finale:

experiments/
└── fixed_period_inception_period_24_vs_48_vs_17/
    ├── fixed_period_inception_etth1_24/
    │   ├── period_24/
    │   ├── period_48/
    │   └── period_17/
    ├── fixed_period_inception_etth1_96/
    │   ├── period_24/
    │   ├── period_48/
    │   └── period_17/
    └── ...

Durante il training train_prova.py può creare una struttura
temporanea del tipo:

    fixed_period_inception_timesblocks_<config>/
    └── period_<period>/
        └── num_times_blocks_1/

Il runner copia i risultati nella struttura finale e rimuove
la cartella temporanea quando tutti i file sono stati trasferiti.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.runners.runner_utils import (
    build_train_command,
    find_config_files,
    get_runner_output_dir,
    print_runner_header,
    run_training_command,
    temporary_yaml_values,
    write_runner_summary,
)


# ============================================================
# CONFIGURAZIONE DEL RUNNER
# ============================================================

GROUP_NAME = (
    "fixed_period_inception_"
    "period_24_vs_48_vs_17"
)

FIXED_PERIODS = [
    24,
    48,
    17,
]

NUM_TIMES_BLOCKS = 1

# Se metrics.json è già presente nella cartella finale,
# il relativo esperimento non viene eseguito nuovamente.
SKIP_COMPLETED = True

# Elimina la cartella temporanea prodotta da train_prova.py
# dopo aver copiato correttamente tutti i risultati.
DELETE_TEMPORARY_DIRECTORIES = True


# ============================================================
# PERCORSI FINALI
# ============================================================

def get_config_output_directory(
    output_dir: Path,
    config_name: str,
) -> Path:
    """
    Restituisce la cartella principale della configurazione.

    Esempio:

        fixed_period_inception_period_24_vs_48_vs_17/
        └── fixed_period_inception_etth1_24/
    """

    return (
        output_dir
        / f"fixed_period_inception_{config_name}"
    )


def get_final_period_directory(
    output_dir: Path,
    config_name: str,
    fixed_period: int,
) -> Path:
    """
    Restituisce la cartella finale di un periodo.

    Esempio:

        fixed_period_inception_etth1_24/
        └── period_24/
    """

    return (
        get_config_output_directory(
            output_dir=output_dir,
            config_name=config_name,
        )
        / f"period_{fixed_period}"
    )


# ============================================================
# RICERCA DEI RISULTATI GENERATI
# ============================================================

def find_generated_directory(
    output_dir: Path,
    config_name: str,
    fixed_period: int,
) -> Path | None:
    """
    Cerca la cartella contenente metrics.json generata
    da train_prova.py.

    Supporta diverse possibili strutture di salvataggio.
    """

    temporary_experiment_name = (
        "fixed_period_inception_timesblocks_"
        f"{config_name}"
    )

    final_experiment_name = (
        f"fixed_period_inception_{config_name}"
    )

    candidates = [
        # Struttura prevista dal nuovo train_prova.py:
        #
        # fixed_period_inception_timesblocks_config/
        # └── period_X/
        #     └── num_times_blocks_1/
        (
            output_dir
            / temporary_experiment_name
            / f"period_{fixed_period}"
            / (
                "num_times_blocks_"
                f"{NUM_TIMES_BLOCKS}"
            )
        ),

        # Possibile variante senza cartella num_times_blocks:
        (
            output_dir
            / temporary_experiment_name
            / f"period_{fixed_period}"
        ),

        # Possibile struttura già quasi finale:
        (
            output_dir
            / final_experiment_name
            / f"period_{fixed_period}"
            / (
                "num_times_blocks_"
                f"{NUM_TIMES_BLOCKS}"
            )
        ),

        # Struttura finale:
        (
            output_dir
            / final_experiment_name
            / f"period_{fixed_period}"
        ),
    ]

    for candidate in candidates:
        metrics_path = (
            candidate
            / "metrics.json"
        )

        if (
            candidate.exists()
            and metrics_path.exists()
        ):
            return candidate

    # Ricerca di riserva nel caso train_prova.py
    # usi una struttura leggermente diversa.
    matching_directories = []

    for metrics_path in output_dir.rglob(
        "metrics.json"
    ):
        parent_directory = (
            metrics_path.parent
        )

        normalized_path = str(
            parent_directory
        ).lower()

        contains_config = (
            config_name.lower()
            in normalized_path
        )

        contains_period = (
            f"period_{fixed_period}"
            in normalized_path
        )

        contains_model = (
            "fixed_period_inception"
            in normalized_path
        )

        if (
            contains_config
            and contains_period
            and contains_model
        ):
            matching_directories.append(
                parent_directory
            )

    if not matching_directories:
        return None

    # Preferisce una cartella che contenga esplicitamente
    # num_times_blocks_1.
    matching_directories.sort(
        key=lambda directory: (
            (
                f"num_times_blocks_"
                f"{NUM_TIMES_BLOCKS}"
            )
            not in str(directory),
            len(str(directory)),
        )
    )

    return matching_directories[0]


# ============================================================
# COPIA DEI RISULTATI
# ============================================================

def copy_directory_contents(
    source_dir: Path,
    destination_dir: Path,
) -> None:
    """
    Copia tutto il contenuto della cartella sorgente nella
    cartella finale del periodo.
    """

    source_dir = source_dir.resolve()
    destination_dir = (
        destination_dir.resolve()
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Se train_prova.py ha già salvato direttamente
    # nella cartella finale, non serve copiare.
    if source_dir == destination_dir:
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

    final_metrics_path = (
        destination_dir
        / "metrics.json"
    )

    if not final_metrics_path.exists():
        raise FileNotFoundError(
            "La copia non è stata completata: "
            "metrics.json non è presente nella "
            "cartella finale.\n"
            f"Sorgente: {source_dir}\n"
            f"Destinazione: {destination_dir}"
        )


# ============================================================
# AGGIORNAMENTO METRICHE
# ============================================================

def update_metrics_file(
    destination_dir: Path,
    config_name: str,
    fixed_period: int,
) -> None:
    """
    Aggiorna metrics.json con i metadati coerenti
    con la struttura finale.
    """

    metrics_path = (
        destination_dir
        / "metrics.json"
    )

    if not metrics_path.exists():
        raise FileNotFoundError(
            "metrics.json non trovato:\n"
            f"{metrics_path}"
        )

    with metrics_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metrics = json.load(file)

    metrics.update(
        {
            "model": (
                "fixed_period_inception"
            ),
            "config": config_name,
            "fixed_period": int(
                fixed_period
            ),
            "num_blocks": int(
                NUM_TIMES_BLOCKS
            ),
            "num_times_blocks": int(
                NUM_TIMES_BLOCKS
            ),
            "experiment_directory": str(
                destination_dir
            ),
        }
    )

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )


# ============================================================
# ELIMINAZIONE CARTELLE TEMPORANEE
# ============================================================

def remove_empty_parent_directories(
    directory: Path,
    stop_directory: Path,
) -> None:
    """
    Elimina le cartelle vuote risalendo fino alla cartella
    temporanea principale, senza oltrepassare stop_directory.
    """

    current_directory = directory

    while (
        current_directory.exists()
        and current_directory
        != stop_directory
    ):
        try:
            current_directory.rmdir()
        except OSError:
            break

        current_directory = (
            current_directory.parent
        )


def remove_temporary_period_directory(
    output_dir: Path,
    config_name: str,
    fixed_period: int,
) -> None:
    """
    Elimina soltanto la parte temporanea del periodo appena
    completato.

    Non elimina i risultati degli altri periodi.
    """

    temporary_root = (
        output_dir
        / (
            "fixed_period_inception_timesblocks_"
            f"{config_name}"
        )
    )

    temporary_period_directory = (
        temporary_root
        / f"period_{fixed_period}"
    )

    if temporary_period_directory.exists():
        shutil.rmtree(
            temporary_period_directory
        )

        print(
            "Cartella temporanea del periodo eliminata:",
            temporary_period_directory,
        )

    # Se la cartella temporanea principale è rimasta vuota,
    # viene eliminata.
    if temporary_root.exists():
        try:
            temporary_root.rmdir()

            print(
                "Cartella temporanea principale eliminata:",
                temporary_root,
            )

        except OSError:
            # La cartella contiene ancora altri periodi
            # o altri file.
            pass


# ============================================================
# VERIFICA ESPERIMENTO COMPLETATO
# ============================================================

def experiment_is_completed(
    final_directory: Path,
) -> bool:
    """
    Un esperimento è considerato completato quando nella
    cartella finale sono presenti almeno metrics.json
    e best_model.pth.
    """

    metrics_exists = (
        final_directory
        / "metrics.json"
    ).exists()

    checkpoint_exists = (
        final_directory
        / "best_model.pth"
    ).exists()

    return (
        metrics_exists
        and checkpoint_exists
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    output_dir = get_runner_output_dir(
        GROUP_NAME
    )

    yaml_files = find_config_files()

    print_runner_header(
        title=(
            "RUNNER FIXED PERIOD "
            "24 VS 48 VS 17"
        ),
        output_dir=output_dir,
        yaml_files=yaml_files,
    )

    print(
        "\nPeriodi da eseguire:",
        FIXED_PERIODS,
    )

    print(
        "Numero TimesBlock:",
        NUM_TIMES_BLOCKS,
    )

    completed = []
    skipped = []
    failed = []

    total_experiments = (
        len(yaml_files)
        * len(FIXED_PERIODS)
    )

    experiment_index = 0

    # ========================================================
    # CICLO SULLE CONFIGURAZIONI
    # ========================================================

    for yaml_file in yaml_files:
        config_name = yaml_file.stem

        # ====================================================
        # CICLO SUI PERIODI
        # ====================================================

        for fixed_period in FIXED_PERIODS:
            experiment_index += 1

            final_directory = (
                get_final_period_directory(
                    output_dir=output_dir,
                    config_name=config_name,
                    fixed_period=fixed_period,
                )
            )

            print("\n" + "=" * 80)

            print(
                f"[{experiment_index}/"
                f"{total_experiments}]"
            )

            print(
                "Configurazione:",
                config_name,
            )

            print(
                "Periodo fisso:",
                fixed_period,
            )

            print(
                "Numero TimesBlock:",
                NUM_TIMES_BLOCKS,
            )

            print(
                "Destinazione:",
                final_directory,
            )

            print("=" * 80)

            # ------------------------------------------------
            # SKIP DEGLI ESPERIMENTI GIÀ COMPLETATI
            # ------------------------------------------------

            if (
                SKIP_COMPLETED
                and experiment_is_completed(
                    final_directory
                )
            ):
                print(
                    "Esperimento già completato. "
                    "Training saltato."
                )

                skipped.append(
                    {
                        "config": config_name,
                        "fixed_period": (
                            fixed_period
                        ),
                        "num_times_blocks": (
                            NUM_TIMES_BLOCKS
                        ),
                        "directory": str(
                            final_directory
                        ),
                        "reason": (
                            "metrics.json e "
                            "best_model.pth già presenti"
                        ),
                    }
                )

                continue

            # ------------------------------------------------
            # TRAINING
            # ------------------------------------------------

            try:
                # Modifica temporaneamente fixed_period
                # nel file YAML.
                #
                # Al termine il file originale viene sempre
                # ripristinato.
                with temporary_yaml_values(
                    yaml_path=yaml_file,
                    updates={
                        "fixed_period": int(
                            fixed_period
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

                if result.returncode != 0:
                    failed.append(
                        {
                            "config": config_name,
                            "fixed_period": (
                                fixed_period
                            ),
                            "num_times_blocks": (
                                NUM_TIMES_BLOCKS
                            ),
                            "stage": "training",
                            "returncode": (
                                result.returncode
                            ),
                        }
                    )

                    print(
                        "Training fallito con "
                        f"return code {result.returncode}."
                    )

                    continue

                # ------------------------------------------------
                # RICERCA DEI RISULTATI
                # ------------------------------------------------

                generated_directory = (
                    find_generated_directory(
                        output_dir=output_dir,
                        config_name=config_name,
                        fixed_period=(
                            fixed_period
                        ),
                    )
                )

                if generated_directory is None:
                    failed.append(
                        {
                            "config": config_name,
                            "fixed_period": (
                                fixed_period
                            ),
                            "num_times_blocks": (
                                NUM_TIMES_BLOCKS
                            ),
                            "stage": (
                                "find_generated_directory"
                            ),
                            "error": (
                                "Nessuna cartella con "
                                "metrics.json trovata"
                            ),
                        }
                    )

                    print(
                        "Impossibile trovare i risultati "
                        "generati da train_prova.py."
                    )

                    continue

                print(
                    "Risultati generati in:",
                    generated_directory,
                )

                # ------------------------------------------------
                # COPIA NELLA STRUTTURA FINALE
                # ------------------------------------------------

                copy_directory_contents(
                    source_dir=(
                        generated_directory
                    ),
                    destination_dir=(
                        final_directory
                    ),
                )

                update_metrics_file(
                    destination_dir=(
                        final_directory
                    ),
                    config_name=config_name,
                    fixed_period=fixed_period,
                )

                # ------------------------------------------------
                # RIMOZIONE CARTELLA TEMPORANEA
                # ------------------------------------------------

                if DELETE_TEMPORARY_DIRECTORIES:
                    remove_temporary_period_directory(
                        output_dir=output_dir,
                        config_name=config_name,
                        fixed_period=(
                            fixed_period
                        ),
                    )

                completed.append(
                    {
                        "config": config_name,
                        "fixed_period": int(
                            fixed_period
                        ),
                        "num_times_blocks": int(
                            NUM_TIMES_BLOCKS
                        ),
                        "directory": str(
                            final_directory
                        ),
                    }
                )

                print(
                    "Esperimento completato."
                )

                print(
                    "Risultati finali:",
                    final_directory,
                )

            except Exception as error:
                failed.append(
                    {
                        "config": config_name,
                        "fixed_period": int(
                            fixed_period
                        ),
                        "num_times_blocks": int(
                            NUM_TIMES_BLOCKS
                        ),
                        "stage": "exception",
                        "error": repr(error),
                    }
                )

                print(
                    "Errore durante l'esperimento:"
                )

                print(
                    repr(error)
                )

    # ========================================================
    # RIEPILOGO
    # ========================================================

    summary_path = write_runner_summary(
        output_dir=output_dir,
        runner_name="run_fixed_period",
        completed=completed,
        skipped=skipped,
        failed=failed,
    )

    print("\n" + "=" * 80)
    print(
        "RUNNER FIXED PERIOD TERMINATO"
    )
    print("=" * 80)

    print(
        "Esperimenti totali:",
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
        "Cartella risultati:",
        output_dir,
    )

    print(
        "Riepilogo JSON:",
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