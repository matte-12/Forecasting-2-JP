"""
Configurazione centralizzata dei dataset da eseguire
per ciascun runner.

Per abilitare o disabilitare un dataset basta modificare
RUNNER_DATASETS.

Nomi supportati:

    etth1
    ettm1
    electricity
"""

from __future__ import annotations


SUPPORTED_DATASETS = {
    "etth1",
    "ettm1",
    "electricity",
}


# ============================================================
# DATASET ABILITATI PER CIASCUN RUNNER
# ============================================================

RUNNER_DATASETS = {
    # Sweep seq_len 96, 192 e 384
    "run_cicli": [
        "etth1",
       # "ettm1",
       # "electricity",
    ],

    # Sweep top_k × numero di TimesBlock
    "run_times_block": [
        "etth1",
       # "ettm1",
       # "electricity",
    ],

    # Confronto dei backbone leggeri
    "run_backbone2d": [
        "etth1",
       # "ettm1",
       # "electricity",
    ],

    # Confronto periodi 24, 48 e 17
    "run_fixed_period": [
        "etth1",
       # "ettm1",
       # "electricity",
    ],
}


def get_enabled_datasets(
    runner_name: str,
) -> list[str]:
    """
    Restituisce i dataset abilitati per uno specifico runner.

    Args:
        runner_name:
            Nome del runner, per esempio:

                run_cicli
                run_times_block
                run_backbone2d
                run_fixed_period

    Returns:
        Lista dei dataset abilitati.

    Raises:
        KeyError:
            Se il runner non è presente in RUNNER_DATASETS.

        ValueError:
            Se è stato indicato un dataset non supportato
            oppure se la lista è vuota.
    """

    runner_name = str(
        runner_name
    ).strip()

    if runner_name not in RUNNER_DATASETS:
        available_runners = sorted(
            RUNNER_DATASETS
        )

        raise KeyError(
            f"Runner non configurato: {runner_name}. "
            f"Runner disponibili: {available_runners}"
        )

    datasets = [
        str(dataset).strip().lower()
        for dataset in RUNNER_DATASETS[
            runner_name
        ]
    ]

    if not datasets:
        raise ValueError(
            f"Nessun dataset abilitato per {runner_name}."
        )

    invalid_datasets = [
        dataset
        for dataset in datasets
        if dataset not in SUPPORTED_DATASETS
    ]

    if invalid_datasets:
        raise ValueError(
            "Dataset non supportati per "
            f"{runner_name}: {invalid_datasets}. "
            "Dataset ammessi: "
            f"{sorted(SUPPORTED_DATASETS)}"
        )

    # Rimuove eventuali duplicati preservando l'ordine.
    unique_datasets = list(
        dict.fromkeys(
            datasets
        )
    )

    return unique_datasets


def dataset_is_enabled(
    runner_name: str,
    dataset_name: str,
) -> bool:
    """
    Controlla se un dataset è abilitato per un runner.

    Esempio:

        dataset_is_enabled(
            "run_cicli",
            "ettm1",
        )

    restituisce False se ETTm1 è disabilitato.
    """

    dataset_name = str(
        dataset_name
    ).strip().lower()

    return dataset_name in get_enabled_datasets(
        runner_name
    )


def print_enabled_datasets(
    runner_name: str,
) -> None:
    """
    Stampa i dataset abilitati per il runner.
    """

    datasets = get_enabled_datasets(
        runner_name
    )

    print(
        f"Dataset abilitati per {runner_name}:"
    )

    for dataset in datasets:
        print(
            f"- {dataset}"
        )