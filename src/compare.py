import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMNS = [
    "model",
    "config",
    "dataset",
    "seq_len",
    "pred_len",
    "fixed_period",
    "top_k",
    "device",
    "test_mse",
    "test_mae",
    "trainable_parameters",
    "average_epoch_time_seconds",
    "total_training_time_seconds",
    "inference_ms_per_sample",
    "samples_per_second",
    "peak_gpu_memory_mb",
    "checkpoint_size_mb",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Confronta i metrics.json prodotti "
            "dagli esperimenti."
        )
    )

    parser.add_argument(
        "--experiments-dir",
        type=str,
        default=None,
        help=(
            "Cartella principale degli esperimenti. "
            "Se omessa usa EXPERIMENTS_DIR oppure "
            "<project_root>/experiments."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Cartella in cui salvare tabella e grafici. "
            "Default: <experiments-dir>/comparison."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filtra per dataset, ad esempio ETTh1.",
    )

    parser.add_argument(
        "--pred-len",
        type=int,
        default=None,
        help="Filtra per orizzonte di previsione.",
    )

    return parser.parse_args()


def resolve_experiments_dir(
    argument: str | None,
) -> Path:
    project_root = (
        Path(__file__).resolve().parent.parent
    )

    if argument is not None:
        experiments_dir = Path(
            argument
        ).expanduser()

    else:
        experiments_dir = Path(
            os.environ.get(
                "EXPERIMENTS_DIR",
                project_root / "experiments",
            )
        )

    if not experiments_dir.exists():
        raise FileNotFoundError(
            "Cartella degli esperimenti non trovata: "
            f"{experiments_dir}"
        )

    return experiments_dir


def load_metrics(
    experiments_dir: Path,
) -> pd.DataFrame:
    rows = []

    metrics_files = sorted(
        experiments_dir.rglob("metrics.json")
    )

    if not metrics_files:
        raise FileNotFoundError(
            "Nessun metrics.json trovato in: "
            f"{experiments_dir}"
        )

    for metrics_file in metrics_files:
        try:
            with metrics_file.open(
                "r",
                encoding="utf-8",
            ) as file:
                metrics = json.load(file)

        except (
            json.JSONDecodeError,
            OSError,
        ) as error:
            print(
                f"Saltato {metrics_file}: {error}"
            )
            continue

        metrics["metrics_path"] = str(
            metrics_file
        )

        metrics["experiment_dir"] = str(
            metrics_file.parent
        )

        rows.append(metrics)

    if not rows:
        raise RuntimeError(
            "Nessun metrics.json valido trovato."
        )

    dataframe = pd.DataFrame(rows)

    for column in METRIC_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = None

    return dataframe


def build_experiment_label(
    row: pd.Series,
) -> str:
    parts = [
        str(row.get("model", "unknown")),
        str(row.get("config", "unknown")),
    ]

    fixed_period = row.get("fixed_period")

    if pd.notna(fixed_period):
        parts.append(
            f"P{int(fixed_period)}"
        )

    top_k = row.get("top_k")

    if pd.notna(top_k):
        parts.append(
            f"k{int(top_k)}"
        )

    return " | ".join(parts)


def save_bar_plot(
    dataframe: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
    lower_is_better: bool,
) -> None:
    plot_data = dataframe[
        ["experiment_label", metric]
    ].dropna()

    if plot_data.empty:
        print(
            f"Grafico {metric} saltato: "
            "nessun dato disponibile."
        )
        return

    plot_data = plot_data.sort_values(
        metric,
        ascending=lower_is_better,
    )

    figure_width = max(
        8,
        len(plot_data) * 1.2,
    )

    plt.figure(
        figsize=(figure_width, 6)
    )

    plt.bar(
        plot_data["experiment_label"],
        plot_data[metric],
    )

    plt.ylabel(ylabel)
    plt.xlabel("Esperimento")
    plt.title(f"Confronto {metric}")
    plt.xticks(
        rotation=45,
        ha="right",
    )
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
    )
    plt.close()


def main():
    args = parse_args()

    experiments_dir = (
        resolve_experiments_dir(
            args.experiments_dir
        )
    )

    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir is not None
        else experiments_dir / "comparison"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_metrics(
        experiments_dir
    )

    if args.dataset is not None:
        dataframe = dataframe[
            dataframe["dataset"]
            == args.dataset
        ]

    if args.pred_len is not None:
        dataframe = dataframe[
            dataframe["pred_len"]
            == args.pred_len
        ]

    if dataframe.empty:
        raise RuntimeError(
            "Nessun esperimento corrisponde "
            "ai filtri selezionati."
        )

    dataframe["experiment_label"] = (
        dataframe.apply(
            build_experiment_label,
            axis=1,
        )
    )

    # Ordine principale: MSE crescente.
    dataframe = dataframe.sort_values(
        by="test_mse",
        ascending=True,
        na_position="last",
    )

    columns_to_save = [
        "experiment_label",
        *METRIC_COLUMNS,
        "experiment_dir",
    ]

    comparison_path = (
        output_dir / "comparison.csv"
    )

    dataframe[
        columns_to_save
    ].to_csv(
        comparison_path,
        index=False,
    )

    print("\nConfronto esperimenti:\n")

    display_columns = [
        "experiment_label",
        "test_mse",
        "test_mae",
        "trainable_parameters",
        "average_epoch_time_seconds",
        "inference_ms_per_sample",
        "peak_gpu_memory_mb",
    ]

    print(
        dataframe[
            display_columns
        ].to_string(
            index=False
        )
    )

    save_bar_plot(
        dataframe=dataframe,
        metric="test_mse",
        ylabel="Test MSE",
        output_path=output_dir / "mse.png",
        lower_is_better=True,
    )

    save_bar_plot(
        dataframe=dataframe,
        metric="test_mae",
        ylabel="Test MAE",
        output_path=output_dir / "mae.png",
        lower_is_better=True,
    )

    save_bar_plot(
        dataframe=dataframe,
        metric="trainable_parameters",
        ylabel="Parametri allenabili",
        output_path=(
            output_dir / "parameters.png"
        ),
        lower_is_better=True,
    )

    save_bar_plot(
        dataframe=dataframe,
        metric="average_epoch_time_seconds",
        ylabel="Secondi medi per epoca",
        output_path=(
            output_dir
            / "average_epoch_time.png"
        ),
        lower_is_better=True,
    )

    save_bar_plot(
        dataframe=dataframe,
        metric="inference_ms_per_sample",
        ylabel="ms per campione",
        output_path=(
            output_dir
            / "inference_time.png"
        ),
        lower_is_better=True,
    )

    save_bar_plot(
        dataframe=dataframe,
        metric="peak_gpu_memory_mb",
        ylabel="Peak GPU memory (MB)",
        output_path=(
            output_dir
            / "peak_gpu_memory.png"
        ),
        lower_is_better=True,
    )

    print(
        f"\nTabella salvata in: "
        f"{comparison_path}"
    )

    print(
        f"Grafici salvati in: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()