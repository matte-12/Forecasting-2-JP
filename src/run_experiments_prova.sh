#!/usr/bin/env bash

# ============================================================
# PIPELINE COMPLETA DI TRAINING
#
# Esecuzione:
#
#   bash run_experiments_prova.sh
#
# oppure:
#
#   chmod +x run_experiments_prova.sh
#   ./run_experiments_prova.sh
#
# In Colab:
#
#   !bash run_experiments_prova.sh
# ============================================================

set -uo pipefail


# ============================================================
# ROOT DEL PROGETTO
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"


# ============================================================
# PERCORSI
# ============================================================
#
# Se EXPERIMENTS_ROOT non è impostata:
#     <repository>/experiments
#
# Se DATA_ROOT non è impostata:
#     <repository>/data
# ============================================================

export EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-$PROJECT_ROOT/experiments}"

export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"


# ============================================================
# MACROCARTELLE
# ============================================================

RUNNERS_OUTPUT_DIR="$EXPERIMENTS_ROOT"

LOG_DIR="$EXPERIMENTS_ROOT/logs"

mkdir -p "$EXPERIMENTS_ROOT"
mkdir -p "$LOG_DIR"

# ============================================================
# LOG
# ============================================================

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

LOG_FILE="$LOG_DIR/run_experiments_prova_${TIMESTAMP}.log"


exec > >(
    tee -a "$LOG_FILE"
) 2>&1


# ============================================================
# TIMER
# ============================================================

START_TIME="$(date +%s)"


print_elapsed_time() {
    END_TIME="$(date +%s)"

    ELAPSED=$((END_TIME - START_TIME))

    HOURS=$((ELAPSED / 3600))

    MINUTES=$((ELAPSED % 3600) / 60)

    SECONDS=$((ELAPSED % 60))

    echo
    echo "============================================================"
    echo "Tempo totale: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "Log: $LOG_FILE"
    echo "============================================================"
}


trap print_elapsed_time EXIT


# ============================================================
# GRUPPI FALLITI
# ============================================================

FAILED_GROUPS=()


# ============================================================
# FUNZIONE PER ESEGUIRE UN RUNNER
# ============================================================

run_runner() {
    DESCRIPTION="$1"

    MODULE="$2"

    OUTPUT_DIR="$3"

    echo
    echo "============================================================"
    echo "$DESCRIPTION"
    echo "Modulo: $MODULE"
    echo "Output: $OUTPUT_DIR"
    echo "============================================================"

    if EXPERIMENTS_DIR="$OUTPUT_DIR" \
        python -m "$MODULE"
    then
        echo
        echo "OK: $DESCRIPTION"
    else
        echo
        echo "ERRORE: $DESCRIPTION"

        FAILED_GROUPS+=(
            "$DESCRIPTION"
        )
    fi
}


# ============================================================
# INFORMAZIONI INIZIALI
# ============================================================

echo "============================================================"
echo "PIPELINE ESPERIMENTI"
echo "============================================================"

echo "Project root: $PROJECT_ROOT"

echo "Data root: $DATA_ROOT"

echo "Experiments root: $EXPERIMENTS_ROOT"

echo "Log: $LOG_FILE"


# ============================================================
# TRAINING
# ============================================================

run_runner \
    "Confronto cicli" \
    "src.runners.run_cicli" \
    "$CICLI_DIR"


run_runner \
    "Analisi TimesBlock" \
    "src.runners.run_times_block" \
    "$TIMES_BLOCK_DIR"


run_runner \
    "Confronto backbone 2D" \
    "src.runners.run_backbone2d" \
    "$BACKBONE2D_DIR"


run_runner \
    "Fixed period 17 vs 24 vs 48" \
    "src.runners.run_fixed_period" \
    "$FIXED_PERIOD_DIR"


# ============================================================
# CONFRONTI
# ============================================================
#
# Quando avrai creato:
#
# src/comparison/compare_all.py
#
# puoi impostare:
#
# RUN_COMPARISONS=1 bash run_experiments_prova.sh
# ============================================================

RUN_COMPARISONS="${RUN_COMPARISONS:-0}"


if [ "$RUN_COMPARISONS" = "1" ]; then
    echo
    echo "============================================================"
    echo "CONFRONTO RISULTATI"
    echo "============================================================"

    if python -m src.comparison.compare_all
    then
        echo "Confronti completati."
    else
        echo "Errore durante i confronti."

        FAILED_GROUPS+=(
            "Confronto risultati"
        )
    fi
else
    echo
    echo "Confronti non eseguiti."
    echo "RUN_COMPARISONS=$RUN_COMPARISONS"
fi


# ============================================================
# RIEPILOGO FINALE
# ============================================================

echo
echo "============================================================"
echo "PIPELINE TERMINATA"
echo "============================================================"


if [ "${#FAILED_GROUPS[@]}" -gt 0 ]; then
    echo
    echo "Gruppi falliti:"

    for group in "${FAILED_GROUPS[@]}"; do
        echo "- $group"
    done

    exit 1
fi


echo
echo "Tutti i runner sono terminati correttamente."

echo
echo "Risultati salvati in:"

echo "$EXPERIMENTS_ROOT"