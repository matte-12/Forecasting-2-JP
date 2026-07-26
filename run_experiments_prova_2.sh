#!/usr/bin/env bash

set -uo pipefail


# ============================================================
# ROOT DEL PROGETTO
# ============================================================

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"


# ============================================================
# PERCORSI
# ============================================================

export DATA_ROOT="${
    DATA_ROOT:-$PROJECT_ROOT/data
}"

export EXPERIMENTS_ROOT="${
    EXPERIMENTS_ROOT:-$PROJECT_ROOT/experiments
}"

mkdir -p "$EXPERIMENTS_ROOT"

LOG_DIR="$EXPERIMENTS_ROOT/logs"

mkdir -p "$LOG_DIR"


# ============================================================
# LOG
# ============================================================

TIMESTAMP="$(
    date '+%Y%m%d_%H%M%S'
)"

LOG_FILE="$LOG_DIR/run_manifest_${TIMESTAMP}.log"

exec > >(
    tee -a "$LOG_FILE"
) 2>&1


# ============================================================
# TIMER
# ============================================================

START_TIME="$(
    date +%s
)"


print_elapsed_time() {
    END_TIME="$(
        date +%s
    )"

    ELAPSED=$(
        END_TIME - START_TIME
    )

    HOURS=$(
        ELAPSED / 3600
    )

    MINUTES=$(
        (ELAPSED % 3600) / 60
    )

    SECONDS=$(
        ELAPSED % 60
    )

    echo
    echo "============================================================"
    echo "Tempo totale: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "Log: $LOG_FILE"
    echo "============================================================"
}


trap print_elapsed_time EXIT


# ============================================================
# ARGOMENTI
# ============================================================
#
# Tutti gli argomenti passati allo script vengono inoltrati
# a run_manifest.py.
#
# Esempi:
#
# bash run_experiments_prova.sh
#
# bash run_experiments_prova.sh --group cicli
#
# bash run_experiments_prova.sh --group fixed_period
#
# bash run_experiments_prova.sh --dataset etth1
#
# bash run_experiments_prova.sh \
#     --group times_block \
#     --dataset electricity
#
# bash run_experiments_prova.sh --dry-run
# ============================================================

echo "============================================================"
echo "PIPELINE ESPERIMENTI"
echo "============================================================"

echo "Project root: $PROJECT_ROOT"
echo "Data root: $DATA_ROOT"
echo "Experiments root: $EXPERIMENTS_ROOT"
echo "Log: $LOG_FILE"

echo
echo "Argomenti run_manifest: $*"

python -m src.runners.run_manifest "$@"

RETURN_CODE=$?


# ============================================================
# CONFRONTI OPZIONALI
# ============================================================

RUN_COMPARISONS="${
    RUN_COMPARISONS:-0
}"

if (
    [ "$RETURN_CODE" -eq 0 ]
    && [ "$RUN_COMPARISONS" = "1" ]
); then
    echo
    echo "============================================================"
    echo "CONFRONTO RISULTATI"
    echo "============================================================"

    python -m src.comparison.compare_all

    RETURN_CODE=$?
fi


# ============================================================
# FINE
# ============================================================

if [ "$RETURN_CODE" -eq 0 ]; then
    echo
    echo "Pipeline terminata correttamente."
else
    echo
    echo "Pipeline terminata con errori."
fi

exit "$RETURN_CODE"