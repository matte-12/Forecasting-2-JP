#!/usr/bin/env bash

set -uo pipefail


# ============================================================
# INDIVIDUAZIONE DELLA ROOT DEL PROGETTO
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Priorità:
#
# 1. PROJECT_ROOT già impostata nell'ambiente;
# 2. directory corrente, se contiene src/ e configs/;
# 3. directory in cui si trova questo script.

if [ -n "${PROJECT_ROOT:-}" ]; then
    PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
elif [ -f "$PWD/src/train_prova.py" ] && [ -d "$PWD/configs" ]; then
    PROJECT_ROOT="$(pwd)"
elif [ -f "$SCRIPT_DIR/src/train_prova.py" ] && [ -d "$SCRIPT_DIR/configs" ]; then
    PROJECT_ROOT="$SCRIPT_DIR"
else
    echo "ERRORE: impossibile individuare la root del progetto."
    echo
    echo "La root deve contenere:"
    echo "  - src/train_prova.py"
    echo "  - configs/"
    echo
    echo "Directory corrente:"
    echo "  $PWD"
    echo
    echo "Directory dello script:"
    echo "  $SCRIPT_DIR"
    echo
    echo "Puoi specificarla manualmente con:"
    echo
    echo "  PROJECT_ROOT=/percorso/repository \\"
    echo "  bash run_experiments_prova_2.sh --dry-run"
    exit 1
fi

cd "$PROJECT_ROOT"

export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"


# ============================================================
# PERCORSI
# ============================================================

export DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data}"
export EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-$PROJECT_ROOT/experiments}"

MANIFEST_PATH="${MANIFEST_PATH:-$PROJECT_ROOT/src/runners/experiments_manifest.json}"

mkdir -p "$EXPERIMENTS_ROOT"

LOG_DIR="$EXPERIMENTS_ROOT/logs"
mkdir -p "$LOG_DIR"


# ============================================================
# VALIDAZIONE DEI FILE
# ============================================================

if [ ! -f "$PROJECT_ROOT/src/train_prova.py" ]; then
    echo "ERRORE: file non trovato:"
    echo "  $PROJECT_ROOT/src/train_prova.py"
    exit 1
fi

if [ ! -f "$PROJECT_ROOT/src/runners/run_manifest.py" ]; then
    echo "ERRORE: file non trovato:"
    echo "  $PROJECT_ROOT/src/runners/run_manifest.py"
    exit 1
fi

if [ ! -f "$MANIFEST_PATH" ]; then
    echo "ERRORE: manifest non trovato:"
    echo "  $MANIFEST_PATH"
    exit 1
fi

if [ ! -d "$PROJECT_ROOT/configs" ]; then
    echo "ERRORE: cartella configs non trovata:"
    echo "  $PROJECT_ROOT/configs"
    exit 1
fi


# ============================================================
# LOG
# ============================================================

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

LOG_FILE="$LOG_DIR/run_manifest_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1


# ============================================================
# TIMER
# ============================================================

START_TIME="$(date +%s)"

print_elapsed_time() {
    END_TIME="$(date +%s)"

    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    SECONDS=$((ELAPSED % 60))

    echo
    echo "============================================================"
    echo "Tempo totale: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "Log: $LOG_FILE"
    echo "============================================================"
}

trap print_elapsed_time EXIT


# ============================================================
# INTERPRETE PYTHON
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERRORE: interprete Python non trovato:"
    echo "  $PYTHON_BIN"
    exit 1
fi


# ============================================================
# INFORMAZIONI SULLA PIPELINE
# ============================================================

echo "============================================================"
echo "PIPELINE ESPERIMENTI"
echo "============================================================"

echo "Project root:      $PROJECT_ROOT"
echo "Script directory:  $SCRIPT_DIR"
echo "Data root:         $DATA_ROOT"
echo "Experiments root:  $EXPERIMENTS_ROOT"
echo "Manifest:          $MANIFEST_PATH"
echo "Python:            $PYTHON_BIN"
echo "Log:               $LOG_FILE"

echo
echo "Argomenti inoltrati a run_manifest:"
echo "  $*"

echo
echo "Controllo struttura repository:"

echo "  train_prova.py:"
echo "    $PROJECT_ROOT/src/train_prova.py"

echo "  run_manifest.py:"
echo "    $PROJECT_ROOT/src/runners/run_manifest.py"

echo "  manifest:"
echo "    $MANIFEST_PATH"

echo "  configs:"
echo "    $PROJECT_ROOT/configs"


# ============================================================
# ESECUZIONE DEL MANIFEST
# ============================================================

echo
echo "============================================================"
echo "AVVIO RUN MANIFEST"
echo "============================================================"

"$PYTHON_BIN" -m src.runners.run_manifest \
    --manifest "$MANIFEST_PATH" \
    "$@"

RETURN_CODE=$?


# ============================================================
# CONFRONTO RISULTATI OPZIONALE
# ============================================================

RUN_COMPARISONS="${RUN_COMPARISONS:-0}"

if [ "$RETURN_CODE" -eq 0 ] && [ "$RUN_COMPARISONS" = "1" ]; then
    echo
    echo "============================================================"
    echo "CONFRONTO RISULTATI"
    echo "============================================================"

    if [ -f "$PROJECT_ROOT/src/comparison/compare_all.py" ]; then
        "$PYTHON_BIN" -m src.comparison.compare_all
        RETURN_CODE=$?
    else
        echo "ERRORE: confronto richiesto, ma il file non esiste:"
        echo "  $PROJECT_ROOT/src/comparison/compare_all.py"

        RETURN_CODE=1
    fi
fi


# ============================================================
# RISULTATO FINALE
# ============================================================

echo
echo "============================================================"

if [ "$RETURN_CODE" -eq 0 ]; then
    echo "Pipeline terminata correttamente."
else
    echo "Pipeline terminata con errori."
    echo "Codice di uscita: $RETURN_CODE"
fi

echo "============================================================"

exit "$RETURN_CODE"