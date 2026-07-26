#!/bin/bash

# per windows ti tocca farti un file .bat credo
# script eseguibile dopo: chmod +x run_experiments.sh
# eseguirlo con: ./run_experiments.sh

CONFIGS=("etth1_24" "etth1_48" "etth1_96" "ettm1_24" "ettm1_48" "ettm1_96")
MODELS=("DLinear" "CausalTCN" "TimesNet" "LightTimesNet_Single" "LightTimesNet_Multi")

export PYTHONPATH=$(pwd)

# Timer globale
START_TIME=$(date +%s)

print_elapsed_time() {
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    SECONDS=$((ELAPSED % 60))

    echo
    echo "=========================================================="
    echo " Tempo totale trascorso: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo "=========================================================="
}

# Viene eseguita sempre, sia in caso di successo che di errore/interruzione
trap print_elapsed_time EXIT

echo "Inizio validazione incrociata..."

for config in "${CONFIGS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "=========================================================="
        echo " RUNNING: Modello=$model | Config=$config"
        echo "=========================================================="

        python src/train3.py --config "$config" --model "$model"

        if [ $? -ne 0 ]; then
            echo " ERRORE critico in $model su $config. Interruzione pipeline."
            exit 1
        fi
    done
done

echo " OK - Tutti gli esperimenti sono stati completati con successo."