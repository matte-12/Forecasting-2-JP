import subprocess
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent
    
    # Sostituito ETTm1 con Electricity come richiesto
    configs = [
        "etth1_24", "etth1_48", "etth1_96", 
        "electricity_24", "electricity_48", "electricity_96"
    ]
    
    # Usiamo un SET per garantire matematicamente 0 doppioni
    unique_commands = set()

    # -------------------------------------------------------------------------
    # GRUPPO 1: Benchmark Lunghezza Sequenza (Scalabilità del contesto)
    # Obiettivo: Testare 1D e 2D puro all'aumentare di seq_len [96, 192, 384]
    # Vincolo applicato: seq_len > 96 solo per DLinear, CausalTCN e FixedPeriodInception
    # -------------------------------------------------------------------------
    for config in configs:
        for seq in [96, 192, 384]:
            # Baseline 1D
            unique_commands.add(("DLinear", config, seq, None, None, None))
            unique_commands.add(("CausalTCN", config, seq, None, None, None))
            
            # La vostra proposta (Ablata per scalare sulla lunghezza della sequenza)
            unique_commands.add(("FixedPeriodInception", config, seq, 24, None, 1))
            
        # Aggiungiamo la baseline di TimesNetOriginal solo con seq=96
        unique_commands.add(("TimesNetOriginal", config, 96, None, 2, 1))

    # -------------------------------------------------------------------------
    # GRUPPO 2: Ablazione TimesNet (Studio dell'estrazione stocastica FFT)
    # Obiettivo: Mostrare l'impatto di top_k e profondità
    # Fissi: seq_len = 96
    # -------------------------------------------------------------------------
    for config in configs:
        for k in [1, 2, 3]:  # k=5 eliminato per evitare rumore
            for b in [1, 2, 3]:
                unique_commands.add(("TimesNetOriginal", config, 96, None, k, b))

    # -------------------------------------------------------------------------
    # GRUPPO 3: Ablazione FixedPeriod (Studio topologico e dominio)
    # Obiettivo: Mostrare l'impatto della rigidità del periodo (17 vs 24 vs 48)
    # Fissi: seq_len = 96
    # Vincolo: Solo per FixedPeriodInception
    # -------------------------------------------------------------------------
    for config in configs:
        for p in [17, 24, 48]:
            for b in [1, 2, 3]:
                unique_commands.add(("FixedPeriodInception", config, 96, p, None, b))

    # -------------------------------------------------------------------------
    # GRUPPO 4: Ablazione Backbone Spaziale (L'efficienza dei Kernel)
    # Obiettivo: Inception vs Depthwise vs SingleKernel vs Group
    # Fissi: seq_len = 96, period = 24, blocks = 1
    # -------------------------------------------------------------------------
    backbones = [
        "LightTimesNet_MultiScale", 
        "LightTimesNet_Depthwise", 
        "LightTimesNet_Group", 
        "LightTimesNet_SingleKernel"
    ]
    for config in configs:
        for bb in backbones:
            unique_commands.add((bb, config, 96, 24, None, 1))

    # =========================================================================
    # ESECUZIONE
    # =========================================================================
    # Ordiniamo per Configurazione per mantenere log sequenziali puliti
    sorted_commands = sorted(list(unique_commands), key=lambda x: (x[1], x[0]))
    
    print(f"  INIZIO PIPELINE: {len(sorted_commands)} esperimenti univoci pronti all'esecuzione.")
    failed_runs = []

    for idx, (model, config, seq, period, top_k, blocks) in enumerate(sorted_commands):
        # Utilizzato il tuo comando -m src.train4
        cmd = [
            "python", "-m", "src.train4",
            "--config", config,
            "--model", model,
            "--override-seq-len", str(seq)
        ]
        
        if period is not None:
            cmd.extend(["--override-period", str(period)])
        if top_k is not None:
            cmd.extend(["--override-top-k", str(top_k)])
        if blocks is not None:
            cmd.extend(["--override-num-blocks", str(blocks)])

        print("\n" + "="*90)
        print(f"  RUN [{idx+1}/{len(sorted_commands)}]: {' '.join(cmd)}")
        print("="*90)
        
        res = subprocess.run(cmd, cwd=str(project_root))
        if res.returncode != 0:
            failed_runs.append(' '.join(cmd))

    print("\n" + "="*90)
    if not failed_runs:
        print("✅ PIPELINE COMPLETATA CON SUCCESSO! Zero Errori.")
    else:
        print(f"⚠️ PIPELINE COMPLETATA. Ci sono {len(failed_runs)} errori:")
        for r in failed_runs:
            print(f" - {r}")
    print("="*90)

if __name__ == "__main__":
    main()