import json
import subprocess
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parent
    plan_path = project_root / "experiments_plan.json"
    
    if not plan_path.exists():
        raise FileNotFoundError(f"File {plan_path} non trovato!")
        
    with plan_path.open("r", encoding="utf-8") as f:
        plan = json.load(f)
        
    configs = plan.get("configs", [])
    models = plan.get("models", [])
    periods = plan.get("overrides", {}).get("periods", [24])
    top_k_values = plan.get("overrides", {}).get("top_k_values", [3])
    
    print(f"🚀 Avvio Pipeline. Configurazioni: {len(configs)}, Modelli: {len(models)}")
    
    failed_runs = []
    
    for config in configs:
        for model in models:
            for period in periods:
                for top_k in top_k_values:
                    
                    # Costruzione del comando base
                    cmd = [
                        "python", "-m", "src.train4",
                        "--config", config,
                        "--model", model,
                        "--override-period", str(period),
                        "--override-top-k", str(top_k)
                    ]
                    
                    print("\n" + "="*80)
                    print(f"⚡ ESECUZIONE: {' '.join(cmd)}")
                    print("="*80)
                    
                    # Esecuzione subprocess sincronizzata
                    result = subprocess.run(cmd, cwd=str(project_root))
                    
                    if result.returncode != 0:
                        print(f"❌ ERRORE CRITICO in {model} su {config}")
                        failed_runs.append(' '.join(cmd))
                        
    print("\n" + "="*80)
    print("✅ PIPELINE COMPLETATA")
    if failed_runs:
        print("I seguenti task hanno riportato errori:")
        for r in failed_runs:
            print(f" - {r}")
    print("="*80)

if __name__ == "__main__":
    main()