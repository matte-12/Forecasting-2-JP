# from root just run python src/check_ablation.py

import json
import re
from pathlib import Path
import pandas as pd

def load_all_metrics(exp_dir: Path) -> pd.DataFrame:
    records = []
    for metrics_file in exp_dir.rglob("metrics.json"):
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            # Patch RegEx per recuperare chiavi mancanti dai vecchi run
            folder_name = metrics_file.parent.name
            if 'seq_len' not in data:
                s_match = re.search(r'_S(\d+)', folder_name)
                if s_match: data['seq_len'] = int(s_match.group(1))
            if 'fixed_period' not in data:
                p_match = re.search(r'_P(\d+)', folder_name)
                if p_match: data['fixed_period'] = int(p_match.group(1))
            if 'num_blocks' not in data:
                b_match = re.search(r'_B(\d+)', folder_name)
                if b_match: data['num_blocks'] = int(b_match.group(1))
                
            records.append(data)
        except Exception:
            pass
    return pd.DataFrame(records)

def main():
    exp_dir = Path("experiments")
    if not exp_dir.exists():
        print("Cartella experiments non trovata.")
        return

    df = load_all_metrics(exp_dir)
    if df.empty:
        print("Nessun dato caricato.")
        return

    for col in ['seq_len', 'pred_len', 'fixed_period', 'num_blocks', 'test_mse']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print("\n" + "="*80)
    print("🔍 DOUBLE CHECK: ABLAZIONE FIXED PERIOD (seq_len=96)")
    print("="*80)

    fp_df = df[(df['model'] == 'FixedPeriodInception') & (df['seq_len'] == 96)].copy()

    if fp_df.empty:
        print("Nessun dato trovato per FixedPeriodInception con seq_len=96.")
        return

    for dataset in fp_df['dataset'].dropna().unique():
        print(f"\n📊 DATASET: {dataset.upper()}")
        subset = fp_df[fp_df['dataset'] == dataset]
        
        pivot = pd.pivot_table(
            subset,
            values=['test_mse'],
            index=['pred_len', 'num_blocks'],
            columns=['fixed_period'],
            aggfunc='mean'
        )
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

if __name__ == "__main__":
    main()