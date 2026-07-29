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
            folder_name = metrics_file.parent.name
            if 'seq_len' not in data:
                s_match = re.search(r'_S(\d+)', folder_name)
                if s_match: data['seq_len'] = int(s_match.group(1))
            if 'pred_len' not in data:
                h_match = re.search(r'_H(\d+)', folder_name)
                if h_match: data['pred_len'] = int(h_match.group(1))
            records.append(data)
        except Exception:
            pass
    return pd.DataFrame(records)

def main():
    exp_dir = Path("experiments")
    df = load_all_metrics(exp_dir)
    
    bb_models = [
        'LightTimesNet_MultiScale', 
        'LightTimesNet_Depthwise', 
        'LightTimesNet_Group', 
        'LightTimesNet_SingleKernel'
    ]
    
    # filtra per i backbone a parità di contesto (Seq=96)
    df_bb = df[(df['model'].isin(bb_models)) & (df['seq_len'] == 96)].copy()
    
    for col in ['test_mse', 'average_epoch_time_seconds', 'total_training_time_seconds']:
        df_bb[col] = pd.to_numeric(df_bb[col], errors='coerce')

    print("\n" + "="*80)
    print("🔍 ANALISI COORDINATE PARETO (Tempo vs MSE)")
    print("="*80)

    for dataset in df_bb['dataset'].dropna().unique():
        for pred_len in df_bb['pred_len'].dropna().unique():
            subset = df_bb[(df_bb['dataset'] == dataset) & (df_bb['pred_len'] == pred_len)]
            if subset.empty: continue
            
            print(f"\n📊 DATASET: {dataset.upper()} | PRED_LEN: {int(pred_len)}")
            print(f"{'Modello':<30} | {'Tempo/Epoca (X)':<15} | {'Test MSE (Y)':<15}")
            print("-" * 65)
            
            for _, row in subset.iterrows():
                # Fallback se average_epoch_time non è presente
                x_val = row.get('average_epoch_time_seconds')
                if pd.isna(x_val):
                    x_val = row.get('total_training_time_seconds', 0) / 20.0
                
                y_val = row['test_mse']
                m_name = row['model'].replace('LightTimesNet_', '')
                print(f"{m_name:<30} | {x_val:<15.4f} | {y_val:<15.4f}")

if __name__ == "__main__":
    main()