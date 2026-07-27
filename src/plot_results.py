import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_all_metrics(exp_dir: Path) -> pd.DataFrame:
    records = []
    for metrics_file in exp_dir.rglob("metrics.json"):
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            records.append(data)
    return pd.DataFrame(records)

def plot_experiments(df: pd.DataFrame, dataset="ETTh1", pred_len=96):
    # Setup stile accademico
    plt.rcParams.update({'font.family': 'serif', 'axes.grid': True, 'grid.alpha': 0.3})
    
    # Filtro globale per coerenza del confronto
    df_base = df[(df['dataset'] == dataset) & (df['pred_len'] == pred_len)].copy()
    if df_base.empty:
        print(f"Nessun dato trovato per {dataset} con pred_len={pred_len}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # =================================================================
    # PLOT 1: Sequenza Temporale (Scalabilità Contesto)
    # =================================================================
    ax = axes[0]
    # Filtriamo i modelli mantenendo il period fisico (24) e num_blocks=1 per coerenza
    m1d = df_base[df_base['model'].isin(['DLinear', 'CausalTCN'])]
    m2d = df_base[(df_base['model'] == 'FixedPeriodInception') & (df_base['fixed_period'] == 24) & (df_base['num_blocks'] == 1)]
    
    plot_df = pd.concat([m1d, m2d]).sort_values('seq_len')
    
    for model in ['DLinear', 'CausalTCN', 'FixedPeriodInception']:
        subset = plot_df[plot_df['model'] == model]
        if not subset.empty:
            ax.plot(subset['seq_len'], subset['test_mse'], marker='o', label=model)
    
    ax.set_title('A. Temporal Context Scalability (seq_len)')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Test MSE')
    ax.set_xticks([96, 192, 384])
    ax.legend()

    # =================================================================
    # PLOT 2: Ablazione Estrazione Stocastica (TimesNet)
    # =================================================================
    ax = axes[1]
    tn_df = df_base[(df_base['model'] == 'TimesNetOriginal') & (df_base['seq_len'] == 96)].sort_values('top_k')
    
    for b in tn_df['num_blocks'].dropna().unique():
        subset = tn_df[tn_df['num_blocks'] == b]
        if not subset.empty:
            ax.plot(subset['top_k'], subset['test_mse'], marker='s', label=f'{b} Blocks')

    ax.set_title('B. FFT Extraction Cost (TimesNet)')
    ax.set_xlabel('Top K Frequencies')
    ax.set_ylabel('Test MSE')
    ax.set_xticks([1, 2, 3])
    ax.legend(title='Depth')

    # =================================================================
    # PLOT 3: Iniezione del Dominio Fisico (FixedPeriod)
    # =================================================================
    ax = axes[2]
    fp_df = df_base[(df_base['model'] == 'FixedPeriodInception') & (df_base['seq_len'] == 96) & (df_base['num_blocks'] == 1)].sort_values('fixed_period')
    
    if not fp_df.empty:
        bars = ax.bar([str(int(p)) for p in fp_df['fixed_period']], fp_df['test_mse'], color='#4C72B0')
        ax.set_title('C. Domain Knowledge Injection')
        ax.set_xlabel('Forced Period')
        ax.set_ylabel('Test MSE')
        # Annotazioni per chiarezza sui drop
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f'{yval:.3f}', ha='center')

    # =================================================================
    # PLOT 4: Pareto Front Efficienza Spaziale (Backbone)
    # =================================================================
    ax = axes[3]
    bb_models = ['LightTimesNet_MultiScale', 'LightTimesNet_Depthwise', 'LightTimesNet_Group', 'LightTimesNet_SingleKernel']
    bb_df = df_base[(df_base['model'].isin(bb_models)) & (df_base['seq_len'] == 96)]
    
    if not bb_df.empty:
        for idx, row in bb_df.iterrows():
            ax.scatter(row['average_epoch_time_seconds'], row['test_mse'], label=row['model'].replace('LightTimesNet_', ''), s=100)
        
        ax.set_title('D. Spatial Backbone Efficiency')
        ax.set_xlabel('Avg Epoch Time (s)')
        ax.set_ylabel('Test MSE')
        ax.legend()

    plt.tight_layout()
    save_path = f"ablation_analysis_{dataset}_H{pred_len}.pdf"
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"Analisi vettoriale salvata in {save_path}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    exp_dir = project_root / "experiments"
    
    df_metrics = load_all_metrics(exp_dir)
    if not df_metrics.empty:
        # Puoi richiamare la funzione anche per l'altro dataset
        plot_experiments(df_metrics, dataset="ETTh1", pred_len=96)
        plot_experiments(df_metrics, dataset="electricity", pred_len=96)