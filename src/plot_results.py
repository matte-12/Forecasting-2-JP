import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_all_metrics(exp_dir: Path) -> pd.DataFrame:
    records = []
    files = list(exp_dir.rglob("metrics.json"))
    print(f"🔍 Trovati {len(files)} file metrics.json in {exp_dir}")
    
    for metrics_file in files:
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                
            # =================================================================
            # PATCH DINAMICA: Recupero Metadati dai Nomi delle Cartelle
            # =================================================================
            folder_name = metrics_file.parent.name
            
            s_match = re.search(r'_S(\d+)', folder_name)
            if s_match: data['seq_len'] = int(s_match.group(1))
            
            h_match = re.search(r'_H(\d+)', folder_name)
            if h_match: data['pred_len'] = int(h_match.group(1))
            
            p_match = re.search(r'_P(\d+)', folder_name)
            if p_match: data['fixed_period'] = int(p_match.group(1))
            
            k_match = re.search(r'_K(\d+)', folder_name)
            if k_match: data['top_k'] = int(k_match.group(1))
            
            b_match = re.search(r'_B(\d+)', folder_name)
            if b_match: data['num_blocks'] = int(b_match.group(1))
            
            records.append(data)
        except Exception as e:
            print(f"⚠️ Errore caricando {metrics_file}: {e}")
            
    return pd.DataFrame(records)

def plot_experiments(df: pd.DataFrame, dataset="ETTh1", pred_len=96):
    print("\n" + "="*80)
    print(f"📊 GENERAZIONE PLOT: DATASET {dataset} | ORRIZONTE {pred_len}")
    print("="*80)

    # Assicuriamoci che i tipi siano corretti dopo l'estrazione RegEx
    for col in ['seq_len', 'pred_len', 'fixed_period', 'top_k', 'num_blocks', 'test_mse']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df_base = df[(df['dataset'] == dataset) & (df['pred_len'] == pred_len)].copy()

    if df_base.empty:
        print(f"❌ NESSUN DATO TROVATO per {dataset} con pred_len={pred_len}.")
        return

    # Setup stile accademico
    plt.rcParams.update({'font.family': 'serif', 'axes.grid': True, 'grid.alpha': 0.3})
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    # -------------------------------------------------------------------------
    # PLOT A: Sequenza Temporale (Scalabilità Contesto)
    # -------------------------------------------------------------------------
    ax = axes[0]
    m1d = df_base[df_base['model'].isin(['DLinear', 'CausalTCN'])]
    m2d = df_base[
        (df_base['model'] == 'FixedPeriodInception') & 
        (df_base['fixed_period'] == 24) & 
        (df_base['num_blocks'] == 1)
    ]
    
    plot_df = pd.concat([m1d, m2d]).sort_values('seq_len')
    
    for model in ['DLinear', 'CausalTCN', 'FixedPeriodInception']:
        subset = plot_df[plot_df['model'] == model]
        if not subset.empty:
            ax.plot(subset['seq_len'], subset['test_mse'], marker='o', linewidth=2, label=model)
    
    ax.set_title('A. Temporal Context Scalability (seq_len)', fontweight='bold')
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Test MSE')
    ax.set_xticks([96, 192, 384])
    ax.legend()

    # -------------------------------------------------------------------------
    # PLOT B: Ablazione Estrazione Stocastica (TimesNet)
    # -------------------------------------------------------------------------
    ax = axes[1]
    tn_df = df_base[
        (df_base['model'] == 'TimesNetOriginal') & 
        (df_base['seq_len'] == 96)
    ].sort_values('top_k')
    
    if not tn_df.empty:
        for b in sorted(tn_df['num_blocks'].dropna().unique()):
            subset = tn_df[tn_df['num_blocks'] == b]
            if not subset.empty:
                ax.plot(subset['top_k'], subset['test_mse'], marker='s', linewidth=2, label=f'{int(b)} Blocks')

        ax.set_title('B. FFT Extraction Cost (TimesNet)', fontweight='bold')
        ax.set_xlabel('Top K Frequencies')
        ax.set_ylabel('Test MSE')
        ax.set_xticks(sorted(tn_df['top_k'].dropna().unique()))
        ax.legend(title='Depth')

    # -------------------------------------------------------------------------
    # PLOT C: Iniezione del Dominio Fisico (FixedPeriod)
    # -------------------------------------------------------------------------
    ax = axes[2]
    fp_df = df_base[
        (df_base['model'] == 'FixedPeriodInception') & 
        (df_base['seq_len'] == 96) & 
        (df_base['num_blocks'] == 1)
    ].sort_values('fixed_period')
    
    if not fp_df.empty:
        bars = ax.bar([str(int(p)) for p in fp_df['fixed_period'].dropna()], fp_df['test_mse'], color='#4C72B0', edgecolor='black')
        ax.set_title('C. Domain Knowledge Injection', fontweight='bold')
        ax.set_xlabel('Forced Period')
        ax.set_ylabel('Test MSE')
        
        # Margine superiore per le etichette
        ax.set_ylim(0, fp_df['test_mse'].max() * 1.2)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + (fp_df['test_mse'].max() * 0.02), f'{yval:.3f}', ha='center', fontweight='bold')

    # -------------------------------------------------------------------------
    # PLOT D: Pareto Front Efficienza Spaziale (Backbone)
    # -------------------------------------------------------------------------
    ax = axes[3]
    bb_models = [
        'LightTimesNet_MultiScale', 
        'LightTimesNet_Depthwise', 
        'LightTimesNet_Group', 
        'LightTimesNet_SingleKernel'
    ]
    bb_df = df_base[
        (df_base['model'].isin(bb_models)) & 
        (df_base['seq_len'] == 96)
    ]
    
    if not bb_df.empty:
        markers = ['*', 'o', 's', 'X']
        for idx, row in bb_df.iterrows():
            x_val = row.get('average_epoch_time_seconds')
            # Fallback se le vecchie metriche non hanno il tempo di epoca
            if pd.isna(x_val):
                x_val = row.get('total_training_time_seconds', 0) / 20.0
                
            model_clean = row['model'].replace('LightTimesNet_', '')
            marker = markers[bb_models.index(row['model'])]
            ax.scatter(x_val, row['test_mse'], label=model_clean, marker=marker, s=200, edgecolors='black')
        
        ax.set_title('D. Spatial Backbone Efficiency', fontweight='bold')
        ax.set_xlabel('Average Epoch Time (s)')
        ax.set_ylabel('Test MSE')
        ax.legend()

    plt.tight_layout()
    save_path = f"ablation_analysis_{dataset}_H{pred_len}.pdf"
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"✅ Documento PDF Vettoriale salvato: {save_path}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    exp_dir = project_root / "experiments"
    
    df_metrics = load_all_metrics(exp_dir)
    if not df_metrics.empty:
        plot_experiments(df_metrics, dataset="ETTh1", pred_len=96)
        plot_experiments(df_metrics, dataset="electricity", pred_len=96)