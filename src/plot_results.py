import json
import re
from pathlib import Path
import pandas as pd
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
            pass
            
    return pd.DataFrame(records)


def export_excel_tables(df: pd.DataFrame):
    """
    Genera 4 file CSV che riproducono esattamente l'alberatura del file tabelle_report.xlsx.
    Genera inoltre il plot vettoriale (PDF) della tabella di ablazione del periodo.
    """
    print("\n" + "="*80)
    print("📊 ESPORTAZIONE TABELLE REPORT (CSV e PDF)")
    print("="*80)
    
    # Assicuriamo che le colonne di aggregazione siano rigorosamente numeriche
    numeric_cols = ['seq_len', 'pred_len', 'fixed_period', 'top_k', 'num_blocks', 
                    'test_mse', 'test_mae', 'test_mase', 'trainable_parameters', 
                    'checkpoint_size_mb', 'best_epoch', 'time_to_best_epoch_seconds', 
                    'average_epoch_time_seconds', 'total_training_time_seconds', 
                    'inference_ms_per_sample']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # -------------------------------------------------------------------------
    # TABELLA 1: cicli (Impatto della seq_len sui modelli base)
    # -------------------------------------------------------------------------
    try:
        cond_1d = df['model'].isin(['DLinear', 'CausalTCN'])
        cond_fp = (df['model'] == 'FixedPeriodInception') & (df['fixed_period'] == 24) & (df['num_blocks'] == 1)
        t1_df = df[cond_1d | cond_fp].copy()
        
        pt1 = pd.pivot_table(t1_df, values=['test_mse', 'test_mae'], index=['dataset', 'pred_len'], columns=['seq_len', 'model'], aggfunc='first')
        
        if not pt1.empty:
            avg1 = pt1.groupby(level='dataset').mean()
            avg1.index = pd.MultiIndex.from_product([avg1.index, ['Avg']])
            pt1 = pd.concat([pt1, avg1]).sort_index(level=0)
            pt1.to_csv("table1_cicli.csv")
            print("✅ table1_cicli.csv esportata.")
    except Exception as e:
        print(f"⚠️ Errore Tabella 1: {e}")

    # -------------------------------------------------------------------------
    # TABELLA 2: period_sensitivity (Ablazione Dominio)
    # -------------------------------------------------------------------------
    try:
        t2_df = df[(df['model'] == 'FixedPeriodInception') & (df['seq_len'] == 96) & (df['num_blocks'] == 1)].copy()
        pt2 = pd.pivot_table(t2_df, values=['test_mse', 'test_mae'], index=['dataset', 'pred_len'], columns=['fixed_period'], aggfunc='first')
        
        if not pt2.empty:
            avg2 = pt2.groupby(level='dataset').mean()
            avg2.index = pd.MultiIndex.from_product([avg2.index, ['Avg']])
            pt2 = pd.concat([pt2, avg2]).sort_index(level=0)
            pt2.to_csv("table2_period_sensitivity.csv")
            print("✅ table2_period_sensitivity.csv esportata.")
    except Exception as e:
        print(f"⚠️ Errore Tabella 2: {e}")

    # -------------------------------------------------------------------------
    # TABELLA 3: times_block_x_frequency (TimesNet vs FixedPeriod - Depth Analysis)
    # -------------------------------------------------------------------------
    try:
        df_t3_tn = df[(df['model'] == 'TimesNetOriginal') & (df['seq_len'] == 96)].copy()
        df_t3_fp = df[(df['model'] == 'FixedPeriodInception') & (df['seq_len'] == 96) & (df['fixed_period'] == 24)].copy()
        df_t3_fp['top_k'] = 'N/A' # Allineamento index
        
        t3_df = pd.concat([df_t3_tn, df_t3_fp])
        pt3 = pd.pivot_table(t3_df, values=['test_mse', 'test_mae'], index=['dataset', 'model', 'top_k'], columns=['pred_len', 'num_blocks'], aggfunc='first')
        
        if not pt3.empty:
            pt3.to_csv("table3_times_block_x_frequency.csv")
            print("✅ table3_times_block_x_frequency.csv esportata.")
    except Exception as e:
        print(f"⚠️ Errore Tabella 3: {e}")

    # -------------------------------------------------------------------------
    # TABELLA 4: backbone_efficiency (Metriche computazionali)
    # -------------------------------------------------------------------------
    try:
        bb_models = ['LightTimesNet_MultiScale', 'LightTimesNet_Depthwise', 'LightTimesNet_Group', 'LightTimesNet_SingleKernel']
        # Per TimesNet prendiamo la best configurazione per un confronto onesto
        cond_tn_t4 = (df['model'] == 'TimesNetOriginal') & (df['top_k'] == 2) & (df['num_blocks'] == 2)
        cond_bb_t4 = df['model'].isin(bb_models)
        
        t4_df = df[(df['seq_len'] == 96) & (cond_tn_t4 | cond_bb_t4)].copy()
        
        rename_dict = {
            'checkpoint_size_mb': 'Checkpoint MB',
            'trainable_parameters': '#Parameters',
            'best_epoch': 'Best epoch',
            'time_to_best_epoch_seconds': 'Time to best',
            'average_epoch_time_seconds': 'Average epoch time',
            'total_training_time_seconds': 'Total training time s',
            'inference_ms_per_sample': 'inference ms x sample'
        }
        t4_df = t4_df.rename(columns=rename_dict)
        
        cols = ['test_mse', 'test_mae', 'test_mase', '#Parameters', 'Checkpoint MB', 'Best epoch', 'Time to best', 'Average epoch time', 'Total training time s', 'inference ms x sample']
        cols = [c for c in cols if c in t4_df.columns]
        
        pt4 = pd.pivot_table(t4_df, values=cols, index=['dataset', 'model', 'pred_len'], aggfunc='first')
        if not pt4.empty:
            pt4 = pt4[cols] # Riordino
            pt4.to_csv("table4_backbone_efficiency.csv")
            print("✅ table4_backbone_efficiency.csv esportata.")
    except Exception as e:
        print(f"⚠️ Errore Tabella 4: {e}")

    # -------------------------------------------------------------------------
    # PLOT TABELLA IMMAGINE (Ablazione Fixed Period Vettoriale)
    # -------------------------------------------------------------------------
    try:
        df_img = df[(df['model'] == 'FixedPeriodInception') & (df['seq_len'] == 96)].copy()
        if not df_img.empty:
            pt_img = pd.pivot_table(df_img, values='test_mse', index=['dataset', 'pred_len', 'num_blocks'], columns=['fixed_period'], aggfunc='first')
            pt_img = pt_img.sort_index(level=['dataset', 'pred_len', 'num_blocks'])
            
            fig, ax = plt.subplots(figsize=(10, len(pt_img) * 0.35 + 1.5))
            ax.axis('tight')
            ax.axis('off')
            
            # Formattazione per la resa su Matplotlib
            formatted_tab = pt_img.map(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
            row_labels = [f"{str(d).upper()} | H={int(h)} | B={int(b)}" for d, h, b in formatted_tab.index]
            col_labels = [f"P={int(c)}" for c in formatted_tab.columns]
            
            table_plot = ax.table(
                cellText=formatted_tab.values,
                rowLabels=row_labels,
                colLabels=col_labels,
                loc='center',
                cellLoc='center'
            )
            table_plot.auto_set_font_size(False)
            table_plot.set_fontsize(10)
            table_plot.scale(1.0, 1.8)
            
            # Formattazione estetica header (grigio per emulare l'immagine fornita)
            for (row, col), cell in table_plot.get_celld().items():
                if row == 0 or col == -1:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#f0f0f0')
                    
            plt.title("ABLAZIONE FIXED PERIOD (seq_len=96)\nTest MSE", fontweight='bold', fontsize=14, pad=15)
            plt.savefig("plot_visual_table_ablation.pdf", bbox_inches='tight', dpi=300)
            print("✅ plot_visual_table_ablation.pdf esportato (Pronto per LaTeX).")
    except Exception as e:
        print(f"⚠️ Errore Plot Tabella Immagine: {e}")


def plot_experiments(df: pd.DataFrame, dataset="ETTh1", pred_len=96):
    df_base = df[(df['dataset'] == dataset) & (df['pred_len'] == pred_len)].copy()
    if df_base.empty: return

    # Assicuriamoci che i tipi siano numerici
    for col in ['seq_len', 'fixed_period', 'top_k', 'num_blocks', 'test_mse']:
        if col in df_base.columns:
            df_base[col] = pd.to_numeric(df_base[col], errors='coerce')

    plt.rcParams.update({'font.family': 'serif', 'axes.grid': True, 'grid.alpha': 0.3})
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    axes = axes.flatten()

    # -------------------------------------------------------------------------
    # PLOT A: Sequenza Temporale (Scalabilità Contesto)
    # -------------------------------------------------------------------------
    m1d = df_base[df_base['model'].isin(['DLinear', 'CausalTCN'])]
    m2d = df_base[(df_base['model'] == 'FixedPeriodInception') & (df_base['fixed_period'] == 24) & (df_base['num_blocks'] == 1)]
    
    plot_df = pd.concat([m1d, m2d]).sort_values('seq_len')
    for model in ['DLinear', 'CausalTCN', 'FixedPeriodInception']:
        subset = plot_df[plot_df['model'] == model]
        if not subset.empty:
            axes[0].plot(subset['seq_len'], subset['test_mse'], marker='o', linewidth=2, label=model)
    
    axes[0].set_title('A. Temporal Context Scalability (seq_len)', fontweight='bold')
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Test MSE')
    axes[0].set_xticks([96, 192, 384])
    axes[0].legend()

    # -------------------------------------------------------------------------
    # PLOT B: Ablazione Estrazione Stocastica (TimesNet)
    # -------------------------------------------------------------------------
    tn_df = df_base[(df_base['model'] == 'TimesNetOriginal') & (df_base['seq_len'] == 96)].sort_values('top_k')
    if not tn_df.empty:
        for b in sorted(tn_df['num_blocks'].dropna().unique()):
            subset = tn_df[tn_df['num_blocks'] == b]
            if not subset.empty:
                axes[1].plot(subset['top_k'], subset['test_mse'], marker='s', linewidth=2, label=f'{int(b)} Blocks')

        axes[1].set_title('B. FFT Extraction Cost (TimesNet)', fontweight='bold')
        axes[1].set_xlabel('Top K Frequencies')
        axes[1].set_ylabel('Test MSE')
        axes[1].set_xticks(sorted(tn_df['top_k'].dropna().unique()))
        axes[1].legend(title='Depth')

    # -------------------------------------------------------------------------
    # PLOT C: Iniezione del Dominio Fisico (FixedPeriod)
    # -------------------------------------------------------------------------
    fp_df = df_base[(df_base['model'] == 'FixedPeriodInception') & (df_base['seq_len'] == 96) & (df_base['num_blocks'] == 1)].sort_values('fixed_period')
    if not fp_df.empty:
        bars = axes[2].bar([str(int(p)) for p in fp_df['fixed_period'].dropna()], fp_df['test_mse'], color='#4C72B0', edgecolor='black')
        axes[2].set_title('C. Domain Knowledge Injection', fontweight='bold')
        axes[2].set_xlabel('Forced Period')
        axes[2].set_ylabel('Test MSE')
        
        axes[2].set_ylim(0, fp_df['test_mse'].max() * 1.2)
        for bar in bars:
            yval = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width()/2, yval + (fp_df['test_mse'].max() * 0.02), f'{yval:.3f}', ha='center', fontweight='bold')

    # -------------------------------------------------------------------------
    # PLOT D: Pareto Front Efficienza Spaziale (Backbone)
    # -------------------------------------------------------------------------
    bb_models = ['LightTimesNet_MultiScale', 'LightTimesNet_Depthwise', 'LightTimesNet_Group', 'LightTimesNet_SingleKernel']
    bb_df = df_base[(df_base['model'].isin(bb_models)) & (df_base['seq_len'] == 96)]
    
    if not bb_df.empty:
        markers = ['*', 'o', 's', 'X']
        for idx, row in bb_df.iterrows():
            x_val = row.get('average_epoch_time_seconds')
            if pd.isna(x_val):
                x_val = row.get('total_training_time_seconds', 0) / 20.0
                
            model_clean = row['model'].replace('LightTimesNet_', '')
            marker = markers[bb_models.index(row['model'])]
            axes[3].scatter(x_val, row['test_mse'], label=model_clean, marker=marker, s=200, edgecolors='black')
        
        axes[3].set_title('D. Spatial Backbone Efficiency', fontweight='bold')
        axes[3].set_xlabel('Average Epoch Time (s)')
        axes[3].set_ylabel('Test MSE')
        axes[3].legend()

    plt.tight_layout()
    save_path = f"ablation_analysis_{dataset}_H{pred_len}.pdf"
    plt.savefig(save_path, dpi=300, format='pdf')
    print(f"✅ Documento PDF Vettoriale 2x2 salvato: {save_path}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    exp_dir = project_root / "experiments"
    
    df_metrics = load_all_metrics(exp_dir)
    if not df_metrics.empty:
        # Genera le 4 Tabelle CSV per copiare in Excel e la Tabella Immagine PDF
        export_excel_tables(df_metrics)
        
        # Genera i 4 Grafici Tematici Vettoriali per il Paper
        plot_experiments(df_metrics, dataset="ETTh1", pred_len=96)
        plot_experiments(df_metrics, dataset="electricity", pred_len=96)