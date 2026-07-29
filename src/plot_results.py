import json
import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_all_metrics(exp_dir: Path) -> pd.DataFrame:
    records = []
    files = list(exp_dir.rglob("metrics.json"))
    print(f"  Trovati {len(files)} file metrics.json in {exp_dir}")
    
    for metrics_file in files:
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                
            # PATCH DINAMICA: Recupero Metadati dai Nomi
            folder_name = metrics_file.parent.name
            if 'seq_len' not in data:
                s_match = re.search(r'_S(\d+)', folder_name)
                if s_match: data['seq_len'] = int(s_match.group(1))
            if 'pred_len' not in data:
                h_match = re.search(r'_H(\d+)', folder_name)
                if h_match: data['pred_len'] = int(h_match.group(1))
            if 'fixed_period' not in data:
                p_match = re.search(r'_P(\d+)', folder_name)
                if p_match: data['fixed_period'] = int(p_match.group(1))
            if 'top_k' not in data:
                k_match = re.search(r'_K(\d+)', folder_name)
                if k_match: data['top_k'] = int(k_match.group(1))
            if 'num_blocks' not in data:
                b_match = re.search(r'_B(\d+)', folder_name)
                if b_match: data['num_blocks'] = int(b_match.group(1))
            
            records.append(data)
        except Exception as e:
            pass
            
    return pd.DataFrame(records)

def export_excel_tables(df: pd.DataFrame):
    print("\n" + "="*80)
    print("  ESPORTAZIONE TABELLE REPORT (CSV e LaTeX)")
    print("="*80)
    
    numeric_cols = ['seq_len', 'pred_len', 'fixed_period', 'top_k', 'num_blocks', 
                    'test_mse', 'test_mae', 'test_mase', 'trainable_parameters', 
                    'checkpoint_size_mb', 'best_epoch', 'time_to_best_epoch_seconds', 
                    'average_epoch_time_seconds', 'total_training_time_seconds', 
                    'inference_ms_per_sample', 'masked_features']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Inizializzazione file LaTeX
    tex_file = "tablex.tex"
    with open(tex_file, "w") as f:
        f.write("% Tabelle auto-generate per paper accademico.\n")
        f.write("\\clearpage\n\\onecolumn\n\\appendices\n\\section{Extended Experimental Results}\n\n")
        f.write("% Impostazioni globali per uniformare le dimensioni ed evitare \\resizebox\n")
        f.write("\\tiny\n\\setlength{\\tabcolsep}{1.5pt}\n\\renewcommand{\\arraystretch}{0.85}\n\n")
    # ==========================================
    # TABELLA 1: cicli
    # ==========================================
    try:
        cond_1d = df['model'].isin(['DLinear', 'CausalTCN'])
        cond_fp = (df['model'] == 'FixedPeriodInception') & (df['fixed_period'] == 24) & (df['num_blocks'] == 1)
        t1_df = df[cond_1d | cond_fp].copy()
        pt1 = pd.pivot_table(t1_df, values=['test_mae', 'test_mse'], index=['dataset', 'pred_len'], columns=['seq_len', 'model'], aggfunc='first')
        
        if not pt1.empty:
            avg1 = pt1.groupby(level='dataset').mean()
            avg1.index = pd.MultiIndex.from_product([avg1.index, ['Avg']])
            pt1 = pd.concat([pt1, avg1]).sort_index(level=0)
            pt1.to_csv("table1_cicli.csv", float_format="%.3f")
            
            with open(tex_file, "a") as f:
                f.write("\\begin{table*}[!htbp]\n\\centering\n\\caption{Cicli: Performance across different sequence lengths}\n\\label{tab:cicli}\n")
                f.write("\\begin{tabular}{ll ccc ccc ccc ccc ccc ccc}\n\\toprule\n")
                f.write("& & \\multicolumn{9}{c}{\\textbf{MAE}} & \\multicolumn{9}{c}{\\textbf{MSE}} \\\\\n")
                f.write("\\cmidrule(lr){3-11} \\cmidrule(lr){12-20}\n")
                f.write("& & \\multicolumn{3}{c}{Seq 96} & \\multicolumn{3}{c}{Seq 192} & \\multicolumn{3}{c}{Seq 384} & \\multicolumn{3}{c}{Seq 96} & \\multicolumn{3}{c}{Seq 192} & \\multicolumn{3}{c}{Seq 384} \\\\\n")
                f.write("\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-14} \\cmidrule(lr){15-17} \\cmidrule(lr){18-20}\n")
                f.write("\\textbf{Dataset} & \\textbf{Pred} & CT & DL & FPI & CT & DL & FPI & CT & DL & FPI & CT & DL & FPI & CT & DL & FPI & CT & DL & FPI \\\\\n\\midrule\n")
                
                models = ['CausalTCN', 'DLinear', 'FixedPeriodInception']
                for idx, row in pt1.iterrows():
                    dataset, pred = idx
                    ds_str = str(dataset).replace('electricity', 'elec.')
                    line = f"{ds_str} & {pred}"
                    for metric in ['test_mae', 'test_mse']:
                        for seq in [96, 192, 384]:
                            for mod in models:
                                try:
                                    val = row[(metric, seq, mod)]
                                    line += f" & {val:.3f}" if pd.notna(val) else " & -"
                                except KeyError:
                                    line += " & -"
                    f.write(line + " \\\\\n")
                
                f.write("\\bottomrule\n\\multicolumn{20}{l}{\\textit{Note: CT = CausalTCN, DL = DLinear, FPI = FixedPeriodInception}} \\\\\n\\end{tabular}\n\\end{table*}\n\n")
            print("  table1_cicli.csv ed export LaTeX completati.")
    except Exception as e:
        print(f"  Errore Tabella 1: {e}")

    # ==========================================
    # TABELLA 2: period_sensitivity
    # ==========================================
    try:
        t2_df = df[(df['model'] == 'FixedPeriodInception') & (df['seq_len'] == 96) & (df['num_blocks'] == 1)].copy()
        pt2 = pd.pivot_table(t2_df, values=['test_mae', 'test_mse'], index=['dataset', 'pred_len'], columns=['fixed_period'], aggfunc='first')
        
        if not pt2.empty:
            for metric in ['test_mae', 'test_mse']:
                num_cols = [c for c in pt2[metric].columns if isinstance(c, (int, float))]
                if num_cols: pt2[(metric, 'Avg_Periods')] = pt2[metric][num_cols].mean(axis=1)
                    
            dl_df = df[(df['model'] == 'DLinear') & (df['seq_len'] == 96)]
            if not dl_df.empty:
                dl_mse_map = dl_df.set_index(['dataset', 'pred_len'])['test_mse'].to_dict()
                dl_mae_map = dl_df.set_index(['dataset', 'pred_len'])['test_mae'].to_dict()
                pt2[('test_mse', 'DLinear')] = [dl_mse_map.get(idx, np.nan) for idx in pt2.index]
                pt2[('test_mae', 'DLinear')] = [dl_mae_map.get(idx, np.nan) for idx in pt2.index]

            avg2 = pt2.groupby(level='dataset').mean()
            avg2.index = pd.MultiIndex.from_product([avg2.index, ['Avg']])
            pt2 = pd.concat([pt2, avg2]).sort_index(level=0)
            pt2.to_csv("table2_period_sensitivity.csv", float_format="%.3f")

            with open(tex_file, "a") as f:
                f.write("\\begin{table*}[!htbp]\n\\centering\n\\caption{Period Sensitivity}\n\\label{tab:period_sensitivity}\n")
                f.write("\\begin{tabular}{ll ccccc ccccc}\n\\toprule\n")
                f.write("& & \\multicolumn{5}{c}{\\textbf{MAE}} & \\multicolumn{5}{c}{\\textbf{MSE}} \\\\\n")
                f.write("\\cmidrule(lr){3-7} \\cmidrule(lr){8-12}\n")
                f.write("\\textbf{Dataset} & \\textbf{Pred} & 17 & 24 & 48 & Avg & DLin & 17 & 24 & 48 & Avg & DLin \\\\\n\\midrule\n")
                
                periods = [17.0, 24.0, 48.0, 'Avg_Periods', 'DLinear']
                for idx, row in pt2.iterrows():
                    dataset, pred = idx
                    ds_str = str(dataset).replace('electricity', 'elec.')
                    line = f"{ds_str} & {pred}"
                    for metric in ['test_mae', 'test_mse']:
                        for p in periods:
                            try:
                                val = row[(metric, p)]
                                line += f" & {val:.3f}" if pd.notna(val) else " & -"
                            except KeyError:
                                line += " & -"
                    f.write(line + " \\\\\n")
                f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")
            print("  table2_period_sensitivity.csv ed export LaTeX completati.")
    except Exception as e:
        print(f"  Errore Tabella 2: {e}")

    # ==========================================
    # TABELLA 3: times_block_x_frequency
    # ==========================================
    try:
        df_t3_tn = df[(df['model'] == 'TimesNetOriginal') & (df['seq_len'] == 96)].copy()
        df_t3_fp = df[(df['model'] == 'FixedPeriodInception') & (df['seq_len'] == 96) & (df['fixed_period'] == 24)].copy()
        df_t3_fp['top_k'] = 'N/A' 
        
        t3_df = pd.concat([df_t3_tn, df_t3_fp])
        pt3 = pd.pivot_table(t3_df, values=['test_mae', 'test_mse'], index=['dataset', 'model', 'top_k'], columns=['pred_len', 'num_blocks'], aggfunc='first')
        
        if not pt3.empty:
            pt3.to_csv("table3_times_block_x_frequency.csv", float_format="%.3f")
            
            with open(tex_file, "a") as f:
                f.write("\\begin{table*}[!htbp]\n\\centering\n\\caption{Times Block vs Frequency (Top-$K$)}\n\\label{tab:times_block}\n")
                f.write("\\begin{tabular}{llc ccc ccc ccc ccc ccc ccc}\n\\toprule\n")
                f.write("& & & \\multicolumn{9}{c}{\\textbf{MAE}} & \\multicolumn{9}{c}{\\textbf{MSE}} \\\\\n")
                f.write("\\cmidrule(lr){4-12} \\cmidrule(lr){13-21}\n")
                f.write("& & & \\multicolumn{3}{c}{Pred 24} & \\multicolumn{3}{c}{Pred 48} & \\multicolumn{3}{c}{Pred 96} & \\multicolumn{3}{c}{Pred 24} & \\multicolumn{3}{c}{Pred 48} & \\multicolumn{3}{c}{Pred 96} \\\\\n")
                f.write("\\cmidrule(lr){4-6} \\cmidrule(lr){7-9} \\cmidrule(lr){10-12} \\cmidrule(lr){13-15} \\cmidrule(lr){16-18} \\cmidrule(lr){19-21}\n")
                f.write("\\textbf{Dataset} & \\textbf{Model} & \\textbf{Top-$K$} & 1B & 2B & 3B & 1B & 2B & 3B & 1B & 2B & 3B & 1B & 2B & 3B & 1B & 2B & 3B & 1B & 2B & 3B \\\\\n\\midrule\n")
                
                for idx, row in pt3.iterrows():
                    dataset, model, top_k = idx
                    ds_str = str(dataset).replace('electricity', 'elec.')
                    mod_str = "FPI" if model == "FixedPeriodInception" else "TN" if model == "TimesNetOriginal" else str(model)
                    k_str = "-" if str(top_k) == "N/A" else str(int(float(top_k))) if pd.notna(top_k) else "-"
                    
                    line = f"{ds_str} & {mod_str} & {k_str}"
                    for metric in ['test_mae', 'test_mse']:
                        for pred in [24, 48, 96]:
                            for b in [1.0, 2.0, 3.0]:
                                try:
                                    val = row[(metric, pred, b)]
                                    line += f" & {val:.3f}" if pd.notna(val) else " & -"
                                except KeyError:
                                    line += " & -"
                    f.write(line + " \\\\\n")
                f.write("\\bottomrule\n\\multicolumn{21}{l}{\\textit{Note: 1B, 2B, 3B refer to Num Blocks. elec. = electricity, FPI = FixedPeriodInception, TN = TimesNet}} \\\\\n\\end{tabular}\n\\end{table*}\n\n")
            print("  table3_times_block_x_frequency.csv ed export LaTeX completati.")
    except Exception as e:
        print(f"  Errore Tabella 3: {e}")

    # ==========================================
    # TABELLA 4: backbone_efficiency
    # ==========================================
    try:
        bb_models = ['LightTimesNet_MultiScale', 'LightTimesNet_Depthwise', 'LightTimesNet_Group', 'LightTimesNet_SingleKernel']
        cond_tn_t4 = (df['model'] == 'TimesNetOriginal') & (df['top_k'] == 2) & (df['num_blocks'] == 1)
        t4_df = df[(df['seq_len'] == 96) & (cond_tn_t4 | df['model'].isin(bb_models) | (df['model'] == 'DLinear'))].copy()
        
        rename_dict = {
            'checkpoint_size_mb': 'Checkpoint MB', 'trainable_parameters': '#Parameters',
            'best_epoch': 'Best epoch', 'time_to_best_epoch_seconds': 'Time to best',
            'average_epoch_time_seconds': 'Average epoch time', 'total_training_time_seconds': 'Total training time s',
            'inference_ms_per_sample': 'inference ms x sample'
        }
        t4_df = t4_df.rename(columns=rename_dict)
        cols = ['test_mse', 'test_mae', 'test_mase', '#Parameters', 'Checkpoint MB', 'Best epoch', 'Time to best', 'Average epoch time', 'Total training time s', 'inference ms x sample']
        cols = [c for c in cols if c in t4_df.columns]
        
        pt4 = pd.pivot_table(t4_df, values=cols, index=['dataset', 'model', 'pred_len'], aggfunc='first')[cols]
        
        if not pt4.empty:
            pt4.to_csv("table4_backbone_efficiency.csv", float_format="%.3f")
            
            with open(tex_file, "a") as f:
                f.write("\\begin{table*}[!htbp]\n\\centering\n\\caption{Backbone Efficiency Analysis}\n\\label{tab:backbone_efficiency}\n")
                f.write("\\begin{tabular}{llc ccc ccc ccc c}\n\\toprule\n")
                f.write("\\textbf{Dataset} & \\textbf{Model} & \\textbf{Pred} & \\textbf{MSE} & \\textbf{MAE} & \\textbf{MASE} & \\textbf{Params} & \\textbf{Ckpt (MB)} & \\textbf{Best Ep.} & \\textbf{T/Best} & \\textbf{Avg Ep.} & \\textbf{Tot. T.} & \\textbf{Inf (ms)} \\\\\n\\midrule\n")
                
                pt4_flat = pt4.reset_index()
                for _, row in pt4_flat.iterrows():
                    ds_str = str(row['dataset']).replace('electricity', 'elec.')
                    mod_str = str(row['model']).replace('LightTimesNet_', 'LTN_').replace('TimesNetOriginal', 'TN')
                    pred = int(row['pred_len']) if pd.notna(row['pred_len']) else "-"
                    
                    def fmt(val, template="{:.3f}"): return template.format(val) if pd.notna(val) else "-"
                    def fmt_int(val): return str(int(val)) if pd.notna(val) else "-"
                    
                    mse, mae, mase = fmt(row.get('test_mse')), fmt(row.get('test_mae')), fmt(row.get('test_mase'))
                    params = fmt_int(row.get('#Parameters'))
                    ckpt = fmt(row.get('Checkpoint MB'))
                    best_ep = fmt_int(row.get('Best epoch'))
                    t_best = fmt(row.get('Time to best'))
                    avg_ep = fmt(row.get('Average epoch time'))
                    tot_t = fmt(row.get('Total training time s'))
                    inf = fmt(row.get('inference ms x sample'))
                    
                    f.write(f"{ds_str} & {mod_str} & {pred} & {mse} & {mae} & {mase} & {params} & {ckpt} & {best_ep} & {t_best} & {avg_ep} & {tot_t} & {inf} \\\\\n")
                f.write("\\bottomrule\n\\multicolumn{13}{l}{\\textit{Note: LTN = LightTimesNet, TN = TimesNet}} \\\\\n\\end{tabular}\n\\end{table*}\n\n")
            print("  table4_backbone_efficiency.csv ed export LaTeX completati.")
    except Exception as e:
        print(f"  Errore Tabella 4: {e}")

    # PLOT TABELLA IMMAGINE
    try:
        df_img = df[(df['model'] == 'FixedPeriodInception') & (df['seq_len'] == 96)].copy()
        if not df_img.empty:
            pt_img = pd.pivot_table(df_img, values='test_mse', index=['dataset', 'pred_len', 'num_blocks'], columns=['fixed_period'], aggfunc='first')
            
            num_cols = [c for c in pt_img.columns if isinstance(c, (int, float))]
            pt_img['Avg_Periods'] = pt_img[num_cols].mean(axis=1)
            
            dl_df = df[(df['model'] == 'DLinear') & (df['seq_len'] == 96)]
            if not dl_df.empty:
                dl_map = dl_df.set_index(['dataset', 'pred_len'])['test_mse'].to_dict()
                pt_img['DLinear'] = [dl_map.get((d, h), np.nan) for d, h, b in pt_img.index]

            def sort_key_img(col):
                if isinstance(col, (int, float)): return (0, col)
                elif col == 'Avg_Periods': return (1, 0)
                else: return (2, 0)
            pt_img = pt_img[sorted(pt_img.columns, key=sort_key_img)]
            pt_img = pt_img.sort_index(level=['dataset', 'pred_len', 'num_blocks'])
            
            fig, ax = plt.subplots(figsize=(12, len(pt_img) * 0.35 + 1.5))
            ax.axis('tight')
            ax.axis('off')
            
            formatted_tab = pt_img.map(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")
            row_labels = [f"{str(d).upper()} | H={int(h)} | B={int(b)}" for d, h, b in formatted_tab.index]
            
            col_labels = []
            for c in formatted_tab.columns:
                if isinstance(c, (int, float)): col_labels.append(f"P={int(c)}")
                elif c == 'Avg_Periods': col_labels.append("Avg Periods")
                else: col_labels.append(str(c))
            
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
            
            for (row, col), cell in table_plot.get_celld().items():
                if row == 0 or col == -1:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#f0f0f0')
                    
            plt.title("ABLAZIONE FIXED PERIOD (seq_len=96)\nTest MSE", fontweight='bold', fontsize=14, pad=15)
            plt.savefig("plot_visual_table_ablation.pdf", bbox_inches='tight', dpi=300)
            print("  plot_visual_table_ablation.pdf esportato (Pronto per LaTeX).")
            plt.close(fig)
    except Exception as e:
        print(f"  Errore Plot Tabella Immagine: {e}")

def plot_experiments(df: pd.DataFrame, dataset="ETTh1", pred_len=96):
    df_base = df[(df['dataset'] == dataset) & (df['pred_len'] == pred_len)].copy()
    if df_base.empty: return

    for col in ['seq_len', 'fixed_period', 'top_k', 'num_blocks', 'test_mse', 'trainable_parameters']:
        if col in df_base.columns:
            df_base[col] = pd.to_numeric(df_base[col], errors='coerce')

    plt.rcParams.update({'font.family': 'serif', 'axes.grid': True, 'grid.alpha': 0.3})
    
    base_name = f"ablation_analysis_{dataset}_H{pred_len}"

    # --- PLOT A: Sequenza Temporale ---
    fig_a, ax_a = plt.subplots(figsize=(7, 3.5))
    m1d = df_base[df_base['model'].isin(['DLinear', 'CausalTCN'])]
    m2d = df_base[(df_base['model'] == 'FixedPeriodInception') & (df_base['fixed_period'] == 24) & (df_base['num_blocks'] == 1)]
    
    plot_df = pd.concat([m1d, m2d]).sort_values('seq_len')
    for model in ['DLinear', 'CausalTCN', 'FixedPeriodInception']:
        subset = plot_df[plot_df['model'] == model]
        if not subset.empty:
            ax_a.plot(subset['seq_len'], subset['test_mse'], marker='o', linewidth=2, label=model)
    
    ax_a.set_title('A. Temporal Context Scalability (seq_len)', fontweight='bold')
    ax_a.set_xlabel('Sequence Length')
    ax_a.set_ylabel('Test MSE')
    ax_a.set_xticks([96, 192, 384])
    ax_a.legend()
    fig_a.tight_layout()
    fig_a.savefig(f"{base_name}_A.png", dpi=300, bbox_inches='tight')
    plt.close(fig_a)

    # --- PLOT B: Ablazione FFT vs FixedPeriod ---
    fig_b, ax_b = plt.subplots(figsize=(7, 4.2)) # Maggiore altezza per accogliere la doppia legenda
    tn_df = df_base[(df_base['model'] == 'TimesNetOriginal') & (df_base['seq_len'] == 96)].sort_values('top_k')
    fp_df = df_base[(df_base['model'] == 'FixedPeriodInception') & (df_base['seq_len'] == 96) & (df_base['fixed_period'] == 24)]
    
    blocks = [1, 2, 3]
    colors = plt.cm.tab10.colors
    
    for i, b in enumerate(blocks):
        color = colors[i % len(colors)]
        
        if not tn_df.empty:
            subset_tn = tn_df[tn_df['num_blocks'] == b]
            if not subset_tn.empty:
                ax_b.plot(subset_tn['top_k'], subset_tn['test_mse'], marker='s', linewidth=2, color=color, label=f'TimesNet (B={int(b)})')
                
        if not fp_df.empty:
            subset_fp = fp_df[fp_df['num_blocks'] == b]
            if not subset_fp.empty:
                fp_mse = subset_fp['test_mse'].values[0]
                ax_b.axhline(y=fp_mse, linestyle='--', linewidth=2, color=color, label=f'FixedPeriod (B={int(b)})')

    ax_b.set_title('B. FFT Extraction Cost vs Fixed Baseline', fontweight='bold')
    ax_b.set_xlabel('Top K Frequencies')
    ax_b.set_ylabel('Test MSE')
    if not tn_df.empty:
        ax_b.set_xticks(sorted(tn_df['top_k'].dropna().unique()))
    ax_b.legend(ncol=2) # Formattazione compatta per risparmiare asse Y
    fig_b.tight_layout()
    fig_b.savefig(f"{base_name}_B.png", dpi=300, bbox_inches='tight')
    plt.close(fig_b)

    # --- PLOT C: Domain Knowledge ---
    fig_c, ax_c = plt.subplots(figsize=(7, 3.5))
    fp_df_c = df_base[(df_base['model'] == 'FixedPeriodInception') & (df_base['seq_len'] == 96) & (df_base['num_blocks'] == 1)].sort_values('fixed_period')
    if not fp_df_c.empty:
        bars = ax_c.bar([str(int(p)) for p in fp_df_c['fixed_period'].dropna()], fp_df_c['test_mse'], color='#4C72B0', edgecolor='black')
        ax_c.set_title('C. Domain Knowledge Injection', fontweight='bold')
        ax_c.set_xlabel('Forced Period')
        ax_c.set_ylabel('Test MSE')
        
        ax_c.set_ylim(0, fp_df_c['test_mse'].max() * 1.2)
        for bar in bars:
            yval = bar.get_height()
            ax_c.text(bar.get_x() + bar.get_width()/2, yval + (fp_df_c['test_mse'].max() * 0.02), f'{yval:.3f}', ha='center', fontweight='bold')
            
    fig_c.tight_layout()
    fig_c.savefig(f"{base_name}_C.png", dpi=300, bbox_inches='tight')
    plt.close(fig_c)

    # --- PLOT D: Pareto Front Efficienza Spaziale ---
    fig_d, ax_d = plt.subplots(figsize=(7, 3.5))
    bb_models = ['LightTimesNet_MultiScale', 'LightTimesNet_Depthwise', 'LightTimesNet_Group', 'LightTimesNet_SingleKernel', 'DLinear']
    bb_df = df_base[(df_base['model'].isin(bb_models)) & (df_base['seq_len'] == 96)].copy()
    
    if not bb_df.empty:
        markers = ['*', 'o', 's', 'X', 'D'] 
        pareto_points = []
        
        for _, row in bb_df.iterrows():
            x_val = row.get('trainable_parameters')
            if pd.isna(x_val):
                continue
                
            y_val = row['test_mse']
            model_clean = row['model'].replace('LightTimesNet_', '')
            marker = markers[bb_models.index(row['model'])]
            
            ax_d.scatter(x_val, y_val, label=model_clean, marker=marker, s=200, edgecolors='black', zorder=3)
            pareto_points.append((x_val, y_val))
            
        pareto_points.sort(key=lambda p: (p[0], p[1]))
        front = []
        min_y = float('inf')
        
        for x, y in pareto_points:
            if y < min_y:
                front.append((x, y))
                min_y = y
                
        if len(front) > 1:
            front_x = [p[0] for p in front]
            front_y = [p[1] for p in front]
            ax_d.plot(front_x, front_y, 'k--', alpha=0.6, linewidth=2, label='Pareto Front', zorder=2)
            
        ax_d.set_title('D. Spatial Backbone Efficiency', fontweight='bold')
        ax_d.set_xlabel('Trainable Parameters')
        ax_d.set_ylabel('Test MSE')
        ax_d.set_xscale('log') 
        ax_d.legend()

    fig_d.tight_layout()
    fig_d.savefig(f"{base_name}_D.png", dpi=300, bbox_inches='tight')
    plt.close(fig_d)
    
    print(f"  OK: 4 PNG Vettoriali esportati per {dataset} (H={pred_len})")

def plot_mase_masking(df: pd.DataFrame):
    if 'masked_features' not in df.columns or df['masked_features'].isnull().all():
        print("  Nessun dato sui masked_features trovato. Eseguire prima un run aggiornato.")
        return
        
    print("\n" + "="*80)
    print("  GENERAZIONE PLOT: ANALISI MASE BOOLEAN MASKING")
    print("="*80)
    
    df_mask = df[df['model'] == 'DLinear'].copy() 
    if df_mask.empty:
        df_mask = df.drop_duplicates(subset=['dataset', 'pred_len']).copy()
        
    df_mask['total_features'] = df_mask['dataset'].apply(lambda x: 321 if 'electricity' in str(x).lower() else 7)
    df_mask['masked_pct'] = (df_mask['masked_features'] / df_mask['total_features']) * 100
    
    df_mask = df_mask.sort_values(by=['dataset', 'pred_len'])
    
    labels = [f"{str(d).upper()} (H={int(h)})" for d, h in zip(df_mask['dataset'], df_mask['pred_len'])]
    pcts = df_mask['masked_pct'].values
    counts = df_mask['masked_features'].values
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, pcts, color='#E24A33', edgecolor='black')
    
    threshold = 5.0
    ax.axhline(threshold, color='black', linestyle='--', linewidth=2, label=f'Reliability Threshold ({threshold}%)')
    
    ax.set_ylabel('Masked Features (%)')
    ax.set_title('Impact of Boolean Masking on MASE Computation', fontweight='bold', pad=15)
    
    max_y = max(max(pcts) * 1.2, threshold * 1.5) if len(pcts) > 0 else 10
    ax.set_ylim(0, max_y)
    
    for bar, count in zip(bars, counts):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max_y * 0.02), f"{int(count)} feat.", ha='center', va='bottom', fontweight='bold')
        
    ax.legend()
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("mase_masking_analysis.pdf", dpi=300, format='pdf')
    print("  Documento PDF Vettoriale salvato: mase_masking_analysis.pdf")
    plt.close(fig)

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    exp_dir = project_root / "experiments"
    
    df_metrics = load_all_metrics(exp_dir)
    if not df_metrics.empty:
        export_excel_tables(df_metrics)
        
        for h in [24, 48, 96]:
            plot_experiments(df_metrics, dataset="ETTh1", pred_len=h)
            plot_experiments(df_metrics, dataset="electricity", pred_len=h)
            
        plot_mase_masking(df_metrics)