import matplotlib.pyplot as plt
import numpy as np

# Stile accademico
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'axes.spines.top': False,
    'axes.spines.right': False
})

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# =====================================================================
# PLOT A: Analisi Cicli (DLinear vs TCN vs FixedPeriod)
# =====================================================================
seq_lens = [96, 192, 384]
dlinear_mse = [0.38, 0.41, 0.48]
tcn_mse = [0.37, 0.38, 0.42]
fixed_mse = [0.34, 0.32, 0.30]

axes[0].plot(seq_lens, dlinear_mse, marker='o', linestyle='--', color='#C44E52', label='DLinear')
axes[0].plot(seq_lens, tcn_mse, marker='d', linestyle='-.', color='#8C564B', label='CausalTCN')
axes[0].plot(seq_lens, fixed_mse, marker='^', linestyle='-', color='#55A868', label='FixedPeriod (Ours)')

axes[0].set_xticks(seq_lens)
axes[0].set_xlabel('Sequence Length (Lookback)')
axes[0].set_ylabel('Test MSE')
axes[0].set_title('A. Temporal Context Scalability (Cicli)')
axes[0].legend()

# =====================================================================
# PLOT B: Ablazione Topologica (FixedPeriod 17 vs 24 vs 48)
# =====================================================================
periods = ['P=17\n(Asincrono)', 'P=24\n(Fisico)', 'P=48\n(Multiplo)']
mse_vals = [0.58, 0.33, 0.35]
colors = ['#C44E52', '#55A868', '#4C72B0']

bars = axes[1].bar(periods, mse_vals, color=colors, edgecolor='black', width=0.5)
axes[1].set_ylabel('Test MSE')
axes[1].set_title('B. Period Topology Impact')

for bar in bars:
    yval = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')
axes[1].set_ylim(0, 0.7)

# =====================================================================
# PLOT C: Backbone2D Efficiency (Pareto: Epoch Time vs MSE)
# =====================================================================
# Dati simulati: Tempo per Epoca (s) e MSE
epoch_time = [18.5, 5.2, 8.4, 4.1]
mses = [0.32, 0.36, 0.34, 0.45]
labels = ['MultiScale', 'Depthwise', 'Group', 'SingleKernel']
markers = ['*', 'o', 's', 'X']

for i in range(4):
    axes[2].scatter(epoch_time[i], mses[i], label=labels[i], marker=markers[i], s=200)

axes[2].plot([5.2, 18.5], [0.36, 0.32], 'k--', alpha=0.5, label='Pareto Front')

axes[2].set_xlabel('Average Epoch Time (Seconds)')
axes[2].set_ylabel('Test MSE')
axes[2].set_title('C. Backbone Efficiency (Time vs Error)')
axes[2].legend(loc='upper right')

plt.tight_layout()
plt.savefig('notebook_analysis_mockups.pdf', format='pdf', dpi=300)
plt.show()