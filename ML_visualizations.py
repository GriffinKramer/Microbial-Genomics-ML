import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


# load data 
results_dir = 'results/'
files = {'rare_removal': 'ML_remove_rare_summary.csv',
    'no_removal':   'ML_no_removal_summary.csv',
    'n_100':        'ML_no_removal_summary.csv',
    'n_300':        'ML_n_300_summary.csv',
    'n_500':        'ML_n_500_summary.csv',
    'n_nocap':      'ML_n_nocap_summary.csv',
    'k_2000':       'ML_n_300_summary.csv',
    'k_5000':       'ML_k_5000_summary.csv',
    'k_10000':      'ML_k_10000_summary.csv',
    'k_nocap':      'ML_k_nocap_summary.csv',}

dfs = {name: pd.read_csv(os.path.join(results_dir, f)) for name, f in files.items()}

def best_f1(df, label):
    subset = df[df['Label'] == label]
    best = subset.loc[subset['Macro F1'].idxmax()]
    return best['Macro F1'], f"{best['Model']} {best['Rebalancing']}"

colors2 = ['#4878CF', '#6ACC65']
colors4 = ['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7']

# figure 1
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
fig.suptitle('Effect of Rare Kmer Removal on Model Performance', fontsize=13)

for ax, label in zip(axes, ['Country', 'Region']):
    f1_removal, label_removal = best_f1(dfs['rare_removal'], label)
    f1_noremoval, label_noremoval = best_f1(dfs['no_removal'], label)
    bars = ax.bar(['Rare Removal\n' + label_removal, 'No Removal\n' + label_noremoval],[f1_removal, f1_noremoval],color=colors2)
    ax.set_title(f'{label} Prediction')
    ax.set_ylabel('Macro F1')
    ax.set_ylim(0, 0.7)
    for bar, val in zip(bars, [f1_removal, f1_noremoval]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('results/fig1_rare_removal.png', dpi=300, bbox_inches='tight')
plt.close()

# figure 2
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.suptitle('Effect of N Cap on Model Performance', fontsize=13)
n_conditions = [('N=100',  dfs['n_100']),
    ('N=300',  dfs['n_300']),
    ('N=500',  dfs['n_500']),
    ('No Cap', dfs['n_nocap'])]

for ax, label in zip(axes, ['Country', 'Region']):
    f1s, xlabels = [], []
    for name, df in n_conditions:
        f1, best_label = best_f1(df, label)
        f1s.append(f1)
        xlabels.append(name + '\n' + best_label)

    bars = ax.bar(xlabels, f1s, color=colors4)
    ax.set_title(f'{label} Prediction')
    ax.set_ylabel('Macro F1')
    ax.set_ylim(0, 0.7)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('results/fig2_n_cap.png', dpi=300, bbox_inches='tight')
plt.close()

# figure 3
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
fig.suptitle('Effect of Feature Selection (k) on Model Performance', fontsize=13)

k_conditions = [('k=2000',  dfs['k_2000']),
    ('k=5000',  dfs['k_5000']),
    ('k=10000', dfs['k_10000']),
    ('No FS',   dfs['k_nocap'])]

for ax, label in zip(axes, ['Country', 'Region']):
    f1s, xlabels = [], []
    for name, df in k_conditions:
        f1, best_label = best_f1(df, label)
        f1s.append(f1)
        xlabels.append(name + '\n' + best_label)

    bars = ax.bar(xlabels, f1s, color=colors4)
    ax.set_title(f'{label} Prediction')
    ax.set_ylabel('Macro F1')
    ax.set_ylim(0, 0.7)
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('results/fig3_k_comparison.png', dpi=300, bbox_inches='tight')
plt.close()



# heatmap
model_rebal = [('GBDT','Oversample'),('GBDT','None'),('RF','Oversample'),
               ('RF','None'),('LinearSVC','Oversample'),('LinearSVC','None')]
row_labels = [f"{m} {r}" for m, r in model_rebal]
col_labels = ['Rare\nRemoval','No\nRemoval','N=100','N=300','N=500','No Cap',
              'k=2000','k=5000','k=10000','No FS']
df_keys = ['rare_removal','no_removal','n_100','n_300','n_500','n_nocap',
           'k_2000','k_5000','k_10000','k_nocap']

for label in ['Country', 'Region']:
    matrix = []
    for model, rebal in model_rebal:
        row = []
        for key in df_keys:
            df = dfs[key]
            match = df[(df['Label']==label)&(df['Model']==model)&(df['Rebalancing']==rebal)]
            row.append(match['Macro F1'].values[0] if len(match) else np.nan)
        matrix.append(row)
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(matrix, cmap='RdYlBu_r', vmin=0.2, vmax=0.6)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(f'{label} Prediction: Macro F1 Heatmap', fontsize=12)
    plt.colorbar(im, ax=ax, label='Macro F1')
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(f'results/fig_heatmap_{label.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"saved fig_heatmap_{label.lower()}.png")
