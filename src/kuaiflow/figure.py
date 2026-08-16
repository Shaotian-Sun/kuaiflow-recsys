import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 消融实验数据 - 测试集
data = {
    'Model': ['ID-only', 'History Only', 'Feature Only', 'Feature+History'],
    'Recall@50': [4.09, 4.62, 5.47, 5.97],
    'Recall@100': [7.34, 8.30, 9.43, 10.13],
    'HitRate@50': [11.82, 12.68, 15.02, 15.84],
    'HitRate@100': [19.48, 21.48, 23.78, 25.08],
    'NDCG@50': [1.57, 1.75, 2.16, 2.31],
    'NDCG@100': [2.30, 2.58, 3.03, 3.23],
    'Coverage@50': [99.93, 94.71, 99.95, 95.86],
    'Coverage@100': [99.99, 98.97, 100.00, 99.16]
}

df = pd.DataFrame(data)

# 设置图形样式
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Two-Tower Retrieval: Ablation Study Results (Test Set)', fontsize=16, fontweight='bold')

metrics = [
    ('Recall@50', 'Recall (%)', axes[0, 0]),
    ('Recall@100', 'Recall (%)', axes[0, 1]),
    ('HitRate@50', 'Hit Rate (%)', axes[0, 2]),
    ('HitRate@100', 'Hit Rate (%)', axes[1, 0]),
    ('NDCG@50', 'NDCG (%)', axes[1, 1]),
    ('NDCG@100', 'NDCG (%)', axes[1, 2]),
]

colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # ID-only, History, Feature, Both

for metric, ylabel, ax in metrics:
    values = df[metric].values
    bars = ax.bar(df['Model'], values, color=colors, edgecolor='black', linewidth=0.5)
    
    # 添加数值标签
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.15)
    ax.tick_params(axis='x', rotation=15, labelsize=9)

plt.tight_layout()
plt.savefig('artifacts/week2_ablation_results.png', dpi=300, bbox_inches='tight')
plt.show()