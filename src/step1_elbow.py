# -*- coding: utf-8 -*-
"""步骤1：肘部法则图"""
import os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = r"D:\DWDM_Report\data"
FIG_DIR = r"D:\DWDM_Report\figures"
os.makedirs(FIG_DIR, exist_ok=True)

for fc in ["SimHei","Microsoft YaHei","SimSun"]:
    if fc in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams['font.sans-serif'] = [fc,'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

print("加载数据...", flush=True)
X = np.load(os.path.join(DATA_DIR, "tfidf_reduced.npy"))
print(f"X shape: {X.shape}", flush=True)

# 采样5000条用于加速轮廓系数计算（Inertia仍用全量数据）
np.random.seed(42)
if X.shape[0] > 5000:
    sample_idx = np.random.choice(X.shape[0], 5000, replace=False)
    X_sample = X[sample_idx]
else:
    X_sample = X
print(f"采样 shape: {X_sample.shape} (用于轮廓系数加速)", flush=True)

print("计算K=2~24的Inertia和Silhouette...", flush=True)
k_range = list(range(2, 25))
inertias, sils = [], []
for i, k in enumerate(k_range):
    print(f"  k={k} ({i+1}/{len(k_range)})...", flush=True)
    km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=300)
    km.fit(X)
    inertias.append(km.inertia_)
    # 用采样计算轮廓系数加速
    sils.append(silhouette_score(X_sample, km.predict(X_sample)))

print("绘图...")
fig, ax1 = plt.subplots(figsize=(10, 6))
c1, c2 = '#2E86AB', '#F18F01'
ax1.set_xlabel('Number of Clusters (k)', fontsize=13)
ax1.set_ylabel('Inertia (SSE)', color=c1, fontsize=13)
ax1.plot(k_range, inertias, 'o-', color=c1, linewidth=2, markersize=6, label='Inertia')
ax1.tick_params(axis='y', labelcolor=c1)
ax2 = ax1.twinx()
ax2.set_ylabel('Silhouette Score', color=c2, fontsize=13)
ax2.plot(k_range, sils, 's--', color=c2, linewidth=2, markersize=6, label='Silhouette')
ax2.tick_params(axis='y', labelcolor=c2)
ax1.axvline(x=14, color='#A23B72', linestyle=':', linewidth=2)
k14 = k_range.index(14)
ax1.annotate('k=14', xy=(14, inertias[k14]),
             xytext=(17, inertias[k14]*1.05),
             arrowprops=dict(arrowstyle='->', color='#A23B72'),
             fontsize=12, color='#A23B72', fontweight='bold')
ax1.legend(loc='center right', fontsize=11)
ax1.set_title('Elbow Method + Silhouette Analysis', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join(FIG_DIR, "03_elbow_method.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ 肘部法则图已保存: {out}")
