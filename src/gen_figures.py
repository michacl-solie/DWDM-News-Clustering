# -*- coding: utf-8 -*-
"""快速生成缺失图表：肘部法则 + 主题词"""
import os, json, pickle, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.sparse import load_npz
warnings.filterwarnings('ignore')

DATA_DIR = r"D:\DWDM_Report\data"
FIG_DIR = r"D:\DWDM_Report\figures"
os.makedirs(FIG_DIR, exist_ok=True)

# 字体
for fc in ["SimHei","Microsoft YaHei","SimSun"]:
    if fc in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams['font.sans-serif'] = [fc,'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        break

print("加载数据...")
X = np.load(os.path.join(DATA_DIR, "tfidf_reduced.npy"))
with open(os.path.join(DATA_DIR, "feature_names.json"),"r",encoding="utf-8") as f:
    fn = json.load(f)
tfidf = load_npz(os.path.join(DATA_DIR, "tfidf_matrix.npz"))
with open(os.path.join(DATA_DIR, "vectorizer.pkl"),"rb") as f:
    vec = pickle.load(f)
df = pd.read_csv(os.path.join(DATA_DIR, "processed_news.csv"), encoding="utf-8-sig")
labels = df["label"].values
print(f"X={X.shape}, labels={len(labels)}")

# --- 1. 肘部法则 ---
print("1. 肘部法则...")
kr = range(2, 25)
inertias, sils = [], []
for k in kr:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    lbls = km.fit_predict(X)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X, lbls))

fig, ax1 = plt.subplots(figsize=(10, 6))
c1, c2 = '#2E86AB', '#F18F01'
ax1.set_xlabel('Number of Clusters (k)', fontsize=13)
ax1.set_ylabel('Inertia (SSE)', color=c1, fontsize=13)
ax1.plot(list(kr), inertias, 'o-', color=c1, linewidth=2, markersize=6, label='Inertia')
ax1.tick_params(axis='y', labelcolor=c1)
ax2 = ax1.twinx()
ax2.set_ylabel('Silhouette Score', color=c2, fontsize=13)
ax2.plot(list(kr), sils, 's--', color=c2, linewidth=2, markersize=6, label='Silhouette')
ax2.tick_params(axis='y', labelcolor=c2)
ax1.axvline(x=14, color='#A23B72', linestyle=':', linewidth=2)
ax1.annotate('k=14', xy=(14, inertias[kr.index(14)]),
             xytext=(17, inertias[kr.index(14)]*1.05),
             arrowprops=dict(arrowstyle='->', color='#A23B72'),
             fontsize=12, color='#A23B72', fontweight='bold')
ax1.legend(loc='center right', fontsize=11)
ax1.set_title('Elbow Method + Silhouette Analysis', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_elbow_method.png"), dpi=150, bbox_inches='tight')
plt.close()
print("    OK")

# --- 2. 主题词 ---
print("2. 主题词...")
km = KMeans(n_clusters=14, random_state=42, n_init=10, max_iter=300)
km.fit(X)
centers = km.cluster_centers_

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()
for i in range(14):
    ax = axes[i]
    c = centers[i]
    ti = np.argsort(c)[-10:][::-1]
    tw = [fn[j] for j in ti]
    ts = c[ti]
    ax.barh(np.arange(len(tw)), ts, color=plt.cm.viridis(np.linspace(0.2, 0.9, 10)), edgecolor='white')
    ax.set_yticks(np.arange(len(tw)))
    ax.set_yticklabels(tw, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f'Cluster {i}', fontsize=12, fontweight='bold')
    ax.set_xlabel('TF-IDF', fontsize=8)
for i in range(14, 16):
    axes[i].set_visible(False)
plt.suptitle('K-Means Cluster Top-10 Keywords', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_cluster_keywords.png"), dpi=150, bbox_inches='tight')
plt.close()
print("    OK")

# --- 3. 保存聚类结果 ---
km_lbls = km.predict(X)
df_kmeans = pd.DataFrame({'label': labels, 'cluster': km_lbls})
df_kmeans.to_csv(os.path.join(DATA_DIR, "kmeans_clusters.csv"), index=False, encoding="utf-8-sig")

from sklearn.metrics import silhouette_score as ss, calinski_harabasz_score as ch, davies_bouldin_score as db, adjusted_rand_score as ari, normalized_mutual_info_score as nmi
results = {
    'K-Means': {
        'Silhouette': float(ss(X, km_lbls)),
        'CH_Index': float(ch(X, km_lbls)),
        'Davies_Bouldin': float(db(X, km_lbls)),
        'ARI': float(ari(labels, km_lbls)),
        'NMI': float(nmi(labels, km_lbls)),
        'N_Clusters': 14,
        'Inertia': float(km.inertia_),
    }
}
with open(os.path.join(DATA_DIR, "clustering_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("全部完成！")
