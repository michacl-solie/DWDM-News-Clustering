# -*- coding: utf-8 -*-
"""
快速生成主题词图 + 完整聚类结果JSON
（肘部法则图由step1_elbow.py单独生成）
"""
import os, json, pickle, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                              davies_bouldin_score, adjusted_rand_score,
                              normalized_mutual_info_score)
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

print("=" * 60)
print("DWDM聚类结果生成（三算法完整对比）")
print("=" * 60)

# ============================================================
# 1. 加载数据
# ============================================================
print("\n[1/6] 加载数据...")
X = np.load(os.path.join(DATA_DIR, "tfidf_reduced.npy"))
with open(os.path.join(DATA_DIR, "feature_names.json"),"r",encoding="utf-8") as f:
    feature_names = json.load(f)
tfidf = load_npz(os.path.join(DATA_DIR, "tfidf_matrix.npz"))
with open(os.path.join(DATA_DIR, "vectorizer.pkl"),"rb") as f:
    vec = pickle.load(f)
df = pd.read_csv(os.path.join(DATA_DIR, "processed_news.csv"), encoding="utf-8-sig")
labels = df["label"].values
print(f"  X={X.shape}, tfidf={tfidf.shape}, labels={len(labels)}")

# ============================================================
# 2. K-Means聚类（全量）
# ============================================================
print("\n[2/6] K-Means聚类 (全量数据)...")
t0 = __import__('time').time()
km = KMeans(n_clusters=14, random_state=42, n_init=10, max_iter=300)
km_labels = km.fit_predict(X)
print(f"  K-Means完成，耗时 {__import__('time').time()-t0:.1f}s")

# K-Means评估
print("  计算K-Means评估指标...")
km_sil = silhouette_score(X, km_labels)
km_ch = calinski_harabasz_score(X, km_labels)
km_db = davies_bouldin_score(X, km_labels)
km_ari = adjusted_rand_score(labels, km_labels)
km_nmi = normalized_mutual_info_score(labels, km_labels)
print(f"  Silhouette={km_sil:.4f}, CH={km_ch:.2f}, DB={km_db:.4f}, ARI={km_ari:.4f}, NMI={km_nmi:.4f}")

# ============================================================
# 3. 层次聚类（抽样20000）
# ============================================================
print("\n[3/6] 层次聚类 (抽样20000)...")
n_hc = 20000
idx_hc = np.random.RandomState(42).choice(len(X), n_hc, replace=False)
X_hc = X[idx_hc]
t0 = __import__('time').time()
agg = AgglomerativeClustering(n_clusters=14, linkage='ward')
agg_labels = agg.fit_predict(X_hc)
print(f"  层次聚类完成，耗时 {__import__('time').time()-t0:.1f}s")

hc_sil = silhouette_score(X_hc, agg_labels)
hc_ch = calinski_harabasz_score(X_hc, agg_labels)
hc_db = davies_bouldin_score(X_hc, agg_labels)
hc_ari = adjusted_rand_score(labels[idx_hc], agg_labels)
hc_nmi = normalized_mutual_info_score(labels[idx_hc], agg_labels)
print(f"  Silhouette={hc_sil:.4f}, CH={hc_ch:.2f}, DB={hc_db:.4f}, ARI={hc_ari:.4f}, NMI={hc_nmi:.4f}")

# ============================================================
# 4. DBSCAN（抽样10000）
# ============================================================
print("\n[4/6] DBSCAN聚类 (抽样10000)...")
n_db = 10000
idx_db = np.random.RandomState(42).choice(len(X), n_db, replace=False)
X_db = X[idx_db]
t0 = __import__('time').time()
db = DBSCAN(eps=0.8, min_samples=10, metric='cosine')
db_labels = db.fit_predict(X_db)
print(f"  DBSCAN完成，耗时 {__import__('time').time()-t0:.1f}s")

n_noise = int((db_labels == -1).sum())
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
print(f"  噪声点: {n_noise}, 簇数: {n_clusters_db}")

# DBSCAN评估（去除噪声点）
mask = db_labels != -1
if mask.sum() > 1 and len(set(db_labels[mask])) > 1:
    db_sil = silhouette_score(X_db[mask], db_labels[mask])
    db_ch = calinski_harabasz_score(X_db[mask], db_labels[mask])
    db_db = davies_bouldin_score(X_db[mask], db_labels[mask])
else:
    db_sil = db_ch = db_db = float('nan')
db_ari = adjusted_rand_score(labels[idx_db], db_labels)
db_nmi = normalized_mutual_info_score(labels[idx_db], db_labels)
print(f"  Silhouette={db_sil:.4f}, CH={db_ch:.2f}, DB={db_db:.4f}, ARI={db_ari:.4f}, NMI={db_nmi:.4f}")

# ============================================================
# 5. 生成主题词图
# ============================================================
print("\n[5/6] 生成主题词图...")
centers = km.cluster_centers_

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes_flat = axes.flatten()
for i in range(14):
    ax = axes_flat[i]
    c = centers[i]
    ti = np.argsort(c)[-10:][::-1]
    tw = [feature_names[j] for j in ti]
    ts = c[ti]
    ax.barh(np.arange(len(tw)), ts, color=plt.cm.viridis(np.linspace(0.2, 0.9, 10)), edgecolor='white')
    ax.set_yticks(np.arange(len(tw)))
    ax.set_yticklabels(tw, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f'Cluster {i}', fontsize=12, fontweight='bold')
    ax.set_xlabel('TF-IDF', fontsize=8)
for i in range(14, 16):
    axes_flat[i].set_visible(False)
plt.suptitle('K-Means Cluster Top-10 Keywords', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_cluster_keywords.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  [OK] 04_cluster_keywords.png")

# ============================================================
# 6. 保存聚类结果
# ============================================================
print("\n[6/6] 保存聚类结果...")

# 保存K-Means标签
df_km = pd.DataFrame({'label': labels, 'cluster': km_labels})
df_km.to_csv(os.path.join(DATA_DIR, "kmeans_clusters.csv"), index=False, encoding="utf-8-sig")

# 保存评估结果JSON
results = {
    'K-Means': {
        'Silhouette': float(km_sil),
        'CH_Index': float(km_ch),
        'Davies_Bouldin': float(km_db),
        'ARI': float(km_ari),
        'NMI': float(km_nmi),
        'N_Clusters': 14,
        'Inertia': float(km.inertia_),
    },
    'Agglomerative': {
        'Silhouette': float(hc_sil),
        'CH_Index': float(hc_ch),
        'Davies_Bouldin': float(hc_db),
        'ARI': float(hc_ari),
        'NMI': float(hc_nmi),
        'N_Clusters': 14,
    },
    'DBSCAN': {
        'Silhouette': float(db_sil) if not np.isnan(db_sil) else None,
        'CH_Index': float(db_ch) if not np.isnan(db_ch) else None,
        'Davies_Bouldin': float(db_db) if not np.isnan(db_db) else None,
        'ARI': float(db_ari),
        'NMI': float(db_nmi),
        'N_Clusters': int(n_clusters_db),
        'Noise_Points': int(n_noise),
    }
}
with open(os.path.join(DATA_DIR, "clustering_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("  [OK] clustering_results.json")

print("\n" + "=" * 60)
print("全部完成！")
print(f"  轮廓系数: K-Means={km_sil:.4f}, HC={hc_sil:.4f}, DBSCAN={db_sil:.4f}")
print(f"  ARI: K-Means={km_ari:.4f}, HC={hc_ari:.4f}, DBSCAN={db_ari:.4f}")
print(f"  NMI: K-Means={km_nmi:.4f}, HC={hc_nmi:.4f}, DBSCAN={db_nmi:.4f}")
print("=" * 60)
