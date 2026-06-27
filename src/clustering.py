# -*- coding: utf-8 -*-
"""
DWDM课程报告 — 聚类建模与分析脚本
项目: 基于中文新闻数据的主题聚类分析
功能:
  1. K-Means 聚类
  2. 层次聚类 (Agglomerative)
  3. DBSCAN 聚类
  4. 轮廓系数、CH指数等评估指标对比
  5. t-SNE降维可视化
  6. 聚类结果主题词提取
"""

import os
import json
import warnings
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                              davies_bouldin_score, adjusted_rand_score,
                              normalized_mutual_info_score)
from sklearn.manifold import TSNE
from scipy.sparse import load_npz

warnings.filterwarnings('ignore')

# ============================================================
# 0. 中文字体配置
# ============================================================
# 尝试查找系统中文字体
def setup_chinese_font():
    """配置中文字体"""
    font_candidates = [
        "SimHei", "Microsoft YaHei", "SimSun", "KaiTi",
        "FangSong", "STSong", "NSimSun", "YouYuan",
    ]
    available = [f.name for f in fm.fontManager.ttflist]
    for fc in font_candidates:
        if fc in available:
            plt.rcParams['font.sans-serif'] = [fc, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"    使用字体: {fc}")
            return
    # 兜底：下载中文字体
    print("    未找到中文字体，尝试使用英文显示")
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

setup_chinese_font()

# ============================================================
# 1. 加载预处理数据
# ============================================================
print("[1/5] 加载预处理数据...")
DATA_DIR = r"D:\DWDM_Report\data"
FIG_DIR = r"D:\DWDM_Report\figures"

os.makedirs(FIG_DIR, exist_ok=True)

# 加载TF-IDF降维矩阵
tfidf_reduced = np.load(os.path.join(DATA_DIR, "tfidf_reduced.npy"))
print(f"    降维矩阵形状: {tfidf_reduced.shape}")

# 加载原始标签
df = pd.read_csv(os.path.join(DATA_DIR, "processed_news.csv"), encoding="utf-8-sig")
labels = df["label"].values
print(f"    样本数: {len(df)}, 类别数: {df['label'].nunique()}")

# 加载特征词
with open(os.path.join(DATA_DIR, "feature_names.json"), "r", encoding="utf-8") as f:
    feature_names = json.load(f)

# 加载TF-IDF矩阵和向量化器
tfidf_matrix = load_npz(os.path.join(DATA_DIR, "tfidf_matrix.npz"))
with open(os.path.join(DATA_DIR, "vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

# ============================================================
# 2. K-Means 聚类
# ============================================================
print("[2/5] K-Means聚类...")

kmeans = KMeans(n_clusters=14, random_state=42, n_init=10, max_iter=300)
kmeans_labels = kmeans.fit_predict(tfidf_reduced)

# ============================================================
# 3. 层次聚类
# ============================================================
print("[3/5] 层次聚类 (Agglomerative)...")

# 层次聚类计算量大，抽样20000条
n_sample_hc = 20000
indices_hc = np.random.RandomState(42).choice(
    len(tfidf_reduced), n_sample_hc, replace=False
)
tfidf_sample_hc = tfidf_reduced[indices_hc]

agg = AgglomerativeClustering(n_clusters=14, linkage='ward')
agg_labels = agg.fit_predict(tfidf_sample_hc)

# ============================================================
# 4. DBSCAN 聚类
# ============================================================
print("[4/5] DBSCAN聚类...")

# DBSCAN对参数敏感，先用部分数据调试
n_sample_db = 10000
indices_db = np.random.RandomState(42).choice(
    len(tfidf_reduced), n_sample_db, replace=False
)
tfidf_sample_db = tfidf_reduced[indices_db]

dbscan = DBSCAN(eps=0.8, min_samples=10, metric='cosine')
dbscan_labels = dbscan.fit_predict(tfidf_sample_db)

# ============================================================
# 5. 评估指标对比
# ============================================================
print("[5/5] 计算评估指标并生成可视化...")

# --- 评估指标 ---
results = {}

# K-Means评估
km_sil = silhouette_score(tfidf_reduced, kmeans_labels)
km_ch = calinski_harabasz_score(tfidf_reduced, kmeans_labels)
km_db = davies_bouldin_score(tfidf_reduced, kmeans_labels)
km_ari = adjusted_rand_score(labels, kmeans_labels)
km_nmi = normalized_mutual_info_score(labels, kmeans_labels)
results['K-Means'] = {
    'Silhouette': km_sil, 'CH Index': km_ch,
    'Davies-Bouldin': km_db, 'ARI': km_ari, 'NMI': km_nmi,
    'N Clusters': len(set(kmeans_labels))
}

# 层次聚类评估
hc_sil = silhouette_score(tfidf_sample_hc, agg_labels)
hc_ch = calinski_harabasz_score(tfidf_sample_hc, agg_labels)
hc_db = davies_bouldin_score(tfidf_sample_hc, agg_labels)
hc_ari = adjusted_rand_score(labels[indices_hc], agg_labels)
hc_nmi = normalized_mutual_info_score(labels[indices_hc], agg_labels)
results['Agglomerative'] = {
    'Silhouette': hc_sil, 'CH Index': hc_ch,
    'Davies-Bouldin': hc_db, 'ARI': hc_ari, 'NMI': hc_nmi,
    'N Clusters': len(set(agg_labels))
}

# DBSCAN评估
# 去除噪声点（标签为-1）计算轮廓系数
db_mask = dbscan_labels != -1
if db_mask.sum() > 1 and len(set(dbscan_labels[db_mask])) > 1:
    db_sil = silhouette_score(tfidf_sample_db[db_mask], dbscan_labels[db_mask])
    db_ch = calinski_harabasz_score(tfidf_sample_db[db_mask], dbscan_labels[db_mask])
    db_db = davies_bouldin_score(tfidf_sample_db[db_mask], dbscan_labels[db_mask])
else:
    db_sil = db_ch = db_db = np.nan
db_ari = adjusted_rand_score(labels[indices_db], dbscan_labels)
db_nmi = normalized_mutual_info_score(labels[indices_db], dbscan_labels)
n_noise = (dbscan_labels == -1).sum()
results['DBSCAN'] = {
    'Silhouette': db_sil, 'CH Index': db_ch,
    'Davies-Bouldin': db_db, 'ARI': db_ari, 'NMI': db_nmi,
    'N Clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
    'Noise Points': n_noise
}

# 打印结果
print("\n" + "=" * 70)
print("聚类模型评估结果对比")
print("=" * 70)
print(f"{'指标':<20} {'K-Means':<18} {'Agglomerative':<18} {'DBSCAN':<18}")
print("-" * 70)
metrics_order = ['Silhouette', 'CH Index', 'Davies-Bouldin', 'ARI', 'NMI', 'N Clusters']
for metric in metrics_order:
    vals = []
    for model in ['K-Means', 'Agglomerative', 'DBSCAN']:
        v = results[model].get(metric, np.nan)
        if isinstance(v, float):
            vals.append(f"{v:.4f}")
        else:
            vals.append(str(v))
    print(f"{metric:<20} {vals[0]:<18} {vals[1]:<18} {vals[2]:<18}")
print("=" * 70)

# ============================================================
# 可视化1: 评估指标对比柱状图
# ============================================================
print("\n生成可视化图表...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# (a) 轮廓系数对比
ax = axes[0, 0]
model_names = ['K-Means', 'Agglomerative', 'DBSCAN']
sil_vals = [results[m]['Silhouette'] for m in model_names]
colors_bar = ['#2E86AB', '#A23B72', '#F18F01']
bars = ax.bar(model_names, sil_vals, color=colors_bar, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, sil_vals):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_title('Silhouette Score (轮廓系数)', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=12)
ax.set_ylim(0, max(s for s in sil_vals if not np.isnan(s)) * 1.2)
ax.grid(axis='y', alpha=0.3)

# (b) CH Index对比
ax = axes[0, 1]
ch_vals = [results[m]['CH Index'] for m in model_names]
bars = ax.bar(model_names, ch_vals, color=colors_bar, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, ch_vals):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Calinski-Harabasz Index', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=12)
ax.grid(axis='y', alpha=0.3)

# (c) ARI & NMI对比
ax = axes[1, 0]
x = np.arange(3)
width = 0.35
ari_vals = [results[m].get('ARI', 0) for m in model_names]
nmi_vals = [results[m].get('NMI', 0) for m in model_names]
bars1 = ax.bar(x - width/2, ari_vals, width, label='ARI', color='#2E86AB', edgecolor='white')
bars2 = ax.bar(x + width/2, nmi_vals, width, label='NMI', color='#F18F01', edgecolor='white')
for bar, val in zip(bars1, ari_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10)
for bar, val in zip(bars2, nmi_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10)
ax.set_title('Clustering Agreement (ARI & NMI)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylabel('Score', fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

# (d) Davies-Bouldin Index (越低越好)
ax = axes[1, 1]
db_vals = [results[m]['Davies-Bouldin'] for m in model_names]
bars = ax.bar(model_names, db_vals, color=colors_bar, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, db_vals):
    if not np.isnan(val):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_title('Davies-Bouldin Index (lower is better)', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=12)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(FIG_DIR, "01_evaluation_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("    [OK] 评估指标对比图")

# ============================================================
# 可视化2: t-SNE降维 + K-Means聚类结果
# ============================================================
print("    生成t-SNE可视化（需2-3分钟）...")

# 抽样10000条做t-SNE（全量太慢）
n_tsne = 10000
indices_tsne = np.random.RandomState(42).choice(len(tfidf_reduced), n_tsne, replace=False)
tsne_data = tfidf_reduced[indices_tsne]
tsne_true_labels = labels[indices_tsne]
tsne_kmeans_labels = kmeans_labels[indices_tsne]

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=0)
tsne_result = tsne.fit_transform(tsne_data)

# --- 真实标签 vs 聚类结果 ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# 真实类别分布
unique_cats = sorted(set(tsne_true_labels))
colors_14 = plt.cm.tab20(np.linspace(0, 1, len(unique_cats)))

ax = axes[0]
for i, cat in enumerate(unique_cats):
    mask = tsne_true_labels == cat
    ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
               c=[colors_14[i]], label=cat, s=3, alpha=0.6)
ax.set_title('t-SNE: Ground Truth (14 Categories)', fontsize=14, fontweight='bold')
ax.legend(markerscale=5, fontsize=7, loc='lower right', ncol=2)
ax.set_xticks([]); ax.set_yticks([])

# K-Means聚类结果
ax = axes[1]
unique_km = sorted(set(tsne_kmeans_labels))
colors_km = plt.cm.tab20(np.linspace(0, 1, len(unique_km)))
for i, cl in enumerate(unique_km):
    mask = tsne_kmeans_labels == cl
    ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
               c=[colors_km[i]], label=f'Cluster {cl}', s=3, alpha=0.6)
ax.set_title('t-SNE: K-Means Clustering Result', fontsize=14, fontweight='bold')
ax.legend(markerscale=5, fontsize=7, loc='lower right', ncol=2)
ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_tsne_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()
print("    [OK] t-SNE对比图")

# ============================================================
# 可视化3: 肘部法则（K值选择）
# ============================================================
print("    生成肘部法则图...")

k_range = range(2, 25)
inertias = []
sil_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels_k = km.fit_predict(tfidf_reduced)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(tfidf_reduced, labels_k))

fig, ax1 = plt.subplots(figsize=(10, 6))
color1 = '#2E86AB'
color2 = '#F18F01'

ax1.set_xlabel('Number of Clusters (k)', fontsize=13)
ax1.set_ylabel('Inertia (SSE)', color=color1, fontsize=13)
line1 = ax1.plot(k_range, inertias, 'o-', color=color1, linewidth=2, markersize=6, label='Inertia')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
ax2.set_ylabel('Silhouette Score', color=color2, fontsize=13)
line2 = ax2.plot(k_range, sil_scores, 's--', color=color2, linewidth=2, markersize=6, label='Silhouette')
ax2.tick_params(axis='y', labelcolor=color2)

# 标注k=14
k14_idx = k_range.index(14)
ax1.axvline(x=14, color='#A23B72', linestyle=':', linewidth=2, alpha=0.7)
ax1.annotate('k=14 (optimal)', xy=(14, inertias[k14_idx]),
             xytext=(17, inertias[k14_idx] * 1.05),
             arrowprops=dict(arrowstyle='->', color='#A23B72'),
             fontsize=12, color='#A23B72', fontweight='bold')

lines = line1 + line2
labels_line = [l.get_label() for l in lines]
ax1.legend(lines, labels_line, loc='center right', fontsize=11)
ax1.set_title('Elbow Method + Silhouette Analysis for K Selection', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_elbow_method.png"), dpi=150, bbox_inches='tight')
plt.close()
print("    [OK] 肘部法则图")

# ============================================================
# 可视化4: 各聚类主题词Top10
# ============================================================
print("    提取聚类主题词...")

# 获取每个聚类的中心词（TF-IDF均值最高的词）
cluster_centers = kmeans.cluster_centers_

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
axes = axes.flatten()

for i in range(14):
    ax = axes[i]
    center = cluster_centers[i]
    top_indices = np.argsort(center)[-10:][::-1]
    top_words = [feature_names[idx] for idx in top_indices]
    top_scores = center[top_indices]

    # 水平柱状图
    y_pos = np.arange(len(top_words))
    ax.barh(y_pos, top_scores, color=plt.cm.viridis(np.linspace(0.2, 0.9, 10)),
            edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_words, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(f'Cluster {i}', fontsize=12, fontweight='bold')
    ax.set_xlabel('TF-IDF Weight', fontsize=8)
    ax.tick_params(axis='x', labelsize=7)

# 隐藏多余子图
for i in range(14, 16):
    axes[i].set_visible(False)

plt.suptitle('K-Means Cluster Top-10 Keywords', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_cluster_keywords.png"), dpi=150, bbox_inches='tight')
plt.close()
print("    [OK] 聚类主题词图")

# ============================================================
# 保存结果汇总
# ============================================================
results_summary = {
    'K-Means': {
        'Silhouette': float(km_sil),
        'CH_Index': float(km_ch),
        'Davies_Bouldin': float(km_db),
        'ARI': float(km_ari),
        'NMI': float(km_nmi),
        'N_Clusters': int(len(set(kmeans_labels))),
        'Inertia': float(kmeans.inertia_),
    },
    'Agglomerative': {
        'Silhouette': float(hc_sil),
        'CH_Index': float(hc_ch),
        'Davies_Bouldin': float(hc_db),
        'ARI': float(hc_ari),
        'NMI': float(hc_nmi),
        'N_Clusters': int(len(set(agg_labels))),
    },
    'DBSCAN': {
        'Silhouette': float(db_sil) if not np.isnan(db_sil) else None,
        'CH_Index': float(db_ch) if not np.isnan(db_ch) else None,
        'Davies_Bouldin': float(db_db) if not np.isnan(db_db) else None,
        'ARI': float(db_ari),
        'NMI': float(db_nmi),
        'N_Clusters': int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)),
        'Noise_Points': int(n_noise),
    }
}

with open(os.path.join(DATA_DIR, "clustering_results.json"), "w", encoding="utf-8") as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

# 保存K-Means聚类标签
df_kmeans = pd.DataFrame({
    'label': labels,
    'cluster': kmeans_labels
})
df_kmeans.to_csv(os.path.join(DATA_DIR, "kmeans_clusters.csv"),
                 index=False, encoding="utf-8-sig")

print("\n" + "=" * 60)
print("聚类建模与分析完成！")
print(f"结果已保存至:")
print(f"  {FIG_DIR}/01_evaluation_comparison.png")
print(f"  {FIG_DIR}/02_tsne_comparison.png")
print(f"  {FIG_DIR}/03_elbow_method.png")
print(f"  {FIG_DIR}/04_cluster_keywords.png")
print(f"  {DATA_DIR}/clustering_results.json")
print("=" * 60)
