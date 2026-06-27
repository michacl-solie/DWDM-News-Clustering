# -*- coding: utf-8 -*-
"""
DWDM课程报告 — 数据预处理脚本
项目: 基于中文新闻数据的主题聚类分析
功能:
  1. 读取THUCNews 14类原始txt文件
  2. 中文分词 (jieba)
  3. 去除停用词
  4. TF-IDF向量化
  5. 数据仓库星型模型设计
  6. 输出CSV供后续建模
"""

import os
import re
import json
import random
import warnings
import time

import pandas as pd
import numpy as np
import jieba

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings('ignore')

# ============================================================
# 0. 配置参数
# ============================================================
DATA_ROOT = r"D:\DWDM_Report\data\THUCNews (1)\THUCNews"
OUTPUT_DIR = r"D:\DWDM_Report\data"
STOPWORDS_FILE = r"D:\DWDM_Report\data\THUCNewsProject-master\THUCNewsProject-master\data\stopwords.txt"

# 为降低计算量，每个类别抽样一定数量（全部84万篇对聚类来说太重）
SAMPLE_PER_CLASS = 5000  # 每类5000篇，14类共70000篇
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 1. 加载停用词表
# ============================================================
print("[1/6] 加载停用词表...")
stopwords = set()
if os.path.exists(STOPWORDS_FILE):
    with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            stopwords.add(line.strip())
# 补充常用中文停用词
extra_stopwords = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "所", "为", "所以", "因为", "但是", "然而", "虽然", "可以", "这个",
    "那个", "什么", "怎么", "哪", "吗", "啊", "哦", "嗯", "吧", "呢",
    "被", "把", "从", "让", "对", "向", "与", "及", "或", "但", "而",
    "且", "虽", "其", "之", "以", "于", "则", "等", "等", "年", "月", "日",
    "中", "已", "将", "能", "更", "已", "已", "还", "又", "再", "才",
    "目前", "目前", "现在", "进行", "通过", "表示", "认为", "称", "据",
    "报道", "据悉", "记者", "编辑", "责任编辑", "来源", "作者",
    "nbsp", "&nbsp", "&nbsp;", "\u3000", "\xa0", "\t", "\r",
    " ", "　", "  ", "   ",
}
stopwords.update(extra_stopwords)
print(f"    停用词总数: {len(stopwords)}")

# ============================================================
# 2. 读取原始数据
# ============================================================
print("[2/6] 读取原始THUCNews数据...")

categories = sorted(os.listdir(DATA_ROOT))
categories = [c for c in categories if os.path.isdir(os.path.join(DATA_ROOT, c))]
print(f"    发现 {len(categories)} 个类别: {categories}")

all_texts = []
all_labels = []

for cat in categories:
    cat_dir = os.path.join(DATA_ROOT, cat)
    files = [f for f in os.listdir(cat_dir) if f.endswith(".txt")]
    # 抽样
    if len(files) > SAMPLE_PER_CLASS:
        files = random.sample(files, SAMPLE_PER_CLASS)
    print(f"    {cat}: {len(files)} 篇")

    for fname in files:
        fpath = os.path.join(cat_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if len(content) < 20:  # 过滤过短文章
                continue
            all_texts.append(content)
            all_labels.append(cat)
        except Exception:
            continue

print(f"    总样本数: {len(all_texts)}")

# ============================================================
# 3. 文本清洗
# ============================================================
print("[3/6] 文本清洗与分词...")

def clean_text(text):
    """清洗单个文本"""
    # 移除URL
    text = re.sub(r'https?://\S+', '', text)
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除特殊字符，保留中文、英文、数字
    text = re.sub(r'[^\u4e00-\u9fff a-zA-Z0-9]', ' ', text)
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    """分词并去停用词"""
    cleaned = clean_text(text)
    words = jieba.cut(cleaned)
    # 过滤停用词和短词
    tokens = [w.strip() for w in words
              if w.strip() and w.strip() not in stopwords and len(w.strip()) > 1]
    return " ".join(tokens)

# 分词（jieba 首次加载慢，后续快）
jieba.setLogLevel(20)  # 减少日志输出
tokens_list = []
start = time.time()
for i, text in enumerate(all_texts):
    tokens_list.append(tokenize(text))
    if (i + 1) % 10000 == 0:
        elapsed = time.time() - start
        print(f"    已处理 {i+1}/{len(all_texts)} 篇 (耗时 {elapsed:.1f}s)")

elapsed = time.time() - start
print(f"    分词完成，总耗时 {elapsed:.1f}s")

# ============================================================
# 4. TF-IDF 向量化
# ============================================================
print("[4/6] TF-IDF向量化...")

vectorizer = TfidfVectorizer(
    max_features=5000,       # 保留Top 5000词
    max_df=0.5,              # 出现在>50%文档中的词视为噪音
    min_df=5,                # 至少出现在5篇文档中
    ngram_range=(1, 2),      # unigram + bigram
    sublinear_tf=True,       # 1+log(tf)，抑制高频词
)

tfidf_matrix = vectorizer.fit_transform(tokens_list)
feature_names = vectorizer.get_feature_names_out()

print(f"    TF-IDF矩阵形状: {tfidf_matrix.shape}")
print(f"    特征词数量: {len(feature_names)}")
print(f"    矩阵密度: {tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]) * 100:.2f}%")

# ============================================================
# 5. 降维（用于后续可视化）
# ============================================================
print("[5/6] SVD降维（保留200维用于聚类）...")

svd = TruncatedSVD(n_components=200, random_state=RANDOM_SEED)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

explained_var = svd.explained_variance_ratio_.sum()
print(f"    降维后形状: {tfidf_reduced.shape}")
print(f"    累计解释方差: {explained_var:.2%}")

# ============================================================
# 6. 保存预处理结果
# ============================================================
print("[6/6] 保存预处理结果...")

# 保存CSV（原始文本+标签+分词结果）
df = pd.DataFrame({
    "label": all_labels,
    "raw_text": all_texts,
    "tokens": tokens_list,
})
df.to_csv(os.path.join(OUTPUT_DIR, "processed_news.csv"),
          index=False, encoding="utf-8-sig")

# 保存TF-IDF矩阵（稀疏矩阵用npz）
from scipy.sparse import save_npz
save_npz(os.path.join(OUTPUT_DIR, "tfidf_matrix.npz"), tfidf_matrix)

# 保存降维后的矩阵
np.save(os.path.join(OUTPUT_DIR, "tfidf_reduced.npy"), tfidf_reduced)

# 保存特征词
with open(os.path.join(OUTPUT_DIR, "feature_names.json"), "w", encoding="utf-8") as f:
    json.dump(list(feature_names), f, ensure_ascii=False)

# 保存向量化器（用于后续还原）
import pickle
with open(os.path.join(OUTPUT_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)
with open(os.path.join(OUTPUT_DIR, "svd.pkl"), "wb") as f:
    pickle.dump(svd, f)

# 统计信息
print("\n" + "=" * 60)
print("数据预处理完成！")
print(f"总样本数: {len(all_texts)}")
print(f"类别数: {len(categories)}")
print(f"特征维度: {tfidf_matrix.shape[1]}")
print(f"降维后维度: {tfidf_reduced.shape[1]}")
print(f"\n各类别样本数:")
for cat in categories:
    count = sum(1 for l in all_labels if l == cat)
    print(f"  {cat}: {count}")
print("=" * 60)
