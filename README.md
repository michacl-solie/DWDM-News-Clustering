# 基于THUCNews中文新闻数据的主题聚类分析

> 课程：数据仓库与数据挖掘 (DWDM)
> 姓名：于冬冬　学号：2022101060　专业：信息与计算科学（数学系）

## 项目简介

本项目以清华大学THUCNews中文新闻数据集为研究对象，对14个类别共68,551篇新闻文本进行无监督主题聚类分析。通过TF-IDF特征提取与SVD降维，分别使用K-Means、层次聚类(Agglomerative)和DBSCAN三种算法进行聚类，并采用轮廓系数、Calinski-Harabasz指数、Davies-Bouldin指数、ARI和NMI等多维指标进行评估对比，最后通过t-SNE可视化直观展示聚类效果。

## 目录结构

```
DWDM_Report/
├── data/                           # 数据目录
│   ├── THUCNews (1)/THUCNews/      # 原始数据（14类新闻文本）
│   ├── processed_news.csv          # 预处理后CSV（文本+标签+分词）
│   ├── tfidf_matrix.npz            # TF-IDF稀疏矩阵 (68551×5000)
│   ├── tfidf_reduced.npy           # SVD降维矩阵 (68551×200)
│   ├── feature_names.json          # 5000个特征词
│   ├── vectorizer.pkl              # TF-IDF向量化器
│   ├── svd.pkl                     # SVD模型
│   ├── kmeans_clusters.csv         # K-Means聚类标签
│   └── clustering_results.json     # 聚类评估指标汇总
├── src/                            # 源代码
│   ├── preprocess.py               # 数据预处理（分词/TF-IDF/降维）
│   ├── clustering.py               # 三算法聚类+评估+可视化
│   ├── step1_elbow.py              # 肘部法则K值选择
│   └── gen_figures.py              # 补充图表生成
├── figures/                        # 可视化图表
│   ├── 01_evaluation_comparison.png  # 三算法指标对比
│   ├── 02_tsne_comparison.png        # t-SNE真实vs聚类
│   ├── 03_elbow_method.png           # 肘部法则+轮廓系数
│   └── 04_cluster_keywords.png       # 聚类主题词Top10
├── output/                         # 输出目录
├── 于冬冬2022101060基于THUCNews中文新闻数据的主题聚类分析.docx  # 实验报告
├── 于冬冬2022101060代码.zip         # 代码提交包
├── README.md
├── requirements.txt
└── .gitignore
```

## 环境依赖

- Python 3.14.0
- 依赖包见 `requirements.txt`

## 运行流程

```bash
# 1. 数据预处理（约5分钟）
python src/preprocess.py

# 2. 肘部法则选K（约15分钟）
python src/step1_elbow.py

# 3. 聚类建模与评估（约10分钟）
python src/clustering.py

# 4. 补充图表生成
python src/gen_figures.py
```

## 数据集

THUCNews：清华大学自然语言处理实验室发布的中文新闻文本分类数据集，包含14个类别：
体育、娱乐、家居、彩票、房产、教育、时尚、时政、星座、游戏、社会、科技、股票、财经

本项目每类抽样约5000篇，共68,551篇。
