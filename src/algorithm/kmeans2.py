# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_samples

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("注意：中文字体设置可能有问题")

# 1、获取数据
all_pd_data = pd.read_excel("./data/raw/gastric.xlsx", engine="openpyxl")
print(all_pd_data)

#   * 加载停用词
with open("./data/raw/stop_words.txt", 'r', encoding="utf-8") as f:
    stop_words = [l.strip() for l in f.readlines()]
stop_words.extend(['\n', '（', '）', ' '])
print(stop_words)

# 2、数据预处理
#   * 对中文文本进行分词
import jieba as jb
all_pd_data['Cut_Text'] = all_pd_data['Text'].apply(
    lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stop_words])
)

print(all_pd_data)

#   * 注意：KMeans是无监督学习，不需要划分标签
#     但如果数据中有 Label，我们可以用来对聚类结果进行对比评估
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    all_pd_data['Cut_Text'],
    all_pd_data['Label'],
    test_size=0.2,
    stratify=all_pd_data['Label']
)

# 3、特征工程
from sklearn.feature_extraction.text import TfidfVectorizer

# 3.1、求出 tf-idf 特征
transfer = TfidfVectorizer(stop_words=stop_words)
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

print("文本特征抽取的结果：\n", x_train)
feature_names = transfer.get_feature_names_out()
print("返回特征名字：\n", feature_names)
x_train_feature = feature_names.tolist()

print(x_train.shape)  # 比如 (200, 80)

# 4、构建 K-Means 模型
from sklearn.cluster import KMeans

#   * 实例化聚类器，n_clusters 设置为聚类数（通常与类别数相同）
kmeans = KMeans(n_clusters=5, random_state=43, n_init=50)
kmeans.fit(x_train)

#   * 获取聚类结果
y_pred = kmeans.labels_
print("聚类结果：\n", y_pred)

# 5、评估模型（仅在有标签时可用于评估）
from sklearn.metrics import adjusted_rand_score, silhouette_score

#   * ARI 衡量聚类与真实标签的一致性（有标签时）
ari = adjusted_rand_score(y_train, y_pred)
print(f"调整兰德指数（ARI）：{ari:.4f}")

#   * 轮廓系数 衡量聚类的紧密度和分离度（无监督评价）
sil = silhouette_score(x_train, y_pred)
print(f"轮廓系数（Silhouette Score）：{sil:.4f}")

# ===== 1 & 3. 合并：聚类分布条形图 + 混淆矩阵（放在同一张图上，1行2列） =====
fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1行2列，总宽12英寸，高4英寸

# ---- 子图 0：聚类分布条形图 ----
cluster_counts = pd.Series(y_pred).value_counts().sort_index()
axes[0].bar(cluster_counts.index, cluster_counts.values)
axes[0].set_xticks(cluster_counts.index)
axes[0].set_xticklabels([f'Cluster {i}' for i in cluster_counts.index])
axes[0].set_xlabel('Cluster')
axes[0].set_ylabel('样本数量')
axes[0].set_title('聚类分布条形图')

# ---- 子图 1：混淆矩阵 ----
cm = confusion_matrix(y_train, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_pred),
            yticklabels=np.unique(y_train),
            ax=axes[1])  # 重要：指定在第二个子图上绘制
axes[1].set_xlabel('聚类标签')
axes[1].set_ylabel('真实标签')
axes[1].set_title('混淆矩阵（聚类 vs 真实标签）')

# ---- 保存整张图（包含两个子图）----
plt.tight_layout()  # 避免子图重叠
plt.savefig('kmeans_cluster_and_confusion_matrix_combined.svg')  # 保存为一张图
plt.show()  # 显示

# ===== 2. 轮廓系数分布直方图（所有样本） =====
plt.figure(figsize=(6, 4))
sns.histplot(silhouette_samples(x_train, y_pred), bins=20, kde=False)
plt.xlabel('轮廓系数')
plt.ylabel('样本数量')
plt.title('轮廓系数分布直方图')
plt.savefig('kmeans_silhouette_score_distribution.svg')
plt.show()

# ===== 4. PCA降维散点图（2D可视化聚类）=====
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

# PCA降维到2维
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train.toarray())  # 注意：x_train 是稀疏矩阵，要 .toarray()

# 绘制散点图，按聚类标签着色
plt.figure(figsize=(6, 4))
scatter = plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='聚类标签')
plt.xlabel('主成分 1 (PCA1)')
plt.ylabel('主成分 2 (PCA2)')
plt.title('PCA降维散点图（按聚类着色）')
plt.savefig('kmeans_pca_distribution.svg')
plt.show()
