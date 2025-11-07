# -*- coding: utf-8 -*-
import pandas as pd
import jieba as jb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

# --------------------------
# 1. 全局设置 & 数据准备
# --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取数据
all_pd_data = pd.read_excel("./data/raw/gastric.xlsx", engine="openpyxl")

# 加载停用词
with open("./data/raw/stop_words.txt", 'r', encoding="utf-8") as f:
    stop_words = [l.strip() for l in f.readlines()]
stop_words.extend(['\n', '（', '）', ' '])

# 文本分词
all_pd_data['Cut_Text'] = all_pd_data['Text'].apply(
    lambda x: " ".join([w for w in jb.cut(x) if w not in stop_words])
)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    all_pd_data['Cut_Text'], all_pd_data['Label'],
    test_size=0.2, stratify=all_pd_data['Label'], random_state=42
)

# TF-IDF 特征提取
transfer = TfidfVectorizer(stop_words=stop_words)
x_train_tfidf = transfer.fit_transform(x_train)
x_test_tfidf = transfer.transform(x_test)

# --------------------------
# 2. 模型训练：KNN 和 SVM
# --------------------------

# --- KNN ---
knn_clf = KNeighborsClassifier(n_neighbors=10)
knn_clf.fit(x_train_tfidf, y_train)
y_predict_proba_knn = knn_clf.predict_proba(x_test_tfidf)

# --- SVM ---
svm_clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm_clf.fit(x_train_tfidf, y_train)
y_predict_proba_svm = svm_clf.predict_proba(x_test_tfidf)

# --------------------------
# 3. 绘制并排 ROC 曲线图（KNN 和 SVM 各一张）
# --------------------------

# 获取所有类别（统一）
classes = sorted(set(y_test.unique()).union(set(knn_clf.classes_).union(set(svm_clf.classes_))))

# 创建 1 行 2 列 的子图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 接近矩形，适合论文排版

# === 左图：KNN 的 ROC 曲线 ===
ax1 = axes[0]
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve((y_test == cls).astype(int), y_predict_proba_knn[:, i])
    auc_score = auc(fpr, tpr)
    ax1.plot(fpr, tpr, lw=2, label=f'类别 {cls} (AUC = {auc_score:.2f})')

ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='随机分类器')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('假正率 (FPR)')
ax1.set_ylabel('真正率 (TPR)')
ax1.set_title('KNN 模型 ROC 曲线')
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# === 右图：SVM 的 ROC 曲线 ===
ax2 = axes[1]
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve((y_test == cls).astype(int), y_predict_proba_svm[:, i])
    auc_score = auc(fpr, tpr)
    ax2.plot(fpr, tpr, lw=2, label=f'类别 {cls} (AUC = {auc_score:.2f})')

ax2.plot([0, 1], [0, 1], 'k--', lw=1, label='随机分类器')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('假正率 (FPR)')
ax2.set_ylabel('真正率 (TPR)')
ax2.set_title('SVM 模型 ROC 曲线')
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

# 调整布局防止重叠
plt.tight_layout()

# 保存图像（可用于论文）
plt.savefig("roc_curves_side_by_side.svg", dpi=300, bbox_inches='tight')
plt.show()
