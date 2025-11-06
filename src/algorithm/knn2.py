# -*- coding: utf-8 -*-
import pandas as pd
import jieba as jb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("注意：中文字体设置可能有问题")

# 1、获取数据
all_pd_data = pd.read_excel("./data/raw/gastric.xlsx", engine="openpyxl")

# 加载停用词
with open("./data/raw/stop_words.txt", 'r', encoding="utf-8") as f:
    stop_words = list(l.strip() for l in f.readlines())
stop_words.extend(['\n', '（', '）', ' '])

# 2、数据预处理
all_pd_data['Cut_Text'] = all_pd_data['Text'].apply(
    lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stop_words]))

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    all_pd_data['Cut_Text'], all_pd_data['Label'], 
    test_size=0.2, stratify=all_pd_data['Label'])

# 3、特征工程
transfer = TfidfVectorizer(stop_words=stop_words)
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

# 4、构建KNN模型
clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(x_train, y_train)

# 5、获取预测概率（关键添加！）
y_predict_proba = clf.predict_proba(x_test)  # 添加这行
y_predict = clf.predict(x_test)

# 6、评估模型
print(classification_report(y_predict, y_test))

# 7、绘制ROC曲线（最简单版本）

# 将真实标签转换为numpy数组
y_true = np.array(y_test)

# 绘制每个类别的ROC曲线
plt.figure(figsize=(8, 6))

for i in range(len(clf.classes_)):
    fpr, tpr, _ = roc_curve(y_true == clf.classes_[i], y_predict_proba[:, i])
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {clf.classes_[i]} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('误判代价')
plt.ylabel('识别能力')
plt.title('KNN算法的ROC曲线')
plt.legend()
plt.grid(True)
plt.savefig("roc-knn.svg")
plt.show()