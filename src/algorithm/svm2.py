# -*- coding: utf-8 -*-
import pandas as pd

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

#   * 划分训练集和测试集（分层抽样，保证比例一致）
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    all_pd_data['Cut_Text'],
    all_pd_data['Label'],
    test_size=0.2,
    stratify=all_pd_data['Label']
)

# 3、特征工程
from sklearn.feature_extraction.text import TfidfVectorizer

# 3.1、求出训练集 tf-idf
transfer = TfidfVectorizer(stop_words=stop_words)
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

print("文本特征抽取的结果：\n", x_train)
feature_names = transfer.get_feature_names_out()
print("返回特征名字：\n", feature_names)
x_train_feature = feature_names.tolist()
y_train = list(y_train)
y_test = list(y_test)

print(x_train.shape)  # 比如 (200, 80)

# 4、构建SVM模型
from sklearn.svm import SVC

#   * 实例化SVM分类器
#   * 重要：添加 probability=True 才能获取概率
clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
clf.fit(x_train, y_train)

#   * 预测
y_predict = clf.predict(x_test)

# 5、评估模型
from sklearn.metrics import classification_report
print(classification_report(y_predict, y_test))

# ============ 新增的ROC曲线绘制代码 ============

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# 获取预测概率（关键步骤）
y_predict_proba = clf.predict_proba(x_test)

# 将真实标签转换为numpy数组
y_true = np.array(y_test)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))

# 为每个类别绘制ROC曲线
for i in range(len(clf.classes_)):
    # 计算当前类别的ROC曲线
    fpr, tpr, _ = roc_curve(y_true == clf.classes_[i], y_predict_proba[:, i])
    
    # 计算AUC值
    auc_score = auc(fpr, tpr)
    
    # 绘制曲线
    plt.plot(fpr, tpr, linewidth=2, 
             label=f'Class {clf.classes_[i]} (AUC = {auc_score:.3f})')

# 绘制随机分类器参考线
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')

# 设置图表属性
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM ROC Curves')
plt.legend()
plt.grid(True, alpha=0.3)

# 显示图表
plt.tight_layout()
plt.savefig("roc-svm.svg")
plt.show()

print("ROC曲线绘制完成！")
print(f"模型包含 {len(clf.classes_)} 个类别: {clf.classes_}")
