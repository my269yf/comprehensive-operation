# -*- coding: utf-8 -*-
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
plt.xlabel('误判代价')
plt.ylabel('识别能力')
plt.title('SVM算法的ROC曲线')
plt.legend()
plt.grid(True, alpha=0.3)

# 显示图表
plt.tight_layout()
plt.savefig("roc-svm.svg")
plt.show()

print("ROC曲线绘制完成！")
print(f"模型包含 {len(clf.classes_)} 个类别: {clf.classes_}")

# 计算混淆矩阵（真实标签 vs 预测标签）
cm = confusion_matrix(y_test, y_predict)

# 绘制混淆矩阵热力图
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=clf.classes_, 
            yticklabels=clf.classes_)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵（SVM）')
plt.tight_layout()  # 防止标签重叠
plt.savefig('confusion_matrix_svm.svg')  # 保存为 svg 文件（可改为 png 等）
plt.show()
