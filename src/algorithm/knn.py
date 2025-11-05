# -*- coding: utf-8 -*-
import pandas as pd

# 1、获取数据
all_pd_data = pd.read_excel("./data/raw/gastric.xlsx", engine="openpyxl")
print(all_pd_data)

#   * 加载停用词
with open("./data/raw/stop_words.txt", 'r', encoding="utf-8") as f:
    stop_words = list(l.strip() for l in f.readlines())
stop_words.extend(['\n', '（', '）', ' '])  # 由于停用词中没有'\n'和中文的左右括号和空格，所以单独再加上去
print(stop_words)

# 2、数据预处理
#   * 对中文文本进行分词

import jieba as jb
all_pd_data['Cut_Text'] = all_pd_data['Text'].apply(
    lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stop_words]))

print(all_pd_data)

#   * 划分训练集和测试集 （按照Label采用分层抽样，保证训练集和测试集样本均匀）
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(all_pd_data['Cut_Text'],all_pd_data['Label'], test_size=0.2, stratify=all_pd_data['Label'])
# 3、特征工程

from sklearn.feature_extraction.text import TfidfVectorizer

# 3.1、求出训练集 tf-idf
# 3.1.1、实例化一个转换器类
transfer = TfidfVectorizer(stop_words=stop_words)
# 3.1.2、调用 fit_transform
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)
# 打印特征抽取结果
print("文本特征抽取的结果：\n", x_train)
# 新版本使用 get_feature_names_out()
feature_names = transfer.get_feature_names_out()
print("返回特征名字：\n", feature_names)
x_train_feature = feature_names.tolist()
y_train = list(y_train)
y_test = list(y_test)

print(x_train.shape)  # (200, 80)

# 4、构建KNN模型
from sklearn.neighbors import KNeighborsClassifier
clf =  KNeighborsClassifier(n_neighbors=10)
clf.fit(x_train, y_train) # 训练数据
y_predict = clf.predict(x_test)

#5、评估模型

from sklearn.metrics import classification_report
print(classification_report(y_predict, y_test))
