import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# KNN算法指标
knn_precision = [1.00, 0.80, 1.00, 0.70, 0.40]
knn_recall = [0.77, 0.80, 0.77, 0.78, 0.80]
knn_f1 = [0.87, 0.80, 0.87, 0.74, 0.53]

# SVM算法指标
svm_precision = [0.90, 1.00, 1.00, 0.90, 0.40]
svm_recall = [0.90, 0.71, 0.77, 1.00, 1.00]
svm_f1 = [0.90, 0.83, 0.87, 0.95, 0.57]

# 整体指标
overall_metrics = ['Accuracy', 'Macro Avg F1', 'Weighted Avg F1', 'Macro Avg Recall']
knn_overall = [0.78, 0.76, 0.80, 0.78]
svm_overall = [0.84, 0.82, 0.86, 0.88]

# 类别标签
categories = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']

# 创建4个子图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('KNN与SVM算法详细性能指标对比', fontsize=16, fontweight='bold')

# 1. 精确率对比
x = np.arange(len(categories))
width = 0.35
axes[0, 0].bar(x - width/2, knn_precision, width, label='KNN', alpha=0.8, color='skyblue')
axes[0, 0].bar(x + width/2, svm_precision, width, label='SVM', alpha=0.8, color='lightcoral')
axes[0, 0].set_title('各类别精确率(Precision)对比')
axes[0, 0].set_xlabel('类别')
axes[0, 0].set_ylabel('精确率')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(categories)
axes[0, 0].legend()
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

# 添加数值标签
for i, (knn_val, svm_val) in enumerate(zip(knn_precision, svm_precision)):
    axes[0, 0].text(i - width/2, knn_val + 0.01, f'{knn_val:.2f}', ha='center', va='bottom', fontsize=9)
    axes[0, 0].text(i + width/2, svm_val + 0.01, f'{svm_val:.2f}', ha='center', va='bottom', fontsize=9)

# 2. 召回率对比
axes[0, 1].bar(x - width/2, knn_recall, width, label='KNN', alpha=0.8, color='skyblue')
axes[0, 1].bar(x + width/2, svm_recall, width, label='SVM', alpha=0.8, color='lightcoral')
axes[0, 1].set_title('各类别召回率(Recall)对比')
axes[0, 1].set_xlabel('类别')
axes[0, 1].set_ylabel('召回率')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(categories)
axes[0, 1].legend()
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

# 添加数值标签
for i, (knn_val, svm_val) in enumerate(zip(knn_recall, svm_recall)):
    axes[0, 1].text(i - width/2, knn_val + 0.01, f'{knn_val:.2f}', ha='center', va='bottom', fontsize=9)
    axes[0, 1].text(i + width/2, svm_val + 0.01, f'{svm_val:.2f}', ha='center', va='bottom', fontsize=9)

# 3. F1分数对比
axes[1, 0].bar(x - width/2, knn_f1, width, label='KNN', alpha=0.8, color='skyblue')
axes[1, 0].bar(x + width/2, svm_f1, width, label='SVM', alpha=0.8, color='lightcoral')
axes[1, 0].set_title('各类别F1-Score对比')
axes[1, 0].set_xlabel('类别')
axes[1, 0].set_ylabel('F1-Score')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(categories)
axes[1, 0].legend()
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

# 添加数值标签
for i, (knn_val, svm_val) in enumerate(zip(knn_f1, svm_f1)):
    axes[1, 0].text(i - width/2, knn_val + 0.01, f'{knn_val:.2f}', ha='center', va='bottom', fontsize=9)
    axes[1, 0].text(i + width/2, svm_val + 0.01, f'{svm_val:.2f}', ha='center', va='bottom', fontsize=9)

# 4. 整体指标对比
x_overall = np.arange(len(overall_metrics))
axes[1, 1].bar(x_overall - width/2, knn_overall, width, label='KNN', alpha=0.8, color='skyblue')
axes[1, 1].bar(x_overall + width/2, svm_overall, width, label='SVM', alpha=0.8, color='lightcoral')
axes[1, 1].set_title('整体性能指标对比')
axes[1, 1].set_xlabel('指标类型')
axes[1, 1].set_ylabel('分数')
axes[1, 1].set_xticks(x_overall)
axes[1, 1].set_xticklabels(overall_metrics, rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

# 添加数值标签
for i, (knn_val, svm_val) in enumerate(zip(knn_overall, svm_overall)):
    axes[1, 1].text(i - width/2, knn_val + 0.01, f'{knn_val:.2f}', ha='center', va='bottom', fontsize=9)
    axes[1, 1].text(i + width/2, svm_val + 0.01, f'{svm_val:.2f}', ha='center', va='bottom', fontsize=9)

# 调整布局
plt.tight_layout()

# 保存为SVG格式
svg_path = 'detailed_comparison.svg'
plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
print(svg_path)
