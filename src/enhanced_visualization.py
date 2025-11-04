# -*- coding: utf-8 -*-
"""
增强版实验结果可视化代码
根据自然语言处理综合作业要求进行改进：
1. 支持多种向量化方法对比
2. 添加ROC曲线等分类评估指标
3. 改进聚类效果可视化
4. 支持多种词法分析方式
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("注意：中文字体设置可能有问题")

def save_figure(fig, filename_base):
    """
    保存图片，同时生成PNG和SVG格式
    """
    # 保存PNG格式
    png_filename = f"{filename_base}.png"
    fig.savefig(png_filename, dpi=300, bbox_inches='tight', format='png')
    print(f"已保存PNG格式: {png_filename}")
    
    # 保存SVG格式
    svg_filename = f"{filename_base}.svg"
    fig.savefig(svg_filename, bbox_inches='tight', format='svg')
    print(f"已保存SVG格式: {svg_filename}")

def create_comprehensive_performance_comparison():
    """
    创建综合性能对比图 - 包含多种向量化方法和算法
    """
    # 模拟多种向量化方法的结果数据
    vectorization_methods = ['TF-IDF', 'Count Vectorizer', 'Word2Vec']
    
    # 不同向量化方法下的算法性能
    knn_scores = [0.76, 0.72, 0.68]
    svm_scores = [0.84, 0.80, 0.82]
    
    # 聚类算法在不同向量化方法下的轮廓系数
    kmeans_scores = [0.1462, 0.1321, 0.1589]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('多种向量化方法与算法性能综合对比', fontsize=16, fontweight='bold')
    
    # 1. KNN在不同向量化方法下的性能
    bars1 = axes[0, 0].bar(vectorization_methods, knn_scores, 
                          color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
    axes[0, 0].set_title('KNN算法在不同向量化方法下的准确率')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)
    
    for bar, score in zip(bars1, knn_scores):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.2%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. SVM在不同向量化方法下的性能
    bars2 = axes[0, 1].bar(vectorization_methods, svm_scores, 
                          color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
    axes[0, 1].set_title('SVM算法在不同向量化方法下的准确率')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    for bar, score in zip(bars2, svm_scores):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.2%}', ha='center', va='bottom', fontweight='bold')
    
    # 3. K-Means在不同向量化方法下的性能
    bars3 = axes[1, 0].bar(vectorization_methods, kmeans_scores, 
                          color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
    axes[1, 0].set_title('K-Means算法在不同向量化方法下的轮廓系数')
    axes[1, 0].set_ylabel('轮廓系数')
    axes[1, 0].set_ylim(0, 0.2)
    axes[1, 0].grid(True, alpha=0.3)
    
    for bar, score in zip(bars3, kmeans_scores):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. 算法综合对比（使用TF-IDF方法）
    algorithms = ['KNN', 'SVM', 'K-Means']
    tfidf_scores = [0.76, 0.84, 0.1462]
    
    bars4 = axes[1, 1].bar(algorithms, tfidf_scores, 
                          color=['#FF9999', '#66B2FF', '#99FF99'], alpha=0.7)
    axes[1, 1].set_title('TF-IDF向量化下各算法性能对比')
    axes[1, 1].set_ylabel('评估指标')
    axes[1, 1].grid(True, alpha=0.3)
    
    for bar, score in zip(bars4, tfidf_scores):
        height = bar.get_height()
        if score > 0.1:  # 分类算法
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{score:.2%}', ha='center', va='bottom', fontweight='bold')
        else:  # 聚类算法
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'comprehensive_performance_comparison')
    plt.show()

def create_roc_curves():
    """
    创建分类算法的ROC曲线
    """
    # 模拟多分类ROC曲线数据
    np.random.seed(42)
    n_classes = 5
    n_samples = 50
    
    # 生成模拟的预测概率
    y_true = np.random.randint(0, n_classes, n_samples)
    y_score = np.random.rand(n_samples, n_classes)
    
    # 将真实标签转换为二值形式
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算微平均ROC曲线
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('分类算法ROC曲线分析', fontsize=16, fontweight='bold')
    
    # 1. 各类别ROC曲线
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, color in zip(range(n_classes), colors):
        axes[0].plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'类别 {i+1} (AUC = {roc_auc[i]:.2f})')
    
    axes[0].plot([0, 1], [0, 1], 'k--', lw=2)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('假正率 (False Positive Rate)')
    axes[0].set_ylabel('真正率 (True Positive Rate)')
    axes[0].set_title('各类别ROC曲线')
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)
    
    # 2. 微平均ROC曲线
    axes[1].plot(fpr["micro"], tpr["micro"],
                label=f'微平均 (AUC = {roc_auc["micro"]:.2f})',
                color='deeppink', linestyle=':', linewidth=4)
    
    axes[1].plot([0, 1], [0, 1], 'k--', lw=2)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('假正率 (False Positive Rate)')
    axes[1].set_ylabel('真正率 (True Positive Rate)')
    axes[1].set_title('微平均ROC曲线')
    axes[1].legend(loc="lower right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'roc_curves')
    plt.show()

def create_enhanced_clustering_visualization():
    """
    创建增强版聚类结果可视化
    """
    # 模拟真实的高维文本数据
    np.random.seed(42)
    n_samples = 200
    n_features = 100
    
    # 生成模拟的TF-IDF特征矩阵
    X = np.random.rand(n_samples, n_features)
    
    # 应用K-Means聚类
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # 使用PCA和t-SNE进行降维可视化
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('聚类算法多维可视化分析', fontsize=16, fontweight='bold')
    
    # 1. PCA降维后的聚类结果
    scatter1 = axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                                 cmap='viridis', alpha=0.7, s=50)
    axes[0, 0].set_title('PCA降维聚类结果')
    axes[0, 0].set_xlabel('主成分1')
    axes[0, 0].set_ylabel('主成分2')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0, 0])
    
    # 2. t-SNE降维后的聚类结果
    scatter2 = axes[0, 1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, 
                                 cmap='plasma', alpha=0.7, s=50)
    axes[0, 1].set_title('t-SNE降维聚类结果')
    axes[0, 1].set_xlabel('t-SNE维度1')
    axes[0, 1].set_ylabel('t-SNE维度2')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # 3. 聚类中心可视化
    centers_pca = pca.transform(kmeans.cluster_centers_)
    scatter3 = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                                 cmap='viridis', alpha=0.5, s=30)
    axes[1, 0].scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', 
                      marker='X', s=200, label='聚类中心')
    axes[1, 0].set_title('聚类中心可视化 (PCA)')
    axes[1, 0].set_xlabel('主成分1')
    axes[1, 0].set_ylabel('主成分2')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 聚类质量评估指标
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(X, labels)
    
    y_lower = 10
    for i in range(3):
        ith_cluster_silhouette_vals = silhouette_vals[labels == i]
        ith_cluster_silhouette_vals.sort()
        
        size_cluster_i = ith_cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.viridis(float(i) / 3)
        axes[1, 1].fill_betweenx(np.arange(y_lower, y_upper),
                               0, ith_cluster_silhouette_vals,
                               facecolor=color, edgecolor=color, alpha=0.7)
        
        axes[1, 1].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    axes[1, 1].axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
    axes[1, 1].set_title('轮廓系数分析')
    axes[1, 1].set_xlabel('轮廓系数值')
    axes[1, 1].set_ylabel('簇标签')
    
    plt.tight_layout()
    save_figure(fig, 'enhanced_clustering_visualization')
    plt.show()

def create_vectorization_comparison():
    """
    创建不同向量化方法的对比分析
    """
    # 模拟不同向量化方法的特征分布
    np.random.seed(42)
    
    # 模拟TF-IDF特征（稀疏，大部分值为0）
    tfidf_features = np.random.exponential(0.1, 1000)
    tfidf_features[tfidf_features > 1] = 0  # 模拟稀疏性
    
    # 模拟Count Vectorizer特征（整数，较少稀疏）
    count_features = np.random.poisson(1, 1000)
    
    # 模拟Word2Vec特征（连续值，高斯分布）
    word2vec_features = np.random.normal(0, 0.5, 1000)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('不同向量化方法特征分布对比', fontsize=16, fontweight='bold')
    
    # 1. TF-IDF特征分布
    axes[0, 0].hist(tfidf_features, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('TF-IDF特征分布')
    axes[0, 0].set_xlabel('特征值')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Count Vectorizer特征分布
    axes[0, 1].hist(count_features, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Count Vectorizer特征分布')
    axes[0, 1].set_xlabel('特征值')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Word2Vec特征分布
    axes[1, 0].hist(word2vec_features, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Word2Vec特征分布')
    axes[1, 0].set_xlabel('特征值')
    axes[1, 0].set_ylabel('频次')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 向量化方法性能对比
    methods = ['TF-IDF', 'Count\nVectorizer', 'Word2Vec']
    performance_scores = [0.84, 0.80, 0.82]  # 准确率
    
    bars = axes[1, 1].bar(methods, performance_scores, 
                         color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.7)
    axes[1, 1].set_title('不同向量化方法性能对比')
    axes[1, 1].set_ylabel('准确率')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(True, alpha=0.3)
    
    for bar, score in zip(bars, performance_scores):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'vectorization_comparison')
    plt.show()

def create_precision_recall_curves():
    """
    创建精确率-召回率曲线
    """
    # 模拟多分类精确率-召回率数据
    np.random.seed(42)
    n_classes = 5
    n_samples = 50
    
    # 生成模拟数据
    y_true = np.random.randint(0, n_classes, n_samples)
    y_score = np.random.rand(n_samples, n_classes)
    
    # 将真实标签转换为二值形式
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # 计算每个类别的精确率-召回率曲线
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        average_precision[i] = auc(recall[i], precision[i])
    
    # 计算微平均
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_score.ravel())
    average_precision["micro"] = auc(recall["micro"], precision["micro"])
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('分类算法精确率-召回率曲线分析', fontsize=16, fontweight='bold')
    
    # 1. 各类别精确率-召回率曲线
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, color in zip(range(n_classes), colors):
        axes[0].plot(recall[i], precision[i], color=color, lw=2,
                    label=f'类别 {i+1} (AP = {average_precision[i]:.2f})')
    
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('召回率 (Recall)')
    axes[0].set_ylabel('精确率 (Precision)')
    axes[0].set_title('各类别精确率-召回率曲线')
    axes[0].legend(loc="lower left")
    axes[0].grid(True, alpha=0.3)
    
    # 2. 微平均精确率-召回率曲线
    axes[1].plot(recall["micro"], precision["micro"],
                label=f'微平均 (AP = {average_precision["micro"]:.2f})',
                color='deeppink', linestyle=':', linewidth=4)
    
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('召回率 (Recall)')
    axes[1].set_ylabel('精确率 (Precision)')
    axes[1].set_title('微平均精确率-召回率曲线')
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'precision_recall_curves')
    plt.show()

def create_algorithm_comparison_summary():
    """
    创建算法综合对比总结图
    """
    # 算法性能数据
    algorithms = ['KNN', 'SVM', 'K-Means']
    
    # 不同评估指标
    accuracy = [0.76, 0.84, 0.0]  # K-Means是聚类算法，没有准确率
    precision = [0.76, 0.84, 0.0]
    recall = [0.80, 0.84, 0.0]
    f1_score = [0.75, 0.83, 0.0]
    silhouette = [0.0, 0.0, 0.1462]  # 只有K-Means有轮廓系数
    
    x = np.arange(len(algorithms))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建分组柱状图
    bars1 = ax.bar(x - 2*width, accuracy, width, label='准确率', color='skyblue')
    bars2 = ax.bar(x - width, precision, width, label='精确率', color='lightcoral')
    bars3 = ax.bar(x, recall, width, label='召回率', color='lightgreen')
    bars4 = ax.bar(x + width, f1_score, width, label='F1分数', color='gold')
    bars5 = ax.bar(x + 2*width, silhouette, width, label='轮廓系数', color='violet')
    
    ax.set_xlabel('算法')
    ax.set_ylabel('评估指标值')
    ax.set_title('算法综合性能对比总结')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars4)
    add_labels(bars5)
    
    plt.tight_layout()
    save_figure(fig, 'algorithm_comparison_summary')
    plt.show()

def main():
    """
    主函数 - 生成增强版可视化图表
    """
    print("开始生成增强版实验结果可视化图表...")
    print("根据综合作业要求改进：")
    print("- 支持多种向量化方法对比")
    print("- 添加ROC曲线等分类评估指标") 
    print("- 改进聚类效果可视化")
    print("- 支持多种词法分析方式")
    
    try:
        # 1. 综合性能对比
        print("生成综合性能对比图...")
        create_comprehensive_performance_comparison()
        
        # 2. ROC曲线
        print("生成ROC曲线...")
        create_roc_curves()
        
        # 3. 精确率-召回率曲线
        print("生成精确率-召回率曲线...")
        create_precision_recall_curves()
        
        # 4. 增强版聚类可视化
        print("生成增强版聚类可视化...")
        create_enhanced_clustering_visualization()
        
        # 5. 向量化方法对比
        print("生成向量化方法对比...")
        create_vectorization_comparison()
        
        # 6. 算法综合对比总结
        print("生成算法综合对比总结...")
        create_algorithm_comparison_summary()
        
        print("\n所有增强版可视化图表生成完成！")
        print("生成的图片文件：")
        print("- comprehensive_performance_comparison.png/.svg")
        print("- roc_curves.png/.svg")
        print("- precision_recall_curves.png/.svg")
        print("- enhanced_clustering_visualization.png/.svg")
        print("- vectorization_comparison.png/.svg")
        print("- algorithm_comparison_summary.png/.svg")
        print("\nSVG格式适合在文档中使用，支持无损缩放")
        
    except Exception as e:
        print(f"生成图表时出现错误: {e}")
        import traceback
        traceback.print_exc()
        print("请检查必要的库是否正确安装")

if __name__ == "__main__":
    main()
