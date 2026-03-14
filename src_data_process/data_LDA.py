import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)


def discriminant_classifier(data_all, col_input, col_target, Norm=False):
    """
    判别式分类接口

    参数:
    data_all: DataFrame, 包含所有数据
    col_input: list, 输入特征列名 ['col1', 'col2', ...]
    col_target: str, 目标类别列名
    Norm: bool, 是否进行特征标准化

    返回:
    discriminant_coef: ndarray, 判别式系数矩阵 (n_classes, n_features)
    class_means: ndarray, 各类别均值向量
    class_labels: ndarray, 类别标签
    metrics: dict, 包含准确率、精确率、召回率等评估指标
    """
    # 1. 数据预处理
    # 提取特征和目标变量
    X = data_all[col_input].values
    y = data_all[col_target].values

    # 获取唯一类别标签
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    n_features = len(col_input)

    # 特征标准化
    if Norm:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # 2. 数据分割 (80%训练, 20%测试)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. 创建并训练判别式分类模型
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # 4. 模型评估
    # 训练集预测
    train_pred = lda.predict(X_train)

    # 测试集预测
    test_pred = lda.predict(X_test)

    # 计算评估指标
    metrics = {
        'train_accuracy': accuracy_score(y_train, train_pred),
        'test_accuracy': accuracy_score(y_test, test_pred),
        'train_precision': precision_score(y_train, train_pred, average='macro'),
        'test_precision': precision_score(y_test, test_pred, average='macro'),
        'train_recall': recall_score(y_train, train_pred, average='macro'),
        'test_recall': recall_score(y_test, test_pred, average='macro'),
        'train_f1': f1_score(y_train, train_pred, average='macro'),
        'test_f1': f1_score(y_test, test_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test, test_pred),
        'classification_report': classification_report(y_test, test_pred)
    }

    # 5. 获取判别式系数矩阵和类别均值向量
    # 判别式系数矩阵形状为 (n_classes, n_features)
    discriminant_coef = lda.coef_

    # 各类别均值向量
    class_means = lda.means_

    # 6. 打印结果
    print(f"类别数量: {n_classes}")
    print(f"特征数量: {n_features}")
    print(f"训练集准确率: {metrics['train_accuracy']:.4f}")
    print(f"测试集准确率: {metrics['test_accuracy']:.4f}")
    print(f"训练集宏平均精确率: {metrics['train_precision']:.4f}")
    print(f"测试集宏平均精确率: {metrics['test_precision']:.4f}")
    print(f"训练集宏平均召回率: {metrics['train_recall']:.4f}")
    print(f"测试集宏平均召回率: {metrics['test_recall']:.4f}")
    print(f"训练集宏平均F1分数: {metrics['train_f1']:.4f}")
    print(f"测试集宏平均F1分数: {metrics['test_f1']:.4f}")
    print("\n混淆矩阵:")
    print(metrics['confusion_matrix'])
    print("\n分类报告:")
    print(metrics['classification_report'])
    print(f"判别式系数矩阵形状: {discriminant_coef.shape}")

    return discriminant_coef, class_means, unique_classes, metrics


def apply_discriminant_classifier(data_all, col_input, discriminant_coef, class_means, class_labels, Norm=False):
    """
    应用判别式分类器进行预测

    参数:
    data_all: DataFrame, 包含需要预测的数据
    col_input: list, 输入特征列名 ['col1', 'col2', ...]
    discriminant_coef: ndarray, 判别式系数矩阵 (n_classes, n_features)
    class_means: ndarray, 各类别均值向量
    class_labels: ndarray, 类别标签
    Norm: bool, 是否进行特征标准化

    返回:
    data_all: DataFrame, 添加了预测类别(predicted_type)的新数据框
    """
    # 1. 数据预处理
    # 提取特征
    X = data_all[col_input].values

    # 特征标准化
    if Norm:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # 2. 计算判别函数值
    # 判别函数公式: f_i(x) = w_i^T * x + b_i
    # 其中 w_i 是第i类的系数向量，b_i 是偏置项
    discriminant_values = X @ discriminant_coef.T

    # 3. 计算预测类别
    # 找到判别函数值最大的类别
    max_indices = np.argmax(discriminant_values, axis=1)

    # 将索引映射回原始类别标签
    predicted_labels = class_labels[max_indices]

    # 添加预测类别列
    data_all['predicted_type'] = predicted_labels

    # 4. 输出结果信息
    print(f"成功添加预测类别列(predicted_type)")

    return data_all


if __name__ == '__main__':
    # 示例数据 - 5个类别
    np.random.seed(42)
    n_samples = 1000
    n_features = 3
    n_classes = 5

    # 创建有区分度的数据
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_classes):
        start_idx = i * (n_samples // n_classes)
        end_idx = (i + 1) * (n_samples // n_classes)
        if i == n_classes - 1:
            end_idx = n_samples

        # 为每个类别创建有区分度的特征
        X[start_idx:end_idx, 0] = np.random.normal(i * 2, 1, end_idx - start_idx)
        X[start_idx:end_idx, 1] = np.random.normal(i * 3, 1.5, end_idx - start_idx)
        X[start_idx:end_idx, 2] = np.random.normal(i * 1.5, 0.8, end_idx - start_idx)
        y[start_idx:end_idx] = i

    # 创建DataFrame
    data = pd.DataFrame(X, columns=[f'feature{j + 1}' for j in range(n_features)])
    data['target'] = y

    # 调用接口
    discriminant_coef, class_means, class_labels, metrics = discriminant_classifier(
        data_all=data,
        col_input=['feature1', 'feature2', 'feature3'],
        col_target='target',
        Norm=True
    )

    print("\n判别式系数矩阵:")
    print(discriminant_coef)
    print("\n类别均值向量:")
    print(class_means)
    print("\n类别标签:")
    print(class_labels)

    # 新数据预测
    data_2 = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
    })

    result = apply_discriminant_classifier(
        data_2,
        ['feature1', 'feature2', 'feature3'],
        discriminant_coef,
        class_means,
        class_labels,
        Norm=True
    )

    print("\n预测结果示例:")
    print(result.head())
    print("\n预测类别分布:")
    print(result['predicted_type'].value_counts())