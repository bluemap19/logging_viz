import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def fisher_discriminant_analysis(data_all, col_input, col_target):
    """
    Fisher判别分析接口 - 优化版

    参数:
    data_all: DataFrame, 包含所有数据
    col_input: list, 输入特征列名 ['col1', 'col2', ...]
    col_target: str, 目标类别列名
    Norm: bool, 是否进行特征标准化

    返回:
    fisher_coef: ndarray, Fisher系数矩阵 (n_classes-1, n_features)
    class_centers: ndarray, 各类别在判别空间中的中心点
    class_labels: ndarray, 类别标签
    train_accuracy: float, 训练集准确率
    test_accuracy: float, 测试集准确率
    scaler: StandardScaler or None, 标准化器对象（如果Norm=True）
    coef: ndarray, 分类系数矩阵 (n_classes, n_features)
    intercept: ndarray, 截距向量 (n_classes,)
    """
    # 1. 数据预处理
    X = data_all[col_input].values
    y = data_all[col_target].values
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # 训练时拟合标准化器

    # 2. 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # 3. 创建并训练Fisher判别模型
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # 4. 模型评估
    train_pred = lda.predict(X_train)
    test_pred = lda.predict(X_test)
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    # 5. 获取模型参数
    fisher_coef = lda.scalings_.T  # Fisher系数矩阵 (n_classes-1, n_features)
    class_centers = lda.means_ @ lda.scalings_  # 判别空间中的类中心
    # 处理二分类特殊情况
    if n_classes == 2:
        # 二分类时coef和intercept只有一行
        coef = np.vstack([-lda.coef_, lda.coef_])  # 创建两行系数
        intercept = np.array([-lda.intercept_, lda.intercept_]).flatten()  # 创建两个截距
    else:
        coef = lda.coef_  # 分类系数 (n_classes, n_features)
        intercept = lda.intercept_  # 截距 (n_classes,)
    # coef = lda.coef_  # 分类系数 (n_classes, n_features)
    # intercept = lda.intercept_  # 截距 (n_classes,)

    # 6. 打印结果
    print(f"类别数量: {n_classes}")
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"Fisher系数矩阵形状: {fisher_coef.shape}")
    print("Fisher系数矩阵:")
    for i, row in enumerate(fisher_coef):
        print(f"判别方向 {i + 1}: {row}")

    return fisher_coef, class_centers, unique_classes, train_accuracy, test_accuracy, scaler, coef, intercept

def fisher_apply(data_input, col_input, fisher_coef, class_centers, class_labels, scaler, coef, intercept):
    """
    应用Fisher判别函数进行特征投影和类别预测 - 优化版

    参数:
    data_all: DataFrame, 包含需要预测的数据
    col_input: list, 输入特征列名 ['col1', 'col2', ...]
    fisher_coef: ndarray, Fisher系数矩阵 (n_classes-1, n_features)
    class_centers: ndarray, 各类别在判别空间中的中心点
    class_labels: ndarray, 类别标签
    scaler: StandardScaler or None, 训练时的标准化器
    coef: ndarray, 分类系数矩阵 (n_classes, n_features)
    intercept: ndarray, 截距向量 (n_classes,)
    Norm: bool, 是否进行特征标准化

    返回:
    data_all: DataFrame, 添加了投影特征(F1, F2,...)和判别分数(D_class0, D_class1,...)的新数据框
    """
    # 1. 数据预处理
    data_all = data_input.copy()
    X = data_all[col_input].values

    if scaler is None:
        raise ValueError("Norm is True but scaler is None. Ensure scaler is passed from training.")
    X = scaler.transform(X)  # 使用训练时的scaler进行转换，确保一致性

    # 2. 计算Fisher投影特征 (K-1个特征)
    fisher_projections = X @ fisher_coef.T  # 投影到判别空间
    n_components = fisher_coef.shape[0]
    for i in range(n_components):
        # data_all[f'F{i + 1}'] = fisher_projections[:, i]
        data_all.loc[:, f'F{i + 1}'] = fisher_projections[:, i]

    # # 3. 计算判别分数 (K个分数，每个类别一个)
    # decision_scores = X @ coef.T + intercept  # 决策函数: (n_samples, n_classes)
    # for i, cls in enumerate(class_labels):
    #     data_all[f'D_{cls}'] = decision_scores[:, i]  # 添加判别分数列
    # 3. 计算判别分数 (K个分数，每个类别一个)
    decision_scores = X @ coef.T + intercept  # 决策函数: (n_samples, n_classes)
    # 确保decision_scores是二维数组
    if decision_scores.ndim == 1:
        decision_scores = decision_scores.reshape(-1, 1)
    # 添加判别分数列
    for i, cls in enumerate(class_labels):
        # data_all[f'D_{cls}'] = decision_scores[:, i]
        data_all.loc[:, f'D_{cls}'] = decision_scores[:, i]

    # 4. 预测类别 (使用决策分数，替代距离计算)
    predicted_labels = class_labels[np.argmax(decision_scores, axis=1)]
    # data_all['Type_Fisher'] = predicted_labels
    data_all.loc[:, 'Type_Fisher'] = predicted_labels

    # 5. 输出结果信息
    print(f"成功添加 {n_components} 个投影特征(F1-F{n_components})")
    print(f"成功添加 {len(class_labels)} 个判别分数(D_class0-D_class{len(class_labels)-1})")
    print(f"成功添加预测类别列(predicted_type)")

    return data_all

if __name__ == '__main__':
    # 示例数据 - 6个类别
    data = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
        'target': np.random.choice([0, 1], size=1000)
    })

    # 调用接口，接收额外返回值
    fisher_coef, class_centers, class_labels, train_acc, test_acc, scaler, coef, intercept = fisher_discriminant_analysis(
        data_all=data,
        col_input=['feature1', 'feature2', 'feature3'],
        col_target='target',
    )

    print("\nFisher系数矩阵:")
    print(fisher_coef)
    print("\n类别中心点:")
    print(class_centers)
    print("\n类别标签:")
    print(class_labels)
    print("\n分类系数:")
    print(coef)
    print("\n截距:")
    print(intercept)

    # 新数据预测
    data_2 = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'feature3': np.random.randn(1000),
    })

    result = fisher_apply(
        data_2,
        ['feature1', 'feature2', 'feature3'],
        fisher_coef,
        class_centers,
        class_labels,
        scaler,
        coef,
        intercept,
    )

    print("\n预测结果示例:")
    print(result.head())
    print("\n预测类别分布:")
    print(result['Type_Fisher'].value_counts())
    print("\n判别分数示例 (D_class0 到 D_class5):")
    print(result[[f'D_{cls}' for cls in class_labels]].head())