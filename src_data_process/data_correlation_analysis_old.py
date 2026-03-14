import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.utils import check_X_y


def random_forest_correlation_analysis(data_all, random_seed=44, plot_index=[2, 2],
                                       figsize=(16, 5), tree_num=10, Norm=True):
    """
    使用随机森林分析输入变量与输出类别之间的相关性

    参数:
    data_all : DataFrame
        包含输入自变量和输出类别的数据集 (M行 x N列)
        前N-1列为输入变量，最后一列为输出类别

    random_seed : int, optional (默认=44)
        随机种子，确保结果可重现

    plot_index : list, optional (默认=[2, 2])
        绘图布局配置 (暂时不使用，保留为接口参数)

    figsize : tuple, optional (默认=(16, 5))
        图形大小配置 (暂时不使用，保留为接口参数)

    tree_num : int, optional (默认=10)
        随机森林中决策树的数量

    Norm: bool, optional (默认=True)
        是否对输入数据进行归一化

    返回:
    scores : ndarray
        交叉验证结果 (5折交叉验证得分)

    class_accuracy : dict
        每个类别的正确率 (召回率)

    auc_total : float
        总体ROC AUC得分 (使用'ovr'多分类策略)

    importances : ndarray
        输入变量的重要性权重 (按原始列顺序)
    """
    # 1. 数据准备和验证
    if not isinstance(data_all, pd.DataFrame):
        raise ValueError("输入数据必须是Pandas DataFrame格式")

    if len(data_all.columns) < 2:
        raise ValueError("数据必须包含至少一个输入变量和一个输出变量")

    # 分离输入变量(X)和输出变量(y)
    X = data_all.iloc[:, :-1].values
    y = data_all.iloc[:, -1].values
    feature_names = data_all.columns[:-1].tolist()

    # 验证数据形状
    try:
        X, y = check_X_y(X, y, multi_output=True)
    except Exception as e:
        raise ValueError(f"数据形状验证失败: {str(e)}")

    # 2. 数据预处理
    if Norm:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=random_seed
    )

    # 4. 建立和训练随机森林模型
    rf_model = RandomForestClassifier(n_estimators=tree_num, random_state=random_seed)

    # 5折交叉验证
    scores = cross_val_score(rf_model, X, y, cv=5)

    # 训练模型
    rf_model.fit(X_train, y_train)

    # 5. 模型评估
    y_pred = rf_model.predict(X_test)

    # # 输出分类报告和混淆矩阵
    # print("分类报告:\n", classification_report(y_test, y_pred))
    # print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))

    # # 计算回归指标 (虽为分类问题，但保留这些指标作为参考)
    # print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
    # print(f"R²: {r2_score(y_test, y_pred):.4f}")

    # 6. 计算每个类别的正确率
    classes = np.unique(y)
    cm = confusion_matrix(y_test, y_pred)
    class_accuracy = {}

    for i, cls in enumerate(classes):
        true_positive = cm[i, i]
        actual_total = cm[i, :].sum()
        class_accuracy[cls] = true_positive / actual_total if actual_total != 0 else 0.0

    # 7. 计算ROC AUC分数
    y_test_bin = label_binarize(y_test, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)
    auc_total = roc_auc_score(y_test_bin, y_pred_bin, multi_class='ovr')
    # print(f"总体AUC分数: {auc_total:.4f}")

    # 8. 获取特征重要性
    importances = rf_model.feature_importances_
    # print("\n特征重要性:")
    # for name, importance in zip(feature_names, importances):
    #     print(f"{name}: {importance:.4f}")

    # 9. 返回分析结果
    return scores, class_accuracy, auc_total, importances





if __name__ == '__main__':
    DATA_NUM = 1000
    # 示例数据 (实践中使用你的真实DataFrame)
    data = pd.DataFrame({
        'feature1': np.random.rand(DATA_NUM),
        'feature2': np.random.rand(DATA_NUM),
        'feature3': np.random.rand(DATA_NUM),
        'target': np.random.randint(0, 3, DATA_NUM)  # 3个类别
    })

    # 运行分析
    scores, accuracies, auc_score, importances = random_forest_correlation_analysis(
        data_all=data,
        random_seed=42,
        tree_num=20
    )

    # 输出结果
    print("交叉验证分数:", scores)
    print("类别准确率:", accuracies)
    print("AUC总分:", auc_score)
    print("特征重要性:", importances)