import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from src_data_process.data_balanace import smart_balance_dataset
from src_data_process.data_supervised_evaluation import evaluate_supervised_clustering

# 设置支持中文的字体，使用黑体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


def supervised_classification(X: pd.DataFrame, y: pd.Series, Norm=False, Type_str={'岩相1':0, '岩相2':1, '岩相3':2, '岩相4':3, '岩相5':4}, y_type_number=5):
    # data_balanced = smart_balance_dataset(pd.concat([X, y], axis=1), target_col=y.name, method='smote', Type_dict=Type_str)
    # X = data_balanced.iloc[:, :-1]
    # y = data_balanced.iloc[:, -1]
    y = y.astype(int)

    # 数据预处理
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # 定义分类器集合
    classifiers = {
        "MLP": make_pipeline(
            StandardScaler() if not Norm else FunctionTransformer(validate=False),  # 避免重复标准化
            MLPClassifier(
                hidden_layer_sizes=(16, 16, 8),  # 增大网络容量
                alpha=1e-4,  # 添加L2正则化
                learning_rate='adaptive',  # 自适应学习率
                max_iter=500,  # 增加迭代次数
                early_stopping=True,
                random_state=42
            )
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=y_type_number)),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
        ),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Random Forest": RandomForestClassifier(
            n_estimators=10,
            class_weight='balanced',
            random_state=42
        ),
        "GBM": GradientBoostingClassifier(
            n_estimators=10,
            subsample=0.8,
            random_state=42
        )
    }

    # 模型训练与评估
    results = []
    for name, clf in classifiers.items():
        # 统一训练流程
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # 评估指标计算
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # 存储混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        # 计算每个类别的正确预测数（对角线元素）
        correct_predictions = np.diag(cm)
        # 计算每个类别的总样本数（行和）
        total_samples_per_class = cm.sum(axis=1)
        # 计算每个类别的正确率
        per_class_accuracy = correct_predictions / total_samples_per_class
        # 处理可能的除零错误（当某类别无真实样本时）
        per_class_accuracy = np.nan_to_num(per_class_accuracy, nan=0.0)

        result_base = {
            "Model": name,
            "Accuracy": acc,
            "Precision": report['macro avg']['precision'],
            "Recall": report['macro avg']['recall'],
            "F1": report['macro avg']['f1-score'],
        }
        type_str = list(Type_str.keys())
        for i in range(per_class_accuracy.shape[0]):
            # result_base['ACC_' + type_str[i]] = per_class_accuracy[i]
            result_base[type_str[i]] = per_class_accuracy[i]
        results.append(result_base)

    df_results = pd.DataFrame(results).set_index('Model')
    return df_results, classifiers


def model_predict(classifiers: dict, X: pd.DataFrame) -> pd.DataFrame:
    """
    模型批量预测接口
    参数：
    classifiers : dict
        训练好的模型字典，格式为 {"模型名称": 模型对象}
    X : pd.DataFrame
        待预测数据，需确保特征与训练时一致
    返回：
    pd.DataFrame
        预测结果矩阵，每列对应不同模型的预测结果
    """
    # 初始化结果容器
    predictions = pd.DataFrame(index=X.index)

    # 遍历模型进行预测
    for model_name, clf in classifiers.items():
        try:
            # 执行预测（Pipeline自动处理预处理）
            y_pred = clf.predict(X)

            # 存储预测结果
            predictions[model_name] = y_pred.astype(int)

        except Exception as e:
            print(f"模型 {model_name} 预测失败: {str(e)}")
            predictions[model_name] = np.nan  # 标记异常预测

    # 数据验证
    if predictions.isnull().sum().sum() > 0:
        warnings.warn("部分模型预测存在缺失值，请检查输入数据与模型兼容性")

    return predictions


if __name__ == '__main__':
    # 设置随机种子以确保可重复性
    np.random.seed(42)

    # 1. 创建一个简单的测试数据集
    # 生成100个样本，5个特征，4个类别
    n_samples = 100
    n_features = 5
    n_classes = 4

    # 创建特征数据
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'Feature_{i + 1}' for i in range(n_features)]
    )

    # 创建目标变量（4个类别）
    # 使类别分布稍微不平衡，以便测试
    y_values = np.array([0] * 20 + [1] * 30 + [2] * 25 + [3] * 25)  # 0:20, 1:30, 2:25, 3:25
    np.random.shuffle(y_values)
    y = pd.Series(y_values, name='岩相')

    # 设置类别标签映射
    type_str = {'岩相1': 0, '岩相2': 1, '岩相3': 2, '岩相4': 3}

    print("测试数据集信息:")
    print(f"特征形状: {X.shape}")
    print(f"目标变量分布:\n{y.value_counts().sort_index()}")

    # 2. 测试 supervised_classification 函数
    print("\n" + "=" * 50)
    print("测试 supervised_classification 函数")
    print("=" * 50)

    df_results, trained_classifiers = supervised_classification(
        X, y,
        Norm=False,  # 不跳过标准化
        Type_str=type_str,
        y_type_number=n_classes
    )

    print("分类结果汇总:")
    print(df_results)
    print(f"\n训练的模型数量: {len(trained_classifiers)}")
    print(f"模型列表: {list(trained_classifiers.keys())}")

    # 3. 测试 model_predict 函数
    print("\n" + "=" * 50)
    print("测试 model_predict 函数")
    print("=" * 50)

    # 使用前20个样本进行预测
    X_sample = X.iloc[:20]
    predictions = model_predict(trained_classifiers, X_sample)

    print("预测结果 (前20个样本):")
    print(predictions.head())

    # 验证预测结果
    print(f"\n预测结果形状: {predictions.shape}")
    print("各模型预测的类别分布:")
    for model in predictions.columns:
        if not predictions[model].isna().all():
            print(f"{model}: {predictions[model].value_counts().to_dict()}")

    # 4. 测试 evaluate_supervised_clustering 函数
    print("\n" + "=" * 50)
    print("测试 evaluate_supervised_clustering 函数")
    print("=" * 50)

    # 创建一个包含真实标签和多个预测标签的DataFrame
    # 先生成一些模拟的预测结果
    np.random.seed(123)  # 重置随机种子

    # 创建测试DataFrame
    df_eval = pd.DataFrame({
        'true_labels': y,  # 真实标签
        'RF_predictions': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.2, 0.3, 0.25, 0.25]),  # 随机预测
        'SVM_predictions': np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.2, 0.3, 0.25, 0.25])  # 随机预测
    })

    # 为了让结果看起来更合理，我们让一部分预测与真实标签一致
    df_eval['RF_predictions'] = np.where(
        np.random.random(n_samples) > 0.7,  # 30%的概率使用真实标签
        df_eval['true_labels'],
        df_eval['RF_predictions']
    )

    df_eval['SVM_predictions'] = np.where(
        np.random.random(n_samples) > 0.6,  # 40%的概率使用真实标签
        df_eval['true_labels'],
        df_eval['SVM_predictions']
    )

    print("评估数据框 (前10行):")
    print(df_eval.head(10))

    # 执行评估
    try:
        evaluation_results = evaluate_supervised_clustering(
            df=df_eval,
            col_org='true_labels',
            cols_compare=['RF_predictions', 'SVM_predictions'],
            save_report=False,  # 设置为True可保存Excel报告
            report_path="test_evaluation.xlsx"
        )

        print("\n评估结果:")
        for model, metrics in evaluation_results.items():
            print(f"\n{model}:")
            print(f"  准确率: {metrics['accuracy']:.4f}")
            print(f"  加权F1分数: {metrics['f1_weighted']:.4f}")
            print(f"  Cohen Kappa: {metrics['cohen_kappa']:.4f}")

    except Exception as e:
        print(f"评估过程中出错: {e}")

    # 5. 完整流程测试
    print("\n" + "=" * 50)
    print("完整流程测试")
    print("=" * 50)

    # 测试1: 使用不同的参数
    print("测试1: Norm=True (跳过标准化)")
    df_results_norm, _ = supervised_classification(
        X, y,
        Norm=True,  # 跳过标准化
        Type_str=type_str,
        y_type_number=n_classes
    )
    print("MLP模型准确率 (Norm=True):", df_results_norm.loc['MLP', 'Accuracy'])

    # 测试2: 小样本测试
    print("\n测试2: 小样本测试 (20个样本)")
    X_small = X.iloc[:20]
    y_small = y.iloc[:20]
    try:
        df_results_small, classifiers_small = supervised_classification(
            X_small, y_small,
            Type_str=type_str,
            y_type_number=min(4, len(np.unique(y_small)))  # 动态设置KNN的k值
        )
        print("小样本测试完成")
    except Exception as e:
        print(f"小样本测试出错: {e}")

    # 测试3: 边界值测试
    print("\n测试3: 边界值测试")

    # 测试单一类别
    X_single = X.iloc[:10]
    y_single = pd.Series([0] * 10, name='岩相')  # 所有样本都属于同一类别
    print("单一类别测试:")
    try:
        df_results_single, _ = supervised_classification(
            X_single, y_single,
            Type_str={'岩相1': 0},
            y_type_number=1
        )
        print("单一类别测试完成")
    except Exception as e:
        print(f"单一类别测试出错: {e}")

    # 测试4: 模型预测函数的异常处理
    print("\n测试4: 模型预测函数的异常处理")
    empty_classifiers = {}
    try:
        empty_predictions = model_predict(empty_classifiers, X_sample)
        print("空模型字典预测结果:", empty_predictions.shape)
    except Exception as e:
        print(f"空模型字典预测出错: {e}")

    print("\n" + "=" * 50)
    print("所有测试完成!")
    print("=" * 50)

    pass