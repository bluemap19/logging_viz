import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
from typing import Tuple, List


def plot_horizontal_bar_charts(
        imp_result: pd.Series,
        figsize: Tuple[int, int] = (14, 10),
        title: str = 'Pearson Correlation Coefficient',
        X_lable='',
        Y_lable='',
):
    """
    绘制水平柱状图展示特征影响力结果

    参数:
    imp_result: 特征重要性结果 (Series)
    figsize: 图形大小
    title: 图标题
    """
    # 创建图形
    plt.figure(figsize=figsize)

    # 1. 皮尔逊系数水平柱状图
    plt.subplot(1, 1, 1)
    # 按值排序
    imp_sorted = imp_result.sort_values()
    # 绘制水平柱状图
    plt.barh(
        imp_sorted.index,
        imp_sorted.values,
        color='skyblue',
        edgecolor='darkblue'
    )

    # 添加标签
    for i, v in enumerate(imp_sorted.values):
        plt.text(
            v + 0.01, i,  # x位置为值+0.01，y位置为索引i
            f'{v:.3f}',
            color='black',
            va='center',
            fontsize=9,
        )

    plt.title(title, fontsize=14)
    plt.xlabel(X_lable, fontsize=12)
    plt.ylabel(Y_lable, fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def feature_influence_analysis(
        df_input: pd.DataFrame,
        input_cols: List[str],
        target_col: str,
        regressor_use: bool = False,
        n_estimators: int = 60,
        random_state: int = 42,
        figsize: Tuple[int, int] = (6, 8),
        replace_dict={},
        plot=True,
) -> Tuple[pd.Series, List[str], pd.Series, List[str]]:
    """
    分析输入属性对目标属性的影响力

    参数:
    df: 包含数据的DataFrame
    input_cols: 输入属性列名列表
    target_col: 目标属性列名
    n_estimators: 随机森林中树的数量
    random_state: 随机种子
    figsize: 图形大小

    返回:
    Pearson_result: 皮尔逊相关系数结果 (Series)
    input_cols_pearson: 按皮尔逊系数排序后的输入属性列表
    RF_result: 随机森林特征重要性结果 (Series)
    input_cols_RF: 按随机森林特征重要性排序后的输入属性列表
    """
    # 1. 数据准备和验证
    if not isinstance(df_input, pd.DataFrame):
        raise ValueError("输入数据必须是Pandas DataFrame格式")

    if not input_cols:
        raise ValueError("必须提供输入属性列名列表")

    if target_col not in df_input.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在")

    # 检查输入列是否存在
    missing_cols = [col for col in input_cols if col not in df_input.columns]
    if missing_cols:
        raise ValueError(f"以下输入列不存在: {', '.join(missing_cols)}")

    # if not replace_dict:
    #     replace_dict = {0:'T0', 1:'T1', 2:'T2', 3:'T3', 4:'T4', 5:'T5', 6:'T6', 7:'T7', 8:'T8'}

    # 2. 数据预处理
    # 复制数据以避免修改原始数据
    data = df_input[input_cols + [target_col]].copy()
    # 判断目标列是否是为Int类型数值，如果是的话，通过replace_dict进行替换
    if not regressor_use:
        if pd.api.types.is_numeric_dtype(data[target_col]):
            data[target_col] = data[target_col].astype(int)
            if replace_dict:
                data[target_col] = data[target_col].map(replace_dict)
    else:
        assert pd.api.types.is_numeric_dtype(data[target_col]), f"列{target_col}必须为数值列"

    # 处理缺失值
    data.dropna(inplace=True)

    # 3. 皮尔逊相关系数计算
    pearson_results = {}

    # 检查目标列类型
    if pd.api.types.is_numeric_dtype(data[target_col]):
        pass
    else:
        # 分类型目标 - 使用标签编码
        le_target = LabelEncoder()
        encoded_target = le_target.fit_transform(data[target_col])
        data[target_col] = encoded_target

    # 数值型目标 - 直接计算皮尔逊相关系数
    for col in input_cols:
        if pd.api.types.is_numeric_dtype(data[col]):
            corr, _ = pearsonr(data[col], data[target_col])
            pearson_results[col] = abs(corr)  # 取绝对值表示影响力大小
        else:
            # 非数值型输入列 - 使用标签编码
            le = LabelEncoder()
            encoded_col = le.fit_transform(data[col])
            corr, _ = pearsonr(encoded_col, data[target_col])
            pearson_results[col] = abs(corr)

    # 创建皮尔逊结果Series
    Pearson_result = pd.Series(pearson_results)
    # 按皮尔逊系数排序
    Pearson_result_sorted = Pearson_result.sort_values(ascending=False)
    input_cols_pearson = Pearson_result_sorted.index.tolist()

    # 4. 随机森林特征重要性计算
    # 准备特征和目标数据
    X = data[input_cols].copy()
    y = data[target_col].copy()

    # 处理分类变量
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # 选择模型类型，默认是使用分类树进行计算
    if regressor_use:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state
        )
    else:
        # 创建编码器字典用于分类特征
        encoders = {}
        for col in input_cols:
            if not pd.api.types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                encoders[col] = le

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

    # 训练模型
    model.fit(X, y)

    # 获取特征重要性
    feature_importances = model.feature_importances_
    RF_result = pd.Series(feature_importances, index=input_cols)
    # 按特征重要性排序
    RF_result_sorted = RF_result.sort_values(ascending=False)
    input_cols_RF = RF_result_sorted.index.tolist()

    # 5. 绘制结果
    if plot:
        # 使用水平柱状图
        plot_horizontal_bar_charts(
            imp_result=Pearson_result,
            figsize=figsize,
            title='Pearson Correlation Coefficient',
            X_lable='Absolute Correlation',
            Y_lable='Features'
        )
        plot_horizontal_bar_charts(
            imp_result=RF_result,
            figsize=figsize,
            title='Random Forest Feature Importance',
            X_lable='Importance Score',
            Y_lable='Features'
        )

    return Pearson_result, input_cols_pearson, RF_result, input_cols_RF


# 生成随机测试数据
def generate_test_data(n_samples=1000):
    """生成测试数据集"""
    # 创建数值型特征
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.uniform(0, 10, n_samples),
        'feature3': np.random.exponential(1, n_samples),
        'feature4': np.random.chisquare(5, n_samples),
    })

    # 创建分类特征
    categories = ['A', 'B', 'C', 'D']
    data['feature5'] = np.random.choice(categories, n_samples)

    # 创建目标变量 - 数值型
    data['target_numeric'] = (
            2 * data['feature1'] +
            0.5 * data['feature2'] -
            1.5 * data['feature3'] +
            np.random.normal(0, 1, n_samples)
    )

    # 创建目标变量 - 分类型
    # 基于特征创建分类目标
    conditions = [
        (data['feature1'] > 0) & (data['feature2'] > 5),
        (data['feature1'] <= 0) & (data['feature3'] > 1),
        (data['feature1'] > 0) & (data['feature2'] <= 5),
        (data['feature1'] <= 0) & (data['feature3'] <= 1)
    ]
    choices = ['Class1', 'Class2', 'Class3', 'Class4']
    data['target_category'] = np.select(conditions, choices, default='Class1')

    return data


if __name__ == '__main__':
    # 测试数值型目标
    print("=== 测试数值型目标 ===")
    test_data = generate_test_data()
    input_cols = ['feature1', 'feature2', 'feature3', 'feature4']
    target_col = 'target_category'
    # target_col = 'target_numeric'
    # target_col = 'feature5'

    print(test_data.describe())

    pearson_result, pearson_sorted, rf_result, rf_sorted = feature_influence_analysis(
        df_input=test_data,
        input_cols=input_cols,
        target_col=target_col,
        regressor_use=False,
        replace_dict={}
    )

    print("\n皮尔逊相关系数结果:", pearson_result)
    print("\n按皮尔逊系数排序的属性:", pearson_sorted)
    print("\n随机森林特征重要性结果:", rf_result)
    print("\n按随机森林特征重要性排序的属性:", rf_sorted)

    # # 测试分类型目标
    # print("\n=== 测试分类型目标 ===")
    # target_col = 'target_category'
    #
    # pearson_result, pearson_sorted, rf_result, rf_sorted = feature_influence_analysis(
    #     df=test_data,
    #     input_cols=input_cols,
    #     target_col=target_col
    # )
    #
    # print("\n皮尔逊相关系数结果:")
    # print(pearson_result)
    # print("\n按皮尔逊系数排序的属性:", pearson_sorted)
    # print("\n随机森林特征重要性结果:")
    # print(rf_result)
    # print("\n按随机森林特征重要性排序的属性:", rf_sorted)