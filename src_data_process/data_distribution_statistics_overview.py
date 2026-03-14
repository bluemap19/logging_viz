import numpy as np
import pandas as pd
from scipy.stats import norm

# 主要作用是对数据分布进行，可叙述化统计，并转换成相对应文本
def data_overview(df: pd.DataFrame = pd.DataFrame(),
                        input_names: list = [],
                        target_col='',
                        target_col_dict: dict = {}):
    """
        对数据进行多维度统计分析并生成可叙述化报告

        参数:
        df: 输入数据DataFrame
        input_names: 需要分析的数值特征列名列表
        target_col: 分类标签列名
        target_col_dict: 分类标签映射字典（编码->可读名称）
        返回:
        包含详细统计信息的DataFrame
    """
    # 1. 输入验证
    if df.empty:
        raise ValueError("输入DataFrame不能为空")
    if not input_names:
        raise ValueError("必须指定分析的特征列")
    if not target_col:
        raise ValueError("必须指定分类标签列")
    # if not target_col_dict:
    #     print("警告：未提供分类标签映射字典，将使用原始编码")
    #     target_col_dict = {k: str(k) for k in df[target_col].unique()}
    # if set(list(target_col_dict.keys())) != set(list(df[target_col].unique())):
    #     print('target_col_dict 可叙述化词典设置错误')
    #     print(list(target_col_dict.keys()), list(df[target_col].unique()))
    #     raise ValueError("target_col_dict.key()必须等于input_names")
    # 获取目标列的唯一值
    unique_target_values = df[target_col].unique().astype(int)
    # 修复检查逻辑：键应该与目标列的唯一值匹配
    target_keys = set(target_col_dict.keys())
    unique_target_set = set(unique_target_values)

    # 检查是否所有目标值都有对应的键
    if not target_keys.issubset(unique_target_set):
        print('target_col_dict 可叙述化词典设置错误')
        print(f"字典键: {list(target_col_dict.keys())}")
        print(f"目标列唯一值: {list(unique_target_values)}")

        # 找出缺失的键
        missing_keys = unique_target_set - target_keys
        print(f"缺失的键: {list(missing_keys)}")

        # 找出多余的键
        extra_keys = target_keys - unique_target_set
        print(f"多余的键: {list(extra_keys)}")

        raise ValueError("target_col_dict的键必须包含目标列的所有唯一值")


    # 2. 准备结果容器
    results = []

    # 3. 按类别分组分析
    grouped = df.groupby(target_col)

    # 预计算整体统计
    overall_stats_dict = {}
    for feature in input_names:
        if pd.api.types.is_numeric_dtype(df[feature]):
            data = df[feature].dropna()
            overall_stats_dict[feature] = calculate_stats(data, '总体', feature)

    # 按类别分析
    for class_code, group in grouped:
        class_name = target_col_dict.get(class_code, str(class_code))

        for feature in input_names:
            if not pd.api.types.is_numeric_dtype(df[feature]):
                continue

            # 获取特征数据
            data = group[feature].dropna()

            # 计算统计量
            class_stats = calculate_stats(data, class_name, feature)
            class_stats['类别编码'] = class_code

            # 获取整体统计
            overall_stats = overall_stats_dict.get(feature)

            # 生成智能叙述
            if overall_stats is not None:
                narrative = generate_narrative(class_stats, overall_stats)
            else:
                narrative = generate_narrative(class_stats, class_stats)  # 使用自身作为整体

            class_stats['叙述性描述'] = narrative

            # 添加到结果
            results.append(class_stats)

    # 4. 添加总体分析
    for feature, stats in overall_stats_dict.items():
        stats['叙述性描述'] = (
            f"总体来看，特征'{feature}'的分布情况："
            f"均值{stats['均值']:.2f}，标准差{stats['标准差']:.2f}。"
        )
        results.append(stats)

    # 5. 创建结果DataFrame
    result_df = pd.DataFrame(results)

    # 设置多级索引
    result_df.set_index(['类别名称', '特征'], inplace=True)

    return result_df


def calculate_stats(data: pd.Series, class_name: str, feature: str) -> dict:
    """
    计算一组数据的统计指标

    参数:
    data: 输入数据
    class_name: 类别名称
    feature: 特征名称

    返回:
    包含统计指标的字典
    """
    # 基本统计量
    mean_val = np.mean(data)
    median_val = np.median(data)
    variance_val = np.var(data, ddof=1)
    std_val = np.std(data, ddof=1)

    # 高斯拟合参数
    try:
        mu, std = norm.fit(data)
    except:
        mu, std = mean_val, std_val

    # 分位数
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    min_val = np.min(data)
    max_val = np.max(data)

    # 偏度和峰度
    skewness = data.skew()
    kurtosis = data.kurtosis()

    return {
        '类别名称': class_name,
        '特征': feature,
        '样本数': len(data),
        '均值': mean_val,
        '中位数': median_val,
        '方差(S2)': variance_val,
        '标准差': std_val,
        '高斯拟合μ': mu,
        '高斯拟合δ': std,
        '偏度': skewness,
        '峰度': kurtosis,
        '最小值': min_val,
        '25%分位数': q25,
        '75%分位数': q75,
        '最大值': max_val,
        'IQR': iqr
    }


def generate_narrative(class_stats: pd.Series, overall_stats: pd.Series) -> str:
    """
    生成智能叙述性描述文本

    参数:
    class_stats: 单个类别的统计结果 (Series)
    overall_stats: 整体数据的统计结果 (Series)

    返回:
    叙述性描述文本
    """
    # 获取基本信息
    class_name = class_stats['类别名称']
    feature = class_stats['特征']
    class_mean = class_stats['均值']
    overall_mean = overall_stats['均值']

    # 1. 描述相对位置
    relative_position = describe_relative_position(class_mean, overall_mean)

    # 2. 描述离散程度
    dispersion = describe_dispersion(class_stats)

    # 3. 描述与整体对比
    comparison = describe_comparison(class_stats, overall_stats)

    # 4. 组合完整描述
    narrative = (
        f"在{class_name}类别中，特征'{feature}'的分布呈现以下特点："
        f"{relative_position}，{dispersion}。"
        f"{comparison}"
    )

    return narrative


# 辅助函数：描述相对位置
def describe_relative_position(class_mean: float, overall_mean: float) -> str:
    """描述类别均值相对于整体均值的位置"""
    diff = class_mean - overall_mean
    abs_diff = abs(diff)
    ratio = abs_diff / overall_mean if overall_mean != 0 else abs_diff

    # # 定义位置描述
    # if ratio < 0.05:
    #     # position = "处于整体平均水平"
    #     position = "中等"
    # elif ratio < 0.15:
    #     # position = "略" + ("高于" if diff > 0 else "低于") + "整体平均水平"
    #     position = "中" + ("高" if diff > 0 else "低")
    # elif ratio < 0.3:
    #     # position = "明显" + ("高于" if diff > 0 else "低于") + "整体平均水平"
    #     position = "偏" + ("高" if diff > 0 else "低") + "整体平均水平"
    # else:
    #     position = "显著" + ("高于" if diff > 0 else "低于") + "整体平均水平"

    # 定义位置描述
    if ratio < 0.05:
        position = "中等"
    elif ratio < 0.3:
        position = "中" + ("高" if diff > 0 else "低")
    else:
        position = "偏" + ("高" if diff > 0 else "低")

    # 添加具体数值
    return f"均值{class_mean:.2f}，{position}，整体均值{overall_mean:.2f}"


# 辅助函数：描述离散程度
def describe_dispersion(stats: pd.Series) -> str:
    """描述数据的离散程度"""
    std = stats['标准差']
    iqr = stats['IQR']
    min_val = stats['最小值']
    max_val = stats['最大值']

    # 基于标准差和IQR的离散程度
    std_description = describe_by_std(std)
    # 基于范围的离散程度
    range_description = describe_by_range(min_val, max_val)

    return f"数据{std_description}，{range_description}"


def describe_by_std(std: float) -> str:
    """基于标准差描述离散程度"""
    if std < 0.1:
        return "高度集中"
    elif std < 0.3:
        return "较为集中"
    elif std < 0.5:
        return "中等离散"
    elif std < 1.0:
        return "较为分散"
    else:
        return "高度分散"


def describe_by_range(min_val: float, max_val: float) -> str:
    """基于范围描述离散程度"""
    data_range = max_val - min_val
    if data_range < 0.1:
        return "变化范围极小"
    elif data_range < 0.5:
        return "变化范围较小"
    elif data_range < 1.0:
        return "变化范围适中"
    elif data_range < 2.0:
        return "变化范围较大"
    else:
        return "变化范围极大"


# 辅助函数：描述与整体对比
def describe_comparison(class_stats: pd.Series, overall_stats: pd.Series) -> str:
    """描述类别与整体的对比"""
    comparisons = []

    # 1. 均值对比
    mean_diff = class_stats['均值'] - overall_stats['均值']
    mean_ratio = abs(mean_diff) / overall_stats['均值'] if overall_stats['均值'] != 0 else abs(mean_diff)

    if mean_ratio > 0.3:
        comparisons.append(f"均值显著{'高于' if mean_diff > 0 else '低于'}整体水平")
    elif mean_ratio > 0.15:
        comparisons.append(f"均值明显{'高于' if mean_diff > 0 else '低于'}整体水平")
    elif mean_ratio > 0.05:
        comparisons.append(f"均值略{'高于' if mean_diff > 0 else '低于'}整体水平")

    # 2. 离散程度对比
    std_ratio = class_stats['标准差'] / overall_stats['标准差']

    if std_ratio > 1.5:
        comparisons.append("离散程度显著高于整体")
    elif std_ratio > 1.2:
        comparisons.append("离散程度明显高于整体")
    elif std_ratio < 0.67:
        comparisons.append("离散程度显著低于整体")
    elif std_ratio < 0.83:
        comparisons.append("离散程度明显低于整体")

    # 组合描述
    if comparisons:
        return "与整体数据相比，" + "，".join(comparisons) + "。"
    return "与整体数据相比，特征分布基本一致。"



if __name__ == '__main__':
    # 示例数据
    data = {
        'feature1': np.random.normal(50, 10, 100),
        'feature2': np.random.exponential(20, 100),
        'feature3': np.random.uniform(0, 100, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    }
    df = pd.DataFrame(data)

    # 类别映射
    category_dict = {'A': '类别A', 'B': '类别B', 'C': '类别C'}

    # 执行分析
    result = data_overview(
        df=df,
        input_names=['feature1', 'feature2', 'feature3'],
        target_col='category',
        target_col_dict=category_dict
    )

    # 查看结果
    print(result.head())

    # 保存结果
    result.to_excel('data_overview.xlsx', index=True)

    # 提取叙述性描述
    narratives = result['叙述性描述'].dropna()
    for narrative in narratives:
        print(narrative)
        print('-' * 180)