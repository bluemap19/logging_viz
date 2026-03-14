import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs


# 数据筛选，将不合适的数据进行剔除
def pandas_data_filtration(
        data_input: pd.DataFrame,
        contamination: float = 0.05,
        eps: float = 0.5,
        min_samples: int = 5,
        pca_variance: float = 0.95,
) -> pd.DataFrame:
    # 输入校验a
    assert isinstance(data_input, pd.DataFrame), "Input data should be a Pandas DataFrame"
    data = data_input.copy(deep=True)

    # Step 3: 异常检测（使用清洗后的数据）
    iso_forest = IsolationForest(n_estimators=50, contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(data.values)  # 改为 data.values

    # Step 4: 降维处理
    pca = PCA(n_components=pca_variance)
    reduced_data = pca.fit_transform(data)
    data = pd.DataFrame(reduced_data, index=data.index)

    # Step 5: 聚类分析
    cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(data)

    # Step 6: 综合筛选
    valid_idx = np.where((outliers == 1) & (cluster_labels != -1))[0]

    return valid_idx

# dataframe数据的筛选，主要功能是删除空值、边界值、等等的无意义数值所在的行
def pdnads_data_drop(data_input: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(data_input, pd.DataFrame), "Input data should be a Pandas DataFrame"
    data = data_input.copy(deep=True)

    # 分列处理无效值 ---
    # 数值列处理
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        # 替换极端值
        data[numeric_cols] = data[numeric_cols].replace(-2147483648, np.nan)
        # 删除全空行
        data = data.dropna(subset=numeric_cols, how='all')
        print('data number drop:{}-->{}'.format(data_input.shape, data.shape))

    # 非数值列处理
    # str_cols = data.select_dtypes(exclude=[np.number]).columns
    invalid_str = {'', 'nan', 'NaN', 'None', 'NULL', np.nan, None, -2147483648, '-2147483648'}
    mask = ~data.isin(invalid_str).any(axis=1)
    data = data[mask].reset_index(drop=True)
    print('data all dropped-->{}'.format(data.shape))

    data = data.reset_index(drop=True)

    # 检查数据有效性
    if data.empty:
        raise ValueError("过滤后数据为空，请调整过滤条件")
    return data


# # 生成测试数据
# np.random.seed(42)
# base_data = np.vstack([
#     make_blobs(n_samples=100, centers=1, cluster_std=0.5)[0],
#     np.array([[1000, -1000], [-1000, 500]]),
#     np.full((5, 2), -2147483648),
#     np.array([['', 1], [np.nan, 3], [None, 5], [4, 8]]),
# ])
#
# test_df = pd.DataFrame(
#     data=base_data,
#     columns=['Feature1', 'Feature2']
# ).sample(frac=1).reset_index(drop=True)
#
# # # 执行清洗
# # cleaned_df = pandas_data_filtration(test_df, enable_visualization=True)
# # print("\n清洗后数据：")
# # print(cleaned_df.head())
#
# a = pdnads_data_drop(test_df)

def remove_static_depth_data(df, depth_col='DEPTH'):
    """
    删除测井数据中深度未变化的无效数据

    参数:
    df : pd.DataFrame - 包含测井数据的DataFrame
    depth_col : str - 深度列的名称，默认为'DEPTH'

    返回:
    pd.DataFrame - 处理后的DataFrame，已删除深度未变化的行
    """
    # 确保深度列存在
    if depth_col not in df.columns:
        raise ValueError(f"列 {depth_col} 在DataFrame中不存在")

    # 创建副本避免修改原始数据
    df_clean = df.copy()

    # 计算深度变化值
    depth_diff = df_clean[depth_col].diff()

    # 标记需要保留的行 - 保留第一个深度值和所有深度变化的值
    mask = depth_diff.fillna(1) > 0

    # 应用筛选条件
    filtered_df = df_clean.loc[mask]

    return filtered_df.reset_index(drop=True)


# 测试用例
def test_data():
    # 创建示例数据（基于题目提供的数据）
    data = {
        'DEPTH': [995.1, 995.2, 995.3, 995.3, 995.3, 995.6, 995.7, 995.8, 995.9,
                  996.0, 996.1, 996.1, 996.1, 996.4, 996.5, 996.6, 996.7, 996.8, 996.9, 997.0],
        'AZI': [281.9487, 281.7962, 281.7486, 281.7045, 281.6114, 281.4754, 281.4118,
                281.3925, 281.3158, 281.2673, 281.2582, 281.2489, 281.2819, 281.2676,
                281.2012, 281.2884, 281.7045, 282.1904, 282.1065, 281.6518],
        'DIP': [62.6915, 62.1646, 62.1213, 62.1200, 62.1200, 62.1200, 62.1200,
                62.1200, 62.1200, 62.1200, 62.1200, 62.1200, 62.1200, 62.1200,
                62.1200, 62.1200, 62.1200, 62.1200, 62.1200, 62.1200],
        'GR': [43.6084, 43.6807, 43.8962, 44.1992, 44.5079, 44.7575, 44.9316,
               45.0633, 45.2018, 45.3659, 45.5257, 45.6372, 45.6971, 45.7564,
               45.8668, 46.0137, 46.1180, 46.1222, 46.0716, 46.0858]
    }

    # 创建DataFrame
    df = pd.DataFrame(data)
    print("原始数据:")
    print(df)
    print(f"\n原始数据行数: {len(df)}")

    # 处理数据
    cleaned_df = remove_static_depth_data(df, 'DEPTH')

    print("\n处理后的数据:")
    print(cleaned_df)
    print(f"\n处理后数据行数: {len(cleaned_df)}")

    # 验证处理结果
    expected_depths = [995.1, 995.2, 995.3, 995.6, 995.7, 995.8, 995.9,
                       996.0, 996.1, 996.4, 996.5, 996.6, 996.7, 996.8, 996.9, 997.0]

    assert cleaned_df['DEPTH'].tolist() == expected_depths
    assert len(cleaned_df) == 16  # 原始20行 - 删除4行深度未变化的数据

    print("\n测试通过: 深度未变化的数据已被正确删除")


if __name__ == "__main__":
    test_data()