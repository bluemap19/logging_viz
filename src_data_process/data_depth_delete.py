import pandas as pd
from typing import Any, List, Tuple, Union
import numpy as np
import warnings


def process_depth_segment(
        df: pd.DataFrame,
        depth_col: str = None,
        depth_config: List[List[Any]] = None,
        drop: bool = True
) -> pd.DataFrame:
    """
    处理测井数据的深度段删除或获取
    Parameters:
    -------
    df : pd.DataFrame
        原始测井数据，M*N的数据体，包含深度列
    depth_col : str, optional
        深度列对应的列名，如果为None则使用第一列
    depth_config : List[List[Any]], optional
        深度段配置列表，格式如：[[depth1, depth2], [depth3, depth4]]
    drop : bool, default=True
        True: 删除depth_config指定的深度段
        False: 保留depth_config指定的深度段，删除其他
    Returns:
    --------
    pd.DataFrame
        处理后的数据
    Raises:
    -------
    ValueError
        当输入参数不符合要求时
    """

    # ========== 1. 输入参数检查 ==========
    if not isinstance(df, pd.DataFrame):
        raise ValueError("输入参数df必须是pandas DataFrame")

    if df.empty:
        warnings.warn("输入DataFrame为空，返回空DataFrame")
        return pd.DataFrame()

    # 处理depth_col参数
    if depth_col is None:
        depth_col = df.columns[0]
        print(f"警告: depth_col参数为空，使用第一列 '{depth_col}' 作为深度列")
    elif depth_col not in df.columns:
        raise ValueError(f"深度列 '{depth_col}' 不在DataFrame的列中")

    # 检查深度列是否为数值类型
    if not pd.api.types.is_numeric_dtype(df[depth_col]):
        try:
            df[depth_col] = pd.to_numeric(df[depth_col])
        except Exception as e:
            raise ValueError(f"深度列 '{depth_col}' 无法转换为数值类型: {e}")

    # 深度段配置处理
    if depth_config is None or len(depth_config) == 0:
        warnings.warn("depth_config为空，返回原始数据")
        return df.copy()

    # 验证depth_config格式
    if not isinstance(depth_config, list):
        raise ValueError("depth_config必须是列表类型")

    # 格式化depth_config，确保每个区间都是升序
    formatted_depth_config = []
    for i, segment in enumerate(depth_config):
        if not isinstance(segment, list) or len(segment) != 2:
            raise ValueError(f"深度段配置 {i} 格式错误，应为包含2个元素的列表，实际为: {segment}")

        try:
            # 尝试转换为float
            start = float(segment[0])
            end = float(segment[1])

            # 确保start <= end
            if start > end:
                start, end = end, start
                warnings.warn(f"深度段 {segment} 的起始深度大于结束深度，已自动交换")

            formatted_depth_config.append([start, end])
        except (ValueError, TypeError) as e:
            raise ValueError(f"深度段 {segment} 中的值无法转换为数值类型: {e}")

    # 合并重叠的深度段（可选，提高效率）
    formatted_depth_config = _merge_overlapping_segments(formatted_depth_config)

    # ========== 2. 数据初始化 ==========
    # 创建数据副本，避免修改原始数据
    result_df = df.copy()

    # 确保深度列有序
    if not result_df[depth_col].is_monotonic_increasing:
        result_df = result_df.sort_values(by=depth_col).reset_index(drop=True)
        warnings.warn("深度列不是单调递增的，已自动排序")

    # ========== 3. 处理逻辑 ==========
    if drop:
        # 删除指定深度段
        mask = _create_deletion_mask(result_df[depth_col], formatted_depth_config)
    else:
        # 保留指定深度段
        mask = _create_retention_mask(result_df[depth_col], formatted_depth_config)

    # 应用掩码
    processed_df = result_df[mask].copy()

    # ========== 4. 结果验证 ==========
    if processed_df.empty:
        warnings.warn("处理后数据为空，请检查深度段配置")

    return processed_df


def _merge_overlapping_segments(segments: List[List[float]]) -> List[List[float]]:
    """
    合并重叠的深度段

    Parameters:
    -----------
    segments : List[List[float]]
        深度段列表

    Returns:
    --------
    List[List[float]]
        合并后的深度段列表
    """
    if not segments:
        return []

    # 按起始深度排序
    sorted_segments = sorted(segments, key=lambda x: x[0])

    merged = []
    current = sorted_segments[0]

    for segment in sorted_segments[1:]:
        # 检查是否重叠
        if current[1] >= segment[0]:  # 有重叠
            # 合并深度段
            current[1] = max(current[1], segment[1])
        else:
            merged.append(current)
            current = segment

    merged.append(current)
    return merged


def _create_deletion_mask(depth_series: pd.Series, segments: List[List[float]]) -> pd.Series:
    """
    创建删除深度段的掩码

    Parameters:
    -----------
    depth_series : pd.Series
        深度列
    segments : List[List[float]]
        要删除的深度段列表

    Returns:
    --------
    pd.Series
        布尔掩码，True表示保留
    """
    mask = pd.Series(True, index=depth_series.index)

    for start, end in segments:
        # 标记在深度段内的数据为False（删除）
        in_segment = (depth_series >= start) & (depth_series <= end)
        mask = mask & ~in_segment

    return mask


def _create_retention_mask(depth_series: pd.Series, segments: List[List[float]]) -> pd.Series:
    """
    创建保留深度段的掩码

    Parameters:
    -----------
    depth_series : pd.Series
        深度列
    segments : List[List[float]]
        要保留的深度段列表

    Returns:
    --------
    pd.Series
        布尔掩码，True表示保留
    """
    mask = pd.Series(False, index=depth_series.index)

    for start, end in segments:
        # 标记在深度段内的数据为True（保留）
        in_segment = (depth_series >= start) & (depth_series <= end)
        mask = mask | in_segment

    return mask


def get_depth_statistics(df: pd.DataFrame, depth_col: str = None) -> dict:
    """
    获取深度统计信息，辅助函数

    Parameters:
    -----------
    df : pd.DataFrame
        数据
    depth_col : str, optional
        深度列名

    Returns:
    --------
    dict
        深度统计信息
    """
    if depth_col is None:
        depth_col = df.columns[0]

    if depth_col not in df.columns:
        return {}

    depth_series = pd.to_numeric(df[depth_col], errors='coerce')

    return {
        'min_depth': depth_series.min(),
        'max_depth': depth_series.max(),
        'depth_range': depth_series.max() - depth_series.min(),
        'num_points': len(depth_series),
        'depth_interval_mean': depth_series.diff().mean() if len(depth_series) > 1 else 0
    }


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 示例1: 创建测试数据
    print("=" * 50)
    print("示例1: 测试数据创建")
    print("=" * 50)

    # 创建示例测井数据
    np.random.seed(42)
    depth = np.arange(1000, 2000, 0.5)  # 1000-2000米，间隔0.5米
    data = {
        'DEPTH': depth,
        'GR': np.random.normal(60, 20, len(depth)),  # 伽马
        'RT': np.random.lognormal(2, 0.5, len(depth)),  # 电阻率
        'DEN': np.random.normal(2.4, 0.2, len(depth)),  # 密度
        'NEU': np.random.uniform(0.1, 0.4, len(depth))  # 中子
    }
    df = pd.DataFrame(data)

    print(f"原始数据形状: {df.shape}")
    print(f"深度范围: {df['DEPTH'].min():.1f} - {df['DEPTH'].max():.1f}")
    print("\n前5行数据:")
    print(df.head())

    # 示例2: 删除深度段
    print("\n" + "=" * 50)
    print("示例2: 删除深度段 (drop=True)")
    print("=" * 50)

    depth_config = [
        [1200, 1250],  # 删除1200-1250米
        [1300, 1350],  # 删除1300-1350米
        [1400, 1450],  # 删除1400-1450米
    ]

    result_drop = process_depth_segment(
        df=df,
        depth_col='DEPTH',
        depth_config=depth_config,
        drop=True
    )

    print(f"删除指定深度段后数据形状: {result_drop.shape}")
    print(f"剩余深度范围: {result_drop['DEPTH'].min():.1f} - {result_drop['DEPTH'].max():.1f}")

    # 示例3: 保留深度段
    print("\n" + "=" * 50)
    print("示例3: 保留深度段 (drop=False)")
    print("=" * 50)

    result_keep = process_depth_segment(
        df=df,
        depth_col='DEPTH',
        depth_config=depth_config,
        drop=False
    )

    print(f"保留指定深度段后数据形状: {result_keep.shape}")
    print(f"保留的深度范围:")
    for i, segment in enumerate(depth_config):
        print(f"  段{i + 1}: {segment[0]} - {segment[1]}")

    # 示例4: 自动处理深度段交换
    print("\n" + "=" * 50)
    print("示例4: 自动处理深度段交换")
    print("=" * 50)

    reversed_config = [
        [1250, 1200],  # 起始>结束，会自动交换
        [1300, 1350]
    ]

    result_reversed = process_depth_segment(
        df=df,
        depth_col='DEPTH',
        depth_config=reversed_config,
        drop=True
    )

    # 示例5: 获取深度统计信息
    print("\n" + "=" * 50)
    print("示例5: 深度统计信息")
    print("=" * 50)

    stats = get_depth_statistics(df, 'DEPTH')
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")

    # 示例6: 处理重叠深度段
    print("\n" + "=" * 50)
    print("示例6: 处理重叠深度段")
    print("=" * 50)

    overlapping_config = [
        [1200, 1250],
        [1240, 1280],  # 与上一段重叠
        [1300, 1350]
    ]

    result_overlap = process_depth_segment(
        df=df,
        depth_col='DEPTH',
        depth_config=overlapping_config,
        drop=True
    )

    print(f"处理重叠深度段后数据形状: {result_overlap.shape}")