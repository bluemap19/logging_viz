import numpy as np
from scipy.stats import shapiro, ttest_ind


def scale_gaussian(source_data, target_data, return_stats=False):
    """
    将源数据缩放到目标数据的分布范围，同时保持高斯分布特性

    参数:
    source_data (np.array): 要缩放的数据 (n2)
    target_data (np.array): 目标数据范围 (n1)
    return_stats (bool): 是否返回统计信息

    返回:
    scaled_data (np.array): 缩放后的数据
    stats (dict): 缩放前后的统计信息（仅当return_stats=True时返回）
    """
    # 计算目标数据的统计参数
    μ_target = np.median(target_data)  # 使用中位数更稳定
    σ_target = np.std(target_data)

    # 计算源数据的统计参数
    μ_source = np.median(source_data)
    σ_source = np.std(source_data)

    # 防止标准差为零
    if σ_source < 1e-10:
        σ_source = 1e-10

    # Z-score标准化
    z_scores = (source_data - μ_source) / σ_source

    # 缩放到目标分布
    scaled_data = z_scores * σ_target + μ_target

    if return_stats:
        # 计算缩放前后统计信息
        def get_range(data):
            """计算95%数据范围"""
            lower = np.percentile(data, 2.5)
            upper = np.percentile(data, 97.5)
            return (lower, upper)

        stats = {
            "source_mean": μ_source,
            "source_std": σ_source,
            "source_range": get_range(source_data),
            "target_mean": μ_target,
            "target_std": σ_target,
            "target_range": get_range(target_data),
            "scaled_mean": np.median(scaled_data),
            "scaled_std": np.std(scaled_data),
            "scaled_range": get_range(scaled_data),
        }
        return scaled_data, stats
    else:
        return scaled_data


def scale_gaussian_by_config(source_data, target_data_config, return_stats=False):
    """
    将源数据缩放到目标数据的分布范围，同时保持高斯分布特性

    参数:
    source_data (np.array): 要缩放的数据 (n2)
    target_data (dict{μ_target: float, σ_target: float}): 目标数据范围 (n1)
    return_stats (bool): 是否返回统计信息

    返回:
    scaled_data (np.array): 缩放后的数据
    stats (dict): 缩放前后的统计信息（仅当return_stats=True时返回）
    """
    # 计算目标数据的统计参数
    μ_target = target_data_config['μ_target']
    σ_target = target_data_config['σ_target']

    # 计算源数据的统计参数
    μ_source = np.median(source_data)
    σ_source = np.std(source_data)

    # 防止标准差为零
    if σ_source < 1e-10:
        σ_source = 1e-10

    # Z-score标准化
    z_scores = (source_data - μ_source) / σ_source

    # 缩放到目标分布
    scaled_data = z_scores * σ_target + μ_target

    if return_stats:
        # 计算缩放前后统计信息
        def get_range(data):
            """计算95%数据范围"""
            lower = np.percentile(data, 2.5)
            upper = np.percentile(data, 97.5)
            return (lower, upper)

        stats = {
            "source_mean": μ_source,
            "source_std": σ_source,
            "target_mean": μ_target,
            "target_std": σ_target,
            "source_range": get_range(source_data),
            "scaled_mean": np.median(scaled_data),
            "scaled_std": np.std(scaled_data),
            "scaled_range": get_range(scaled_data),
        }
        return scaled_data, stats
    else:
        return scaled_data


def scale_by_quantiles(source_data, target_data, quantile=0.2):
    """
    将源数据缩放到目标数据的分布范围，使源数据的20%分位数与目标数据的20%分位数对齐

    参数:
    source_data (np.array): 要缩放的数据 (n2)
    target_data (np.array): 目标数据范围 (n1)
    quantile (float): 使用的分位数 (默认为0.2，即20%)
    return_stats (bool): 是否返回统计信息

    返回:
    scaled_data (np.array): 缩放后的数据
    """
    # 计算源数据的对称分位数范围
    source_lower = np.percentile(source_data, quantile * 100)
    source_upper = np.percentile(source_data, (1 - quantile) * 100)

    # 计算目标数据的对称分位数范围
    target_lower = np.percentile(target_data, quantile * 100)
    target_upper = np.percentile(target_data, (1 - quantile) * 100)

    # 计算范围并避免零范围
    source_range = source_upper - source_lower
    if source_range < 1e-5:
        source_range = 1e-5
    target_range = target_upper - target_lower
    if target_range < 1e-5:
        target_range = 1e-5

    # 计算缩放因子和中点偏移
    scale_factor = target_range / source_range

    # 线性变换
    scaled_data = target_lower + (source_data - source_lower) * scale_factor
    return scaled_data, [target_lower, source_lower, scale_factor]


def scale_by_quantiles_use_config(source_data, config={'target_lower':0, 'source_lower':0, 'scale_factor':0}):
    target_lower = config['target_lower']
    source_lower = config['source_lower']
    scale_factor = config['scale_factor']

    # 线性变换
    scaled_data = target_lower + (source_data - source_lower) * scale_factor
    return scaled_data







