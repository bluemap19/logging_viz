import pandas as pd
import numpy as np
from typing import Union, Optional

# 数据稀释函数，进行数据的抽吸
def dilute_dataframe(df: pd.DataFrame,
                     ratio: Union[float, int],
                     method: str = 'random',
                     random_state: Optional[int] = None,
                     group_by: Optional[str] = None) -> pd.DataFrame:
    """
    对DataFrame进行数据抽稀，保留指定比例的数据

    参数:
    df: 需要抽稀的DataFrame
    ratio: 保留数据的比例，可以是0-1之间的小数或0-100之间的整数
    method: 抽稀方法，可选 'random'(随机抽样) 或 'systematic'(系统抽样)
    random_state: 随机种子，用于重现结果
    group_by: 分组列名，如果提供，则按该列分组后在各组内进行抽稀

    返回:
    抽稀后的DataFrame
    """
    # 参数校验
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df必须是pandas DataFrame")

    if isinstance(ratio, int):
        if not 0 <= ratio <= 100:
            raise ValueError("当ratio为整数时，必须在0-100之间")
        ratio = ratio / 100.0
    elif isinstance(ratio, float):
        if not 0 <= ratio <= 1:
            raise ValueError("当ratio为浮点数时，必须在0-1之间")
    else:
        raise TypeError("ratio必须是整数或浮点数")

    if method not in ['random', 'systematic']:
        raise ValueError("method必须是'random'或'systematic'")

    # 如果ratio为0或1，直接返回空DataFrame或原DataFrame
    if ratio == 0:
        return df.iloc[0:0]  # 返回空DataFrame但保留列结构
    elif ratio == 1:
        return df.copy()

    # 如果有分组列，按组进行抽稀
    if group_by is not None:
        if group_by not in df.columns:
            raise ValueError(f"分组列'{group_by}'不存在于DataFrame中")

        # 按组进行抽样
        result = df.groupby(group_by, group_keys=False).apply(
            lambda x: _sample_group(x, ratio, method, random_state)
        )
        return result.reset_index(drop=True)

    # 无分组情况下的抽样
    return _sample_dataframe(df, ratio, method, random_state)


def _sample_dataframe(df: pd.DataFrame, ratio: float, method: str, random_state: Optional[int]) -> pd.DataFrame:
    """对单个DataFrame进行抽样"""
    n_samples = max(1, int(len(df) * ratio))

    if method == 'random':
        return df.sample(n=n_samples, random_state=random_state)
    else:  # systematic sampling
        step = max(1, int(1 / ratio))
        indices = list(range(0, len(df), step))
        return df.iloc[indices[:n_samples]]


def _sample_group(group: pd.DataFrame, ratio: float, method: str, random_state: Optional[int]) -> pd.DataFrame:
    """对单个组进行抽样"""
    if len(group) == 0:
        return group

    n_samples = max(1, int(len(group) * ratio))

    if method == 'random':
        return group.sample(n=n_samples, random_state=random_state)
    else:  # systematic sampling
        step = max(1, int(1 / ratio))
        indices = list(range(0, len(group), step))
        return group.iloc[indices[:n_samples]]


# 示例使用
if __name__ == "__main__":
    # 创建一个示例DataFrame
    np.random.seed(42)
    n_rows = 1000
    data = {
        'id': range(n_rows),
        'value': np.random.randn(n_rows),
        'category': np.random.choice(['A', 'B', 'C'], n_rows),
        'group': np.random.choice(['X', 'Y'], n_rows)
    }
    df = pd.DataFrame(data)

    print(f"原始数据形状: {df.shape}")

    # 随机抽稀保留30%的数据
    diluted_random = dilute_dataframe(df, ratio=30, method='random', random_state=42)
    print(f"随机抽稀30%后形状: {diluted_random.shape}")

    # 系统抽稀保留20%的数据
    diluted_systematic = dilute_dataframe(df, ratio=0.2, method='systematic')
    print(f"系统抽稀20%后形状: {diluted_systematic.shape}")

    # 按组抽稀，每个组保留50%的数据
    diluted_grouped = dilute_dataframe(df, ratio=50, method='random', group_by='group')
    print(f"按组抽稀50%后形状: {diluted_grouped.shape}")

    # 验证组内比例
    for group_name, group_data in diluted_grouped.groupby('group'):
        original_size = len(df[df['group'] == group_name])
        diluted_size = len(group_data)
        actual_ratio = diluted_size / original_size
        print(f"组 '{group_name}': 原始大小={original_size}, 抽稀后={diluted_size}, 实际比例={actual_ratio:.2f}")