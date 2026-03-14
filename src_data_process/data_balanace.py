import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from matplotlib import pyplot as plt
# 设置支持中文的字体，使用黑体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


def abs_percent_autopct(values):
    def inner_autopct(pct):
        total = sum(values)
        count = int(round(pct * total / 100.0))
        return f"{count}\n({pct:.1f}%)"  # 合并显示数值和百分比
    return inner_autopct

def smart_balance_dataset(df, target_col='岩性', method='smote', random_state=42, need_show=True, axes=None, Type_dict={}):
    """
    智能数据平衡处理接口
    参数：
        df: 原始数据框，最后一列为类别列
        target_col: 目标类别列名
        method: 处理方法(smote/adasyn/under/combine)
        random_state: 随机种子
    返回：
        平衡后的DataFrame
    """
    # 分离特征与目标
    X = df.iloc[:, :-1]
    y = df[target_col]

    # 动态选择处理方法
    if method == 'smote':
        sampler = SMOTE(random_state=random_state)
    elif method == 'adasyn':
        sampler = ADASYN(random_state=random_state)
    elif method == 'under':
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'combine':
        sampler = SMOTEENN(random_state=random_state)

    # 执行采样
    X_res, y_res = sampler.fit_resample(X, y)

    # 重构数据框
    balanced_df = pd.DataFrame(X_res, columns=X.columns)
    balanced_df[target_col] = y_res

    # 绘图逻辑重构
    if axes is None:
        # 创建新画布
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        need_show = True
    else:
        # 使用传入的坐标系
        ax1, ax2 = axes
        need_show = False
    # 绘制原始数据分布
    original_counts = pd.Series(y).value_counts()
    # 创建反向映射字典（数值->中文标签）
    reverse_dict = {v: k for k, v in Type_dict.items()}
    # 转换为整型
    original_counts.index = original_counts.index.astype(int)
    # 带异常处理的标签映射
    labels = original_counts.index.map(
        lambda x: reverse_dict.get(x, f'未知类别_{x}')  # 处理未定义类别
    )

    ax1.pie(
        original_counts,
        labels=labels,
        autopct=abs_percent_autopct(original_counts),
        startangle=90,
        textprops={'fontsize': 8, 'ha': 'center'}  # 对齐优化
    )
    ax1.set_title('原始类别分布', fontsize=12)
    # 绘制平衡后分布
    balanced_counts = pd.Series(y_res).value_counts()
    ax2.pie(
        balanced_counts,
        labels=labels,
        autopct=abs_percent_autopct(original_counts),
        startangle=90,
        textprops={'fontsize': 8, 'ha': 'center'}  # 对齐优化
    )
    ax2.set_title('平衡后类别分布', fontsize=12)
    # 自动调整布局并显示
    if need_show:
        plt.tight_layout()
        plt.show()

    return balanced_df