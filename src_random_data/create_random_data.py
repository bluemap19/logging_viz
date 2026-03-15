import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# 原始数据初始化
data = {
    'AC': [0.264, 0.031, 0.176, 0.214, 1.998, 1.355, 0.001, 0.142, 0.350, 1.0],
    'CNL': [0.621, 0.029, 0.169, 0.585, -0.154, 0.765, 0.215, 0.496, 0.707, 1.0],
    'DEN': [0.938, 0.004, 0.063, 0.953, 2.168, -1.470, 0.662, 0.909, 0.988, 1.0],
    'GR': [0.459, 0.065, 0.254, 0.512, -0.774, -0.151, 0.000, 0.249, 0.645, 1.0]
}
columns = ['Average', 'S2', 'S', 'Median', 'Peak', 'Skewness', 'Min', '25%', '75%', 'MAX']
# 创建原始DataFrame
original_df = pd.DataFrame(data).T
original_df.columns = columns
# 生成新DataFrame的两种方法
def generate_new_df(base_df, method='random', noise_scale=0.1):
    """
    生成新DataFrame的工厂函数
    :param base_df: 基准数据框
    :param method: 生成方式 ['random'|'manual']
    :param noise_scale: 随机噪声强度
    """
    if method == 'random':
        # 方法1：基于随机噪声生成
        noise = np.random.normal(scale=noise_scale, size=base_df.shape)
        new_data = base_df.values + noise
        return pd.DataFrame(new_data, index=base_df.index, columns=base_df.columns)
    elif method == 'manual':
        # 方法2：手动设置新数据
        new_data = {
            'AC': [0.300, 0.035, 0.180, 0.220, 2.000, 1.360, 0.005, 0.150, 0.355, 1.0],
            'CNL': [0.650, 0.030, 0.170, 0.590, -0.160, 0.770, 0.220, 0.500, 0.710, 1.0],
            'DEN': [0.940, 0.005, 0.065, 0.950, 2.170, -1.480, 0.660, 0.910, 0.990, 1.0],
            'GR': [0.460, 0.066, 0.255, 0.510, -0.770, -0.150, 0.001, 0.250, 0.646, 1.0]
        }
        return pd.DataFrame(new_data).T.rename(columns=lambda x: base_df.columns[x])
    else:
        raise ValueError("Unsupported generation method")
# 生成两个新DataFrame
df1 = generate_new_df(original_df, method='random')  # 随机生成
df2 = generate_new_df(original_df, method='manual')  # 手动设置

def get_dataframe_static(dict=['白75', '白159', '白259']):
    result = {}
    for i in dict:
        result[i] = generate_new_df(original_df, method='random')
    return result

# df_static = get_dataframe_static()
# print(df_static['白259'])


# 生成随机的测井曲线数据
def create_random_logging(logging_data_shape=(1000, 5), logging_resolution=0.1, dep_start=100):
    logging_data = np.zeros(logging_data_shape, dtype=float)
    for i in range(logging_data_shape[0]):
        for j in range(logging_data_shape[1]):
            if i==0:
                logging_data[i][j] = np.random.random()
            else:
                logging_data[i][j] = max(min(logging_data[i-1][j] + (np.random.random()-0.5)*0.1, 1), 0)
    # 生成 等间隔数列 当作数据的深度信息
    logging_data[:, 0] = np.arange(dep_start, dep_start+logging_resolution*logging_data_shape[0], logging_resolution)
    print('create random logging_data data as shape:{}, depth information:[{}, {}]'.format(logging_data.shape, logging_data[0, 0], logging_data[-1, 0]))
    return logging_data

def get_random_logging_dataframe(curve_name=['#DEPTH', 'GR', 'AC', 'CNL', 'DEN', 'Type'], logging_resolution=0.1, dep_start=100, dep_end=500):
    data_num = int((dep_end-dep_start)/logging_resolution)
    logging_data_shape = (data_num, len(curve_name))
    logging_data = create_random_logging(logging_data_shape, logging_resolution, dep_start=dep_start)
    df = pd.DataFrame(logging_data, columns=curve_name)
    if 'Type' in df.columns:
        df['Type'] = np.random.choice([0, 1, 2, 3, 4, 5], size=data_num, replace=True)
    return df


# a = get_random_logging_dataframe()
# print(a.describe())

def get_random_logging_dict(dict=['白75', '白159', '白259']):
    result = {}
    for i in dict:
        result[i] = get_random_logging_dataframe()
    return result


def create_random_logging_2(logging_pix=1000):
    # 生成随机测井数据
    log_values = np.cumsum(np.random.randn(logging_pix))  # 累积随机波动

    # 添加地质趋势约束
    trend = -0.5 * np.linspace(0, logging_pix//100, logging_pix)  # 整体递减趋势
    log_values += trend + np.sin(np.linspace(0, 8 * np.pi, logging_pix)) * 0.8  # 叠加周期性波动

    # 高斯滤波平滑处理（控制地质合理性）
    smoothed = gaussian_filter1d(log_values, sigma=15)

    return smoothed

def generate_well_trajectory(model_shape=(2000, 4000)):
    """生成符合地质约束的钻井轨迹"""
    # 基础双曲函数轨迹
    x = np.linspace(1, 4000, 4000)
    a,b,c = 996.49, -991.46, 0.9981
    y_base =  a - b * c ** x
    print(x[:10], y_base)

    # # 添加累积随机噪声
    # noise = np.cumsum(np.random.normal(0, noise_scale, x.shape))
    # y_noise = y_base + noise * model_shape[0] / 1000
    # print(y_noise)
    #
    # print(y_smooth)

    for i in range(len(x)):
        rand_noise = np.random.randint(-10, 10)
        if i == 0:
            y_base[i] = y_base[i] + rand_noise
        else:
            y_base[i] = (3*y_base[i-1] + y_base[i])/4 + rand_noise

    # y_base = savgol_filter(y_base, 5, 3)
    # 高斯滤波平滑
    y_base = gaussian_filter1d(y_base, sigma=15)

    # 坐标转换到模型索引
    x_idx = x.astype(np.int32)
    y_idx = np.clip(y_base.astype(np.int32), 0, model_shape[0] - 1)

    return x_idx, y_idx