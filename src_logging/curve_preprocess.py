import copy
import math
import pandas as pd
from scipy.signal import find_peaks
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# 对聚类结果进行画图
from pylab import mpl
# 设置中文显示字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

def find_mode(arr: np.array) -> np.float64:
    """返回数组中的所有众数"""
    # 展平数组并计算唯一值与频次
    flat_arr = arr.ravel()
    unique_vals, counts = np.unique(flat_arr, return_counts=True)

    # 找出最高频次对应的所有值
    max_count = np.max(counts)
    modes = unique_vals[counts == max_count]
    return modes[0]


def get_resolution_by_depth(depth_array: np.array) -> np.float64:
    depth_array = depth_array.ravel()
    depth_diff = np.diff(depth_array)
    depth_resolution = find_mode(depth_diff)
    return depth_resolution


# 根据深度获取曲线的index，只适合连续测井深度数据
def get_index_by_depth(logging, depth):
    """
    根据深度获取曲线的index，只适合连续测井深度数据
    :param n_logging: depth information, don't include other logging_data data
    :param depth: folat:target depth need to get index
    :return: int:index to taget depth
    """
    depth_t = logging.ravel()
    index_temp = int(logging.shape[0] * (depth-depth_t[0])/(depth_t[-1]-depth_t[0]) - 5)
    while((depth_t[index_temp]<depth) & (index_temp<depth_t.shape[0])):
        index_temp += 1

    return index_temp


# 获取指定曲线 指定比例 指定范围内的 最大值、最小值
def get_extreme_value_by_ratio(curve=np.array([]), ratio_c=0.2, range_c=[-99, 9999]):
    """
    获取指定曲线 指定比例 指定范围内的 最大值、最小值
    :param curve:
    :param ratio_c:
    :param range_c:
    :return:
    """
    max_list = []
    min_list = []

    d_t = curve.ravel()

    num_ratio = int(d_t.shape[0] * ratio_c)
    # print(d_t.shape)
    for i in range(d_t.shape[0]):
        if (d_t[i]>range_c[0]) & (d_t[i]<range_c[1]):
            if len(max_list) < num_ratio:
                # 初始化 最大列表、最小列表
                max_list.append(d_t[i])
                min_list.append(d_t[i])
            else:
                max_from_min_list = max(min_list)
                min_from_max_list = min(max_list)
                if d_t[i] > min_from_max_list:
                    index_min = max_list.index(min_from_max_list)
                    max_list[index_min] = d_t[i]
                if d_t[i] < max_from_min_list:
                    index_max = min_list.index(max_from_min_list)
                    min_list[index_max] = d_t[i]
        else:
            # 在阈值之外，直接跳过，进行下一个的迭代
            continue

    max_list = np.array(max_list)
    min_list = np.array(min_list)
    # print('max_list shape is :{}'.format(max_list.shape))
    if (max_list.size >= 0) & (min_list.size >= 0):
        max_mean = np.mean(max_list)
        min_mean = np.mean(min_list)
    else:
        print('error max or min list:{} or {}'.format(max_list, min_list))
    return max_mean, min_mean


# 整体范围上的数据标准化
def data_normalized(logging_data, max_ratio=0.1, logging_range=[-99, 9999], DEPTH_USE=False):
    """
    整体数据范围特征上的数据标准化
    :param logging_data:
    :param max_ratio:
    :param logging_range:
    :param DEPTH_USE:
    :return:
    """
    if DEPTH_USE:
        logging_data_N = copy.deepcopy(logging_data[:, 1:])
    else:
        logging_data_N = copy.deepcopy(logging_data)

    extreme_list = []
    # print('logging_data_N shape is {}'.format(logging_data_N.shape))
    for j in range(logging_data_N.shape[1]):
        max_F, min_F = get_extreme_value_by_ratio(logging_data_N[:, j], ratio_c=max_ratio, range_c=logging_range)
        if (max_F==None) | (min_F==None):
            logging_data_N[:, j] = 0
            extreme_list.append([max_F, min_F])
        else:
            logging_data_N[:, j] = (logging_data_N[:, j]-min_F)/(max_F-min_F+0.001)
            extreme_list.append([max_F, min_F])
        logging_data_N[:, j][logging_data_N[:, j]<0] = 0
        logging_data_N[:, j][logging_data_N[:, j]>1] = 1

    if DEPTH_USE:
        logging_data_N[(logging_data[:, 1:] < logging_range[0]) | (logging_data[:, 1:] > logging_range[1])] = np.nan
        logging_data_N = np.hstack((logging_data[:, 0].reshape(-1, 1), logging_data_N))
    else:
        logging_data_N[(logging_data >= logging_range[0]) & (logging_data < logging_range[1])] = np.nan

    return logging_data_N, extreme_list


def data_normalized_manually(logging_data, limit=[[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]], DEPTH_USE=False):
    """
    整体数据范围特征上的数据标准化
    :param logging_data:
    :param max_ratio:
    :param logging_range:
    :param DEPTH_USE:
    :return:
    """
    if DEPTH_USE:
        logging_data_N = copy.deepcopy(logging_data[:, 1:])
    else:
        logging_data_N = copy.deepcopy(logging_data)

    extreme_list = limit
    for j in range(logging_data_N.shape[1]):
        max_F, min_F = extreme_list[j]
        logging_data_N[:, j] = (logging_data_N[:, j]-min_F)/(max_F-min_F)

    if DEPTH_USE:
        logging_data_N = np.hstack((logging_data[:, 0].reshape(-1, 1), logging_data_N))

    return logging_data_N


# 基于局部特征的数据标准化
def data_normalized_locally(logging_data, windows_length=400, max_ratio=0.1, logging_range=[-999, 9999], DEPTH_USE=False):
    """
    基于局部特征的数据标准化
    :param logging_data:
    :param windows_length:
    :param max_ratio:
    :param logging_range:
    :param DEPTH_USE:
    :return:
    """
    if DEPTH_USE:
        logging_data_N = copy.deepcopy(logging_data[:, 1:])
        logging_data_N2 = copy.deepcopy(logging_data[:, 1:])
    else:
        logging_data_N = copy.deepcopy(logging_data)
        logging_data_N2 = copy.deepcopy(logging_data)

    for i in range(logging_data.shape[0]):
        index_start = i
        index_end = min(index_start+windows_length//2, logging_data.shape[0]-1)
        index_start = max(0, index_end-windows_length)
        logging_data_temp = logging_data_N[index_start:index_end, :]
        for j in range(logging_data_N.shape[1]):
            max_F, min_F = get_extreme_value_by_ratio(logging_data_temp[:, j], ratio_c=max_ratio, range_c=logging_range)
            logging_data_N2[i,j] = (logging_data_N[i, j]-min_F)/(abs((max_F-min_F))+0.01)

    if DEPTH_USE:
        logging_data_N2 = np.hstack((logging_data[:,0].reshape(-1, 1), logging_data_N2))
        # logging_data[:, 1:] = logging_data_N2
        # logging_data_N2 = logging_data

    return logging_data_N2



# 测井曲线归一化
def data_Normalized(curve_org, DEPTH_USE=True, local_normalized=False, logging_range=[-99, 9999], max_ratio=0.1):
    curve_normalize = copy.deepcopy(curve_org)

    # 局部特征以及整体特征的曲线归一化
    # 局部的曲线归一化，及其消耗时间，非必要一般不进行处理
    if local_normalized:
        curve_normalize_locally = data_normalized_locally(curve_normalize, DEPTH_USE=True)
        return curve_normalize_locally

    curve_normalize_fully, extreme_list = data_normalized(curve_normalize, DEPTH_USE=DEPTH_USE, logging_range=logging_range, max_ratio=max_ratio)
    extreme_list = np.array(extreme_list)
    # np.set_printoptions(precision=4)
    # print('curve normalized shape is :{}, extreme list:\n{}'.format(curve_normalize_fully.shape, extreme_list))

    return curve_normalize_fully


def activity_based_segmentation(df, window_size=5, threshold=0.2, cols=None):
    """
    基于活度函数的地层分割接口
    参数：
        df: 测井数据DataFrame，首列为深度列
        window_size: 滑动窗口大小（奇数）
        threshold: 峰值检测阈值(0-1)
        cols: 指定处理的测井曲线列名列表
    返回：
        分割点深度值列表
    """
    # 数据预处理
    depth_col = df.columns[0]
    if cols is None:
        cols = df.columns[1:]  # 自动排除深度列
    else:
        cols = [c for c in cols if c != depth_col]

    # 计算各曲线活度
    activity = pd.DataFrame()
    for col in cols:
        # 滑动窗口局部方差计算
        rolling_var = df[col].rolling(window=window_size, center=True).var().fillna(0)
        activity[col] = rolling_var / rolling_var.max()  # 归一化

    # 综合活度计算
    composite_activity = activity.mean(axis=1)  # 多参数平均

    # 寻找活度峰值
    peaks, _ = find_peaks(composite_activity, height=threshold)

    # 获取分割深度
    depth_values = df[depth_col].values
    split_depths = depth_values[peaks]

    return sorted(split_depths.tolist())


def cluster_by_activity(df, window_size=5, threshold=0.2, cols=['GR', 'RT']):
    # 调用接口
    segments = activity_based_segmentation(
        df,
        window_size=window_size,
        threshold=threshold,
        cols=cols
    )
    segments = [df.iloc[0,0]] + segments + [df.iloc[-1,0]]
    # print("地层分割深度点：", segments)

    type_3_col = []
    for i in range(len(segments)):
        if i == 0:
            pass
        else:
            type_3_col.append([segments[i-1], segments[i], i-1])
    type_3_col = np.array(type_3_col)


    type_2_col = data_combine_table3col(df.values, type_3_col)


    df_result = pd.DataFrame(type_2_col, columns=list(df.columns)+['Activity'])
    return df_result




