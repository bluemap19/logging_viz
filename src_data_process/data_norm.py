import copy

import numpy as np
import pandas as pd


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
def data_normalized_locally(logging_data, windows_length=500, max_ratio=0.1, logging_range=[-999, 9999], DEPTH_USE=False):
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
    curve_normalize = curve_org.copy()

    curve_normalize_fully, extreme_list = data_normalized(curve_normalize, DEPTH_USE=DEPTH_USE, logging_range=logging_range, max_ratio=max_ratio)
    extreme_list = np.array(extreme_list)
    np.set_printoptions(precision=4)
    print('curve normalized shape is :{}, extreme list:\n{}'.format(curve_normalize_fully.shape, extreme_list))

    # 局部特征以及整体特征的曲线归一化
    # 局部的曲线归一化，及其消耗时间，非必要一般不进行处理
    if local_normalized:
        curve_normalize_locally = data_normalized_locally(curve_normalize, DEPTH_USE=True)
        return curve_normalize_locally

    return curve_normalize_fully



# def data_norm(self, i):
#     print('current well name:{}'.format(self.Config_logging[i]['sheet_name']))
#     logging_data_Temp = data_Normalized(self.data_list[i].values, DEPTH_USE=True, local_normalized=False)
#     logging_data_Temp[logging_data_Temp < -99] = np.nan  # 对不合规的数据 进行原地修改
#     logging_data_Temp = pd.DataFrame(logging_data_Temp, columns=self.Config_logging[i]['curve_name'])
#     print(f'data normalized describe:\n{logging_data_Temp.describe()}')
#     self.data_norm_list.append(logging_data_Temp)

