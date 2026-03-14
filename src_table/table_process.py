import numpy as np
import pandas as pd

def get_replace_dict(df):
    """
    :param df:只能是类别信息，DF或者是array
    :return:
    """
    if isinstance(df, pd.DataFrame):
        type_data = df.values.ravel()
    elif isinstance(df, pd.Series):
        type_data = df.ravel()
    elif isinstance(df, np.ndarray):
        type_data = df.ravel()
    else:
        print(f"不支持的数据类型: {type(df)}")
        exit(1)

    if type_data.shape[0] == 0:
        print('Empty table replace data:{}'.format(df))
        return np.array([])

    # 构建表格属性替换字典
    type_replace_dict = {}
    Type_unique = np.unique(type_data)
    for i in range(Type_unique.shape[0]):
        type_replace_dict[Type_unique[i]] = i

    return type_replace_dict


# 将三维的 顶深-底深-类别 n*3信息 转换成二维的 深度-类别 的n*2信息,适合含有断层的层位信息处理
def table_3_to_2(np_layer_3, step=-1):
    if step <= 0:
        print('set right well resolution : {}'.format(step))
        exit(0)

    if np_layer_3.shape[1] != 3:
        print('label shape error:{}, please give label as n*3 shape...'.format(np_layer_3.shape))
        exit(0)

    np_layer_2 = []
    for i in range(np_layer_3.shape[0]):
        # 进行层位信息的基础检查，下一行开始的深度 不能小于 上一行结束的深度
        if i>0:
            if float(np_layer_3[i][0]) < float(np_layer_3[i - 1][1]):
                print('Error layer config:{}-->{}'.format(np_layer_3[i - 1], np_layer_3[i]))
                exit(0)
        if float(np_layer_3[i][0]) > float(np_layer_3[i][1]):
            print('Error layer config:{}'.format(np_layer_3[i]))
            exit(0)

        dep_start, dep_end, type = np_layer_3[i]
        num_dep = int((float(dep_end) - float(dep_start)) / step) + 1
        for i in range(0, num_dep):
            dep_temp = float(dep_start) + i * step
            np_layer_2.append([dep_temp, type])

    return np.array(np_layer_2)


# 把二维的 深度-类别 的n*2信息 合并成三维的 顶深-底深-类别 n*3信息，必须保持数据是连续的，中间不能有任何的数据缺失
def table_2_to_3(np_layer_2):
    if isinstance(np_layer_2, np.ndarray):
        print('function table_2_to_3 input is ndarray, type correct, continue running')
    elif isinstance(np_layer_2, pd.DataFrame):
        np_layer_2 = np_layer_2.values
        print('function table_2_to_3 input is pd.DataFrame, type change correct, continue running')
    else:
        print(f"<UNK>: {np_layer_2}")
        exit(0)

    if np_layer_2.shape[1] != 2:
        print('label shape error:{}, please give label as n*3 shape...'.format(np_layer_2.shape))
        return np.array([])

    # 记录下每一个类型发生变化的index信息
    index_list = [0]
    for i in range(np_layer_2.shape[0] - 1):
        if np_layer_2[i][-1] != np_layer_2[i + 1][-1]:
            index_list.append(i)
    index_list.append(np_layer_2.shape[0] - 1)

    depth_min = np_layer_2[0][0]
    depth_max = np_layer_2[-1][0]
    depth_step = (depth_max - depth_min)/(np_layer_2.shape[0] - 1)
    # print('depth_step is {}'.format(depth_step), index_list)

    # 通过index信息，收集类型发生变化的深度信息及类型信息
    np_layer_3 = []
    for i in range(len(index_list)-1):
        if i == 0:
            dep_start = np_layer_2[index_list[i]][0]
        else:
            dep_start = (np_layer_2[index_list[i]][0] + depth_step / 2)
        dep_end = min(np_layer_2[index_list[i + 1]][0] + depth_step / 2, depth_max)
        dep_class = np_layer_2[index_list[i + 1]][-1]
        np_layer_3.append(np.array([dep_start, dep_end, dep_class]))

    return np.array(np_layer_3)

