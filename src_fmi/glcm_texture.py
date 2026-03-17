# 计算GLCM矩阵的熵值，输入的是GLCM矩阵，16*16*M*N或者是32*32*M*N、64*64*M*N等
import copy

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from src_file_op.dir_operation import get_all_file_paths
from src_fmi.fmi_data_read import get_ele_data_from_path
from src_fmi.image_operation import show_Pic


def glcm_entropy(glcm):
    """
    计算多维GLCM矩阵的熵值矩阵
    参数:
        glcm: 灰度共生矩阵，形状为 [L, L, M, N] (L=16,32,64等; M=距离数量; N=角度数量)
    返回:
        entropy_matrix: 二维矩阵 [M, N]，每个元素表示对应位置GLCM的熵值
    """
    if len(glcm.shape) == 2:
        glcm = glcm.reshape((glcm.shape[0], glcm.shape[1], 1, 1))
    elif len(glcm.shape) == 3:
        glcm = glcm.reshape((glcm.shape[0], glcm.shape[1], glcm.shape[2], 1))

    # 保存原始形状以便后续恢复
    original_shape = glcm.shape
    L = original_shape[0]  # 灰度级

    # 将多维GLCM重塑为二维矩阵 (L*L, M*N)
    glcm_2d = glcm.reshape(L * L, -1)
    num_glcms = glcm_2d.shape[1]  # GLCM切片数量

    # 计算每个GLCM切片的归一化因子
    totals = np.sum(glcm_2d, axis=0)
    # 避免除零错误（全零矩阵处理）
    totals[totals == 0] = 1.0

    # 归一化所有GLCM切片
    glcm_norm = glcm_2d / totals

    # 计算熵值（向量化操作）
    entropy_vals = np.zeros(num_glcms)
    for i in range(num_glcms):
        non_zero_probs = glcm_norm[:, i][glcm_norm[:, i] > 0]
        if non_zero_probs.size > 0:
            entropy_vals[i] = -np.sum(non_zero_probs * np.log2(non_zero_probs + 1e-10))

    # 恢复为原始维度结构 [M, N]
    return entropy_vals.reshape(original_shape[2:])



### 不能删除，这个调用在起作用
def get_glcm_Features(IMG_gray, level=16, distance=[1, 2], angles=[0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                      feature_descrip=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']):  # 传入灰度图像
    """
    :param IMG_gray: 原始图像，一般为灰度图像
    :param level: 颜色等级，将图像灰度化为多少个颜色level的简单图像，方便进行灰度统计，这个决定了，GLCM矩阵的形状长度
    :param distance: 距离列表，构建glcm矩阵时，使用了那些距离参数
    :param angles: 角度列表，构建glcm矩阵时，使用的角度参数
    :param feature_descrip:都是哪些特征进行纹理信息的提取，目前支持：
    feature_descrip = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    :return: 返回四个矩阵，分别是，原始GLCM矩阵特诊参数，原始GLCM矩阵，平均GLCM矩阵特征参数，平均GLCM特征矩阵
    其中平均GLCM特征矩阵一般为七个，分别是，总的GLCM特征，两个不同distance特征矩阵（四个角度下的均值），四个不同angle特征矩阵（两个距离下的均值）
    """
    img = copy.deepcopy(IMG_gray).astype(np.int32)
    # 灰度压缩（确保处理一致）
    img = np.floor_divide(img, 256 / level).astype(np.uint8)
    img = np.clip(img, 0, level - 1)

    # 得到共生矩阵，参数：图像矩阵，距离，方向，灰度级别，是否对称，是否标准化
    glcm = graycomatrix(img,
                        distances=distance,
                        angles=angles,
                        levels=level,
                        symmetric=False,
                        normed=True)      # 100x100的灰度图像  ---> 16*16*2*4 不同 level * level * distance * angle  的灰度共生矩阵
    # print(glcm.shape)                 ####### 16*16*2*4 即：level * level * distance * angle

    # 总的glcm平均矩阵
    glcm_mean = [glcm.reshape((level, level, -1)).mean(axis=2)]         # 16*16*1
    # 图像  不同距离 下的glcm灰度共生矩阵glcm
    for i in range(glcm.shape[2]):                                      # 16*16*2
        glcm_mean.append(glcm[:, :, i, :].mean(axis=2))
    # 图像 不同方向 上的图形glcm灰度矩阵
    for i in range(glcm.shape[3]):                                      # 16*16*4
        glcm_mean.append(glcm[:, :, :, i].mean(axis=2))

    glcm_mean = np.array(glcm_mean)                                     # ---> 7*16*16
    glcm_mean = glcm_mean.transpose(1, 2, 0)                            # 转换形状7*16*16--->16*16*7
    glcm_mean = np.expand_dims(glcm_mean, axis=-1)                      # 转换形状16*16*7--->16*16*7*1

    # 得到共生矩阵的特征统计值，官方文档
    # http://tonysyu.github.io/scikit-image/api/skimage.feature.html#skimage.feature.greycoprops
    features = []
    # features = cal_glcm_features(glcm_mean, feature_descrip)

    ###### 这个返回的是： 总的平均矩阵特征参数、总的平均GLCM矩阵、六个平均矩阵（分别是2个距离上的，4个角度上的）
    for prop in feature_descrip:
        temp = graycoprops(glcm_mean, prop)
        # print(temp.ravel())
        features.append(temp)
    features.append(glcm_entropy(glcm_mean))

    # feature_mean ---> (7, 1, 1)
    # glcm_mean ---> (16, 16)
    # feature  --->  (7, 3(len(distance)+len(angles)), 1)
    # print(np.array(feature).shape, glcm_mean.shape)
    # glcm_mean  ---> (16, 16, 4(len(distance)+len(angles)+1), 1)

    return np.array(features)[:, 0, 0], glcm_mean[:, :, 0, 0], np.array(features), glcm_mean


def get_glcm_sub(IMG_gray, level=16, distance=[1,2]):  # 传入灰度图像
    """
    :param IMG_gray: 原始图像，一般为灰度图像
    :param level: 颜色等级，将图像灰度化为多少个颜色level的简单图像，方便进行灰度统计，这个决定了，GLCM矩阵的形状长度
    :param distance: 距离列表，构建glcm矩阵时，使用了那些距离参数
    :return: 返回灰度矩阵的X-Y纹理差结果，是一个float类型数值
    """
    texture_x, glcm_map_x, feature_all_x, glcm_map_x = get_glcm_Features(IMG_gray, level=level, distance=distance, angles=[0])
    texture_y, glcm_map_y, feature_all_y, glcm_map_y = get_glcm_Features(IMG_gray, level=level, distance=distance, angles=[np.pi / 2])

    # print(texture_x.shape, texture_y.shape, glcm_map_x.shape, glcm_map_y.shape)
    texture_sub = texture_x - texture_y

    return texture_sub

def get_glcm_xy(IMG_gray, level=16, distance=[1,2]):  # 传入灰度图像
    """
    :param IMG_gray: 原始图像，一般为灰度图像
    :param level: 颜色等级，将图像灰度化为多少个颜色level的简单图像，方便进行灰度统计，这个决定了，GLCM矩阵的形状长度
    :param distance: 距离列表，构建glcm矩阵时，使用了那些距离参数
    :return: 返回灰度矩阵的X-Y纹理差结果，是一个float类型数值
    """
    texture_x, glcm_map_x, feature_all_x, glcm_map_all_x = get_glcm_Features(IMG_gray,
                                                                         level=level, distance=distance, angles=[0])
    texture_y, glcm_map_y, feature_all_y, glcm_map_all_y = get_glcm_Features(IMG_gray,
                                                                         level=level, distance=distance, angles=[np.pi / 2])

    return texture_x, texture_y, glcm_map_x, glcm_map_y


def show_pic_glcm_graymap_effect():
    folder_path = r'D:\GitHubProj\Logging_Interpretation\test\texture_set'
    path_list = get_all_file_paths(folder_path)
    distance = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4]
    level = 16
    pic_all_list = []
    feature_all_list = []
    str_list_all = []

    print(path_list)
    for path in path_list:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))

        _, _, feature, glcm_matric = get_glcm_Features(img, level=level, distance=distance, angles=angles)
        pic_all_list.append(img)
        pic_all_list.append(glcm_matric[:, :, 0, 0])
        feature_all_list.append(feature[:, 0, 0])

        pic_list_glcm_matric = [img, glcm_matric[:, :, 0, 0]]
        pic_str = ['原始图像', 'GLCM_AVERAGE']
        # 不同distance的图像 灰度矩阵添加
        for i in range(len(distance)):
            pic_list_glcm_matric.append(glcm_matric[:, :, i+1, 0])
            pic_str.append('GLCM_distance_{}'.format(distance[i]))

        # 不同angle的图像 灰度矩阵对添加
        for i in range(len(angles)):
            pic_list_glcm_matric.append(glcm_matric[:, :, len(distance)+i+1, 0])
            pic_str.append('GLCM_angle_{:.2f}Π'.format(angles[i] / np.pi))

        pic_order = '42'

        CHARTER = path.split('/')[-1].split('.')[0]
        show_Pic(pic_list_glcm_matric, pic_order=pic_order, save_pic=True,
                 path_save='C:\\Users\\amd\\Desktop\\PIC_TEXTURE\\{}.png'.format(CHARTER),
                 title=path.split('/')[-1].split('.')[0],
                 pic_str=pic_str, figure=(9, 16))
        # print(feature.shape, glcm_matric.shape)
        print(CHARTER, feature.shape, feature.ravel()[:6])
        str_list_all.append(CHARTER)
        str_list_all.append('GLCM_MATRIC')

    show_Pic(pic_all_list, pic_order='46', pic_str=str_list_all, save_pic=True,
             path_save='C:\\Users\\amd\\Desktop\\PIC_TEXTURE\\Abstract.png')
    # 设置打印位数为2 设置是否使用科学计数法
    np.set_printoptions(precision=2, suppress=True)
    print(np.array(feature_all_list))


def show_pic_glcm_graymap_effect_x_y():
    folder_path = r'D:\GitHubProj\Logging_Interpretation\test\texture_set'
    path_list = get_all_file_paths(folder_path)
    distance = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4]
    level = 16
    pic_all_list = []
    feature_all_list = []
    str_list_all = []

    print(path_list)
    for path in path_list:
        # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img, _ = get_ele_data_from_path(path)
        img = cv2.resize(img, (128, 128))

        _, _, feature_x, glcm_matric_x = get_glcm_Features(img, level=level, distance=distance, angles=[0])
        _, _, feature_y, glcm_matric_y = get_glcm_Features(img, level=level, distance=distance, angles=[np.pi / 2])
        _, _, feature_mean, glcm_matric_mean = get_glcm_Features(img, level=level, distance=distance, angles=angles)

        pic_all_list.append(img)
        pic_all_list.append(glcm_matric_mean[:, :, 0, 0])
        pic_all_list.append(glcm_matric_x[:, :, 0, 0])
        pic_all_list.append(glcm_matric_y[:, :, 0, 0])
        feature_all_list.append(feature_mean[:, 0, 0])

        pic_list_glcm_matric = [img, glcm_matric_mean[:, :, 0, 0],
                                glcm_matric_x[:, :, 0, 0], glcm_matric_y[:, :, 0, 0]]
        pic_str = ['原始图像', 'GLCM_AVERAGE', 'GLCM_X', 'GLCM_Y']
        pic_order = '22'
        CHARTER = path.split('/')[-1].split('.')[0]
        show_Pic(pic_list_glcm_matric, pic_order=pic_order, save_pic=True,
                 path_save='C:\\Users\\amd\\Desktop\\PIC_TEXTURE\\{}_angle.png'.format(CHARTER),
                 title=path.split('/')[-1].split('.')[0],
                 pic_str=pic_str, figure=(9, 9))
        # print(feature.shape, glcm_matric.shape)
        # print(CHARTER, feature.shape, feature.ravel()[:6])
        str_list_all.append(CHARTER)
        str_list_all.append('GLCM_MATRIC_MEAN')
        str_list_all.append('GLCM_MATRIC_X')
        str_list_all.append('GLCM_MATRIC_Y')

    print(len(pic_all_list))
    show_Pic(pic_all_list, pic_order='68', pic_str=str_list_all, save_pic=True, figure=(24, 12),
             path_save='C:\\Users\\amd\\Desktop\\PIC_TEXTURE\\Abstract_angle.png')
    # 设置打印位数为2 设置是否使用科学计数法
    np.set_printoptions(precision=2, suppress=True)
    print(np.array(feature_all_list))
