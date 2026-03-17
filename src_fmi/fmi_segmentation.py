import copy
import os.path
from typing import Dict, Any
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from src_fmi.segmentation import FMISegmentation


# 多张fmi同时计算 texture-feature
def cal_fmis_segmentation(imgs=[], depth=np.array([]), windows=100, step=50, method_configs={
        ('tophat_otsu', 'TopHat + Otsu', None),
        ('otsu', 'Otsu Threshold', None),
        ('adaptive', 'Adaptive Threshold', None),
        ('kmeans', 'K-means (K=3)', None),
        ('gmm', 'GMM (n=3)', None),
        ('wavelet', 'Wavelet (db4)', None),
    }):
    # 确保图像的长度与深度列相对应，形状一致
    for img in imgs:
        assert img.shape[0] == depth.shape[0], "image 形状长度必须等于 depth长度"
        assert len(img.shape) == 2, "image 图像必须是二维的灰度图像，不能是其他格式的"

    # 根据图像大小，计算迭代次数
    ITER_NUM = img.shape[0] // step
    print('current step:{}, windows:{}, iter_num:{}'.format(step, windows, ITER_NUM))
    # 分割结果存在重叠的地方，我们使用对结果的加权来进行处理
    mask_weight = np.zeros_like(imgs[0]).astype(np.uint8)

    # 电成像分割结果保存在新的list里面
    segmentation_result = {}
    for dict in method_configs:
        method = dict[0]
        segmentation_result[method] = []
        for img in imgs:
            segmentation_result[method].append(np.zeros_like(img))

    # 图像分割器初始化
    seg = FMISegmentation()

    # 迭代获取图像纹理信息
    with tqdm(total=ITER_NUM) as pbar:
        pbar.set_description('Processing of Extraction image texture information:')
        for i in range(ITER_NUM):
            # 计算图像读取的起始-结束Index
            INDEX_START = max(i * step - windows//2, 0)
            INDEX_END = min(i * step + windows//2, img.shape[0])
            # 相应的加权mask赋值
            mask_weight[INDEX_START:INDEX_END, :] += 1

            # 遍历电成像列表，并获取图像分割的计算结果
            for i in range(len(imgs)):
                img = imgs[i]
                window_img = copy.deepcopy(img[INDEX_START:INDEX_END, :])

                for method, name, enhance in method_configs:
                    mask = seg.segment(window_img, enhance_method=enhance, seg_method=method)
                    segmentation_result[method][i][INDEX_START:INDEX_END, :] += mask

            pbar.update(1)

    # 确保mask的加权结果不为零，不然除法的时候会报错
    mask_weight[mask_weight < 1] = 1
    for dict in method_configs:
        method = dict[0]
        for i in range(len(imgs)):
            # 先对分割结果加权处理
            segmentation_result_weight_temp = segmentation_result[method][i]/mask_weight
            # 结果二值化
            _, segmentation_result_weight_temp = cv2.threshold(segmentation_result_weight_temp.astype(np.uint8), 128, 255, cv2.THRESH_BINARY)
            # _, segmentation_result_weight_temp = cv2.threshold(segmentation_result[method][i]/mask_weight, 0,255, cv2.THRESH_OTSU)
            segmentation_result[method][i] = segmentation_result_weight_temp

    return segmentation_result


if __name__ == '__main__':
    path_test = r'F:\logging_workspace\桃镇1H\桃镇1H_STAT_target.txt'
    fmi_test = np.loadtxt(path_test, skiprows=8, delimiter='\t', dtype=np.float32)
    print(fmi_test[:2, :])
    depth_data = fmi_test[:, 0]
    fmi_data = fmi_test[:, 1:]
    print(depth_data.shape, fmi_data.shape)

    result = cal_fmis_segmentation(imgs=[fmi_data], depth=depth_data, windows=200, step=50, path_segmentation_saved=r'F:\logging_workspace\桃镇1H', charters_str_saved='STAT')
