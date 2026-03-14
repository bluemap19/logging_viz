import copy
import math
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from src_fmi.image_operation import get_glcm_Features, get_glcm_sub


def cal_image_texture(img, depth, windows=20, step=5, texture_config={'level': 32, 'distance':[1, 2], 'angles':[0, np.pi/4, np.pi/2, np.pi*3/4]}, path_texture_saved='',
                      texture_headers=['CON', 'DIS', 'HOM', 'ENG', 'COR', 'ASM', 'ENT', 'XY_CON', 'XY_DIS', 'XY_HOM', 'XY_ENG', 'XY_COR', 'XY_ASM', 'XY_ENT']):
    assert img.shape[0] == depth.shape[0], "image 形状长度必须等于 depth长度"
    assert len(img.shape) == 2, "image 图像必须是二维的灰度图像，不能是其他格式的"
    assert path_texture_saved.endswith('csv') or len(path_texture_saved)==0, "保存路径必须是以csv格式结尾的"
    assert 14 == len(texture_headers), "texture_headers 数量不对"

    DEPTH_LIST = []
    TEXTURE_IMG = []

    # 根据图像大小，计算迭代次数
    ITER_NUM = math.ceil((img.shape[0] - windows) / step) + 1
    print('current step:{}, windows:{}, iter_num:{}, texture config:{}'.format(step, windows, ITER_NUM, texture_config))

    # 迭代获取图像纹理信息
    with tqdm(total=ITER_NUM) as pbar:
        pbar.set_description('Processing of Extraction image texture information:')
        for i in range(ITER_NUM):
            INDEX_START = i * step
            window_img = copy.deepcopy(img[INDEX_START:INDEX_START + windows, :])
            # 保持所有的图像数据全都使用一个宽度配置，全都是192宽的
            window_img = cv2.resize(
                window_img,
                (192, windows),  # 目标尺寸设为None
                interpolation=cv2.INTER_AREA  # 缩小首选插值
            )

            # 动态图像的纹理信息提取
            texture_org, glcm_map, _, _ = get_glcm_Features(window_img,
                                                                      level=texture_config['level'],
                                                                      distance=texture_config['distance'],
                                                                      angles=texture_config['angles'])
            # 静态图像的纹理差信息提取
            texture_sub = get_glcm_sub(window_img, level=texture_config['level'],
                                            distance=texture_config['distance'])

            # 纹理差信息合并
            texture_all = np.concatenate((texture_org.ravel(), texture_sub.ravel()), axis=0)

            DEPTH_LIST.append(depth[INDEX_START + windows // 2])
            TEXTURE_IMG.append(texture_all)
            pbar.update(1)

    DEPTH = np.array(DEPTH_LIST).reshape(-1, 1)
    TEXTURE_IMG = np.array(TEXTURE_IMG)
    TEXTURE_LOGGING = np.concatenate((DEPTH, TEXTURE_IMG), axis=1)
    print('depth shape:{}, texture_img shape:{}, texture_logging:{}'.format(DEPTH.shape, TEXTURE_IMG.shape,
                                                                            TEXTURE_LOGGING.shape))
    if texture_headers:
        pass
    else:
        texture_headers = ('CON\tDIS\tHOM\tENG\tCOR\tASM\tENT\t'
                  'XY_CON\tXY_DIS\tXY_HOM\tXY_ENG\tXY_COR\tXY_ASM\tXY_ENT').split('\t')
    np.set_printoptions(suppress=True)
    target_df = pd.DataFrame(TEXTURE_LOGGING, columns=['DEPTH'] + texture_headers)

    if len(path_texture_saved) == 0:
        pass
    else:
        if path_texture_saved.endswith('csv'):
            target_df.to_csv(path_texture_saved, index=False)
        elif path_texture_saved.endswith('xlsx'):
            target_df.to_excel(path_texture_saved, index=False)

    return TEXTURE_LOGGING

# 多张图像同时计算 texture-feature
def cal_images_texture(imgs=[], depth=np.array([]), windows=20, step=5, texture_config={'level': 32, 'distance':[1, 2], 'angles':[0, np.pi/4, np.pi/2, np.pi*3/4]}, path_texture_saved='',
                       texture_headers=['CON', 'DIS', 'HOM', 'ENG', 'COR', 'ASM', 'ENT', 'XY_CON', 'XY_DIS', 'XY_HOM', 'XY_ENG', 'XY_COR', 'XY_ASM', 'XY_ENT']):
    for img in imgs:
        assert img.shape[0] == depth.shape[0], "image 形状长度必须等于 depth长度"
        assert len(img.shape) == 2, "image 图像必须是二维的灰度图像，不能是其他格式的"
    assert path_texture_saved.endswith('.xlsx') or path_texture_saved.endswith('.csv') or len(path_texture_saved)==0, "保存路径必须是以csv格式结尾的"
    assert len(imgs)*28 == len(texture_headers), "texture_headers 数量不对"

    DEPTH_LIST = []
    TEXTURE_LIST = []

    # 根据图像大小，计算迭代次数
    ITER_NUM = img.shape[0] // step
    print('current step:{}, windows:{}, iter_num:{}, texture config:{}'.format(step, windows, ITER_NUM, texture_config))

    # 迭代获取图像纹理信息
    with tqdm(total=ITER_NUM) as pbar:
        pbar.set_description('Processing of Extraction image texture information:')
        for i in range(ITER_NUM):
            INDEX_START = max(i * step - windows//2, 0)
            INDEX_END = min(i * step + windows//2, img.shape[0])
            texture_all_depth = np.array([])

            for img in imgs:
                window_img = copy.deepcopy(img[INDEX_START:INDEX_END, :])
                # 保持所有的图像数据全都使用一个宽度配置，全都是192宽的
                window_img = cv2.resize(
                    window_img,
                    (192, windows),  # 目标尺寸设为None
                    interpolation=cv2.INTER_AREA  # 缩小首选插值
                )

                # 图像的 纹理信息均值 提取
                texture_mean, glcm_map, _, _ = get_glcm_Features(window_img, level=texture_config['level'], distance=texture_config['distance'], angles=texture_config['angles'])
                # 图像的 纹理差信息 提取
                texture_sub = get_glcm_sub(window_img, level=texture_config['level'], distance=texture_config['distance'])
                # 图像的 纹理信息X 提取
                texture_x, glcm_x, _, _ = get_glcm_Features(window_img, level=texture_config['level'], distance=texture_config['distance'], angles=[0])
                # 图像的 纹理信息Y 提取
                texture_y, glcm_y, _, _ = get_glcm_Features(window_img, level=texture_config['level'], distance=texture_config['distance'], angles=[np.pi/2])

                # 纹理差信息合并
                texture_img_temp = np.concatenate((texture_mean.ravel(), texture_sub.ravel(), texture_x.ravel(), texture_y.ravel()), axis=0)
                if texture_all_depth.size == 0:
                    texture_all_depth = texture_img_temp
                else:
                    texture_all_depth = np.concatenate((texture_all_depth, texture_img_temp), axis=0)

            DEPTH_LIST.append(depth[i * step])
            TEXTURE_LIST.append(texture_all_depth)
            pbar.update(1)

    DEPTH_LIST = np.array(DEPTH_LIST).reshape(-1, 1)
    TEXTURE_LIST = np.array(TEXTURE_LIST)
    TEXTURE_LOGGING = np.concatenate((DEPTH_LIST, TEXTURE_LIST), axis=1)
    print('depth shape:{}, texture_img shape:{}, texture_logging:{}'.format(DEPTH_LIST.shape, TEXTURE_LIST.shape, TEXTURE_LOGGING.shape))
    np.set_printoptions(suppress=True)
    target_df = pd.DataFrame(TEXTURE_LOGGING, columns=['DEPTH'] + texture_headers)

    if len(path_texture_saved) == 0:
        print('\033[31m' + 'DONT SAVE TEXTURE FILE'+ '\033[0m')
    else:
        if path_texture_saved.endswith('csv'):
            target_df.to_csv(path_texture_saved, index=False)
        elif path_texture_saved.endswith('xlsx'):
            target_df.to_excel(path_texture_saved, index=False)
        else:
            print('\033[31m' + 'ERROR SVAE TEXTURE PATH AS:{}, and texture file not saved.'.format(path_texture_saved) + '\033[0m')

    return target_df


