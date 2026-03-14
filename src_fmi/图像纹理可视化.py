import cv2
import numpy as np

from src_file_op.dir_operation import get_all_file_paths
from src_fmi.fmi_data_read import get_ele_data_from_path
from src_fmi.image_operation import get_glcm_sub, show_Pic, get_glcm_xy, get_glcm_Features

if __name__ == '__main__':

    # path_folder = r'C:\Users\Administrator\Desktop\纹理案例'
    # path_folder = r'C:\Users\ZFH\Desktop\纹理案例'
    path_folder = r'E:\桌面\收集的成像特征数据集'
    list_path_texture = get_all_file_paths(path_folder)


    IMG_LIST = []
    GLCM_SUB_LIST= []
    for i in list_path_texture:
        img_data, depth_data = get_ele_data_from_path(i)
        CHAR_TEMP = i.split('/')[-1].split('\\')[-1].split('.txt')[0]
        print('{} image shape is:{}'.format(CHAR_TEMP, img_data.shape))

        distance = [1, 2]
        level = 10
        if i.lower().__contains__('dyna'):
            img_data = cv2.resize(img_data, [256, 256], interpolation=cv2.INTER_NEAREST)
            IMG_LIST.append(256-img_data)

            texture_sub = get_glcm_sub(img_data)
            texture_x, texture_y, glcm_map_x, glcm_map_y = get_glcm_xy(img_data, distance=distance, level=level)
            texture_average, glcm_map_average, _, _ = get_glcm_Features(img_data, distance=distance, level=level)

            print(texture_sub.shape, texture_x.shape, texture_y.shape, glcm_map_x.shape, glcm_map_y.shape, texture_average.shape, glcm_map_average.shape)

            feature_all = [texture_average.ravel(), texture_x.ravel(), texture_y.ravel(), texture_sub.ravel()]
            print(np.array(feature_all))
            IMG_LIST = [img_data, glcm_map_average, glcm_map_x, glcm_map_y]
            show_Pic(IMG_LIST, pic_order='41', figure=(5, 12), pic_str=['Img', 'GLCM_Mean', 'GLCM_X', 'GLCM_Y'], title='')


        if i.lower().__contains__('stat'):
            img_data = cv2.resize(img_data, [256, 256], interpolation=cv2.INTER_NEAREST)
            IMG_LIST.append(256-img_data)

            texture_sub = get_glcm_sub(img_data, distance=distance)
            texture_x, texture_y, glcm_map_x, glcm_map_y = get_glcm_xy(img_data, distance=distance,level=level)
            texture_average, glcm_map_average, _, _ = get_glcm_Features(img_data, distance=distance, level=level)

            print(texture_sub.shape, texture_x.shape, texture_y.shape, glcm_map_x.shape, glcm_map_y.shape, texture_average.shape, glcm_map_average.shape)

            feature_all = [texture_average.ravel(), texture_x.ravel(), texture_y.ravel(), texture_sub.ravel()]
            print(np.array(feature_all))
            IMG_LIST = [img_data, glcm_map_average, glcm_map_x, glcm_map_y]
            show_Pic(IMG_LIST, pic_order='41', figure=(5, 12), pic_str=['Img', 'GLCM_Mean', 'GLCM_X', 'GLCM_Y'], title='')
            exit(0)
