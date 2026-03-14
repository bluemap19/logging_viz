import cv2
import numpy as np
import pandas as pd

from src_file_op.dir_operation import get_all_file_paths
from src_fmi.fmi_data_read import get_ele_data_from_path
from src_fmi.image_operation import get_glcm_sub, show_Pic, get_glcm_xy, get_glcm_Features
from src_plot.plot_radar import draw_radar_chart

np.set_printoptions(precision=4, suppress=True)


if __name__ == '__main__':

    # path_folder = r'C:\Users\Administrator\Desktop\纹理案例'
    # path_folder = r'C:\Users\ZFH\Desktop\纹理案例'
    path_folder = r'E:\桌面\收集的成像特征数据集'
    list_path_texture = get_all_file_paths(path_folder)
    distance = [2, 4]
    RADIA_MAX = np.array([16.4384, 2.3578, 0.9706, 0.9413, 0.9352, 0.8860, 6.3028]) * 1.1
    RADIA_MIN = np.array([0.2730, 0.0821, 0.5280, 0.1423, 0.2204, 0.0202, 0.5757]) * 0.5


    IMG_LIST = []
    GLCM_FEATURE_LIST_VS= []
    name_list = []
    for i in list_path_texture:
        img_data, depth_data = get_ele_data_from_path(i)
        CHAR_TEMP = i.split('/')[-1].split('\\')[-1].split('.txt')[0]
        EMPTY_PIC = np.zeros([256, 256])
        # EMPTY_PIC.fill(0)

        if i.__contains__('STAT'):
            print('{} image shape is:{}'.format(CHAR_TEMP, img_data.shape))
            img_data = cv2.resize(img_data, [256, 256], interpolation=cv2.INTER_NEAREST)

            texture_average_d1, glcm_map_average_d1, _, _ = get_glcm_Features(img_data, distance=[distance[0]], level=16)
            texture_average_d2, glcm_map_average_d2, _, _ = get_glcm_Features(img_data, distance=[distance[1]], level=16)
            texture_average_mean, glcm_map_average_mean, _, _ = get_glcm_Features(img_data, distance=distance, level=16)

            texture_all_vs = np.array([texture_average_mean.ravel(), texture_average_d1.ravel(), texture_average_d2.ravel()])
            name_list.append([CHAR_TEMP, 'Mean'])
            name_list.append([CHAR_TEMP, f'D{distance[0]}'])
            name_list.append([CHAR_TEMP, f'D{distance[1]}'])
            if len(GLCM_FEATURE_LIST_VS) == 0:
                GLCM_FEATURE_LIST_VS = np.array(texture_all_vs)
            else:
                GLCM_FEATURE_LIST_VS = np.vstack((GLCM_FEATURE_LIST_VS, texture_all_vs))

            IMG_LIST = [255-img_data, glcm_map_average_mean, EMPTY_PIC, glcm_map_average_d1, EMPTY_PIC, glcm_map_average_d2, EMPTY_PIC]
            show_Pic(IMG_LIST, pic_order='17', figure=(37, 5), pic_str=['STAT', 'GLCM_Mean', 'Radar_Mean', 'GLCM_D1', 'Radar1', 'GLCM_D2', 'Radar2'], title='')


            RADAR_LIST = [(texture_average_mean.ravel()-RADIA_MIN)/(RADIA_MAX-RADIA_MIN),
                          (texture_average_d1.ravel()-RADIA_MIN)/(RADIA_MAX-RADIA_MIN),
                          (texture_average_d2.ravel()-RADIA_MIN)/(RADIA_MAX-RADIA_MIN)]
            attributes = ['CON', 'DIS', 'HOM', 'ENG', 'COR', 'ASM', 'ENT']
            sub_titles = ['MEAN', 'D1', 'D2']

            from matplotlib import pyplot as plt
            fig2, axes2 = draw_radar_chart(
                data_list=RADAR_LIST,
                radar_str=attributes,
                pic_order='13',
                figure=(17, 5),
                pic_str=sub_titles,
                title='',
                norm=False
            )
            plt.show()
            # exit(0)



    GLCM_FEATURE_VS = np.array(GLCM_FEATURE_LIST_VS)
    ATTRIBUTE = ['CON', 'DIS', 'HOM', 'ENG', 'COR', 'ASM', 'ENT']
    print(GLCM_FEATURE_VS.shape)
    MEANS_LIST = []
    MAX_LIST  = []
    MIN_LIST = []
    for j in range(GLCM_FEATURE_VS.shape[1]):
        MEANS_LIST.append(np.mean(GLCM_FEATURE_VS[:, j]))
        MAX_LIST.append(np.max(GLCM_FEATURE_VS[:, j]))
        MIN_LIST.append(np.min(GLCM_FEATURE_VS[:, j]))

    print(GLCM_FEATURE_VS)
    print(MAX_LIST)
    print(MIN_LIST)


    IMAGE_FEATURES = np.array(GLCM_FEATURE_VS)
    print(IMAGE_FEATURES)
    Curve_Name = ['CON', 'DIS', 'HOM', 'ENG', 'COR', 'ASM', 'ENT']
    Texture_df = pd.DataFrame(IMAGE_FEATURES, columns=Curve_Name)

    # 将 charter 列表拆分为两个单独的列
    file_names = [item[0] for item in name_list]  # 文件名列
    features = [item[1] for item in name_list]  # Feature列
    # 向DataFrame添加新列
    Texture_df['文件名'] = file_names
    Texture_df['Feature'] = features

    print(Texture_df)
    Texture_df.to_csv(path_folder+'\纹理特征-空间距离.csv', index=False)