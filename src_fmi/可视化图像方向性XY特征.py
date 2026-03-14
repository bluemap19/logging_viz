import cv2
import numpy as np
import pandas as pd

from src_file_op.dir_operation import get_all_file_paths
from src_fmi.fmi_data_read import get_ele_data_from_path
from src_fmi.image_operation import get_glcm_sub, show_Pic, get_glcm_xy, get_glcm_Features
from src_plot.plot_radar import draw_radar_chart
np.set_printoptions(precision=4, suppress=True)

if __name__ == '__main__':
    # path_folder = r'F:\桌面\收的成像特征数据集\城96'
    path_folder = r'F:\桌面\收的成像特征数据集\SIMU4'
    list_path_texture = get_all_file_paths(path_folder)

    RADIA_MAX = np.array([18.2128, 2.5686, 0.9731, 0.9448, 0.9388, 0.8927, 6.3658]) * 1.1
    RADIA_MIN = np.array([0.2756,  0.0852, 0.4992, 0.1377, 0.2329, 0.0189, 0.5891]) * 0.9
    RADIA_SUB_MAX = np.array([-0.2000, -0.0336, 0.2232, 0.0972, 0.4620, 0.0740, -0.0245])
    RADIA_SUB_MAX = RADIA_SUB_MAX + np.abs(0.1*RADIA_SUB_MAX)
    RADIA_SUB_MIN = np.array([-14.443414910050308, -1.6562,  0.0075, 0.0046, 0.0696, 0.0086, -0.8994])
    RADIA_SUB_MIN = RADIA_SUB_MIN - np.abs(0.2*RADIA_SUB_MIN)


    IMG_LIST = []
    GLCM_FEATURE_LIST_VS= []
    GLCM_SUB_LIST = []
    name_list_XY = []
    name_list_SUB = []
    for i in list_path_texture:
        if i.__contains__('STAT') and i.endswith('.txt'):
            img_data, depth_data = get_ele_data_from_path(i)
            CHAR_TEMP = i.split('/')[-1].split('\\')[-1].split('.txt')[0]
            EMPTY_PIC = np.zeros([256, 256])
            print('{} image shape is:{}'.format(CHAR_TEMP, img_data.shape))
            img_data = cv2.resize(img_data, [256, 256], interpolation=cv2.INTER_NEAREST)

            distance = [2, 4]
            texture_sub = get_glcm_sub(img_data, distance=distance)
            texture_x, texture_y, glcm_map_x, glcm_map_y = get_glcm_xy(img_data, distance=distance, level=16)
            texture_average, glcm_map_average, _, _ = get_glcm_Features(img_data, distance=distance, level=16)

            texture_all_vs = np.array([texture_average.ravel(), texture_x.ravel(), texture_y.ravel()])
            name_list_XY.append([CHAR_TEMP, 'Mean'])
            name_list_XY.append([CHAR_TEMP, f'GLCM_X'])
            name_list_XY.append([CHAR_TEMP, f'GLCM_Y'])
            GLCM_SUB_LIST.append(texture_sub)
            name_list_SUB.append([CHAR_TEMP, f'GLCM_SUB'])
            if len(GLCM_FEATURE_LIST_VS) == 0:
                GLCM_FEATURE_LIST_VS = np.array(texture_all_vs)
            else:
                GLCM_FEATURE_LIST_VS = np.vstack((GLCM_FEATURE_LIST_VS, texture_all_vs))

            # print(texture_sub.shape, texture_x.shape, texture_y.shape, glcm_map_x.shape, glcm_map_y.shape, texture_average.shape, glcm_map_average.shape)

            # IMG_LIST = [255-img_data, glcm_map_x, glcm_map_y, glcm_map_average, EMPTY_PIC, EMPTY_PIC, EMPTY_PIC, EMPTY_PIC]
            IMG_LIST = [255-img_data, glcm_map_x, glcm_map_y, glcm_map_average]
            # show_Pic(IMG_LIST, pic_order='18', figure=(42, 5), pic_str=['Img', 'GLCM_X', 'GLCM_Y', 'GLCM_Mean', 'Radar', 'Radar', 'Radar_Mean', 'Radar_Sub'], title='')
            show_Pic(IMG_LIST, pic_order='41', figure=(3, 10), pic_str=['', '', '', '', '', '', '', ''], title='')

            RADAR_LIST = [(texture_x.ravel() - RADIA_MIN) / (RADIA_MAX - RADIA_MIN),
                          (texture_y.ravel() - RADIA_MIN) / (RADIA_MAX - RADIA_MIN),
                          (texture_average.ravel() - RADIA_MIN) / (RADIA_MAX - RADIA_MIN),
                          (texture_sub.ravel() - RADIA_SUB_MIN)/(RADIA_SUB_MAX - RADIA_SUB_MIN)]
            attributes = ['CON', 'DIS', 'HOM', 'ENG', 'COR', 'ASM', 'ENT']
            # # sub_titles = ['X', 'Y', 'MEAN', 'SUB']
            # sub_titles = ['', '', '', '', '']
            # from matplotlib import pyplot as plt
            # fig2, axes2 = draw_radar_chart(
            #     data_list=RADAR_LIST,
            #     radar_str=attributes,
            #     pic_order='41',
            #     figure=(5, 26),
            #     pic_str=sub_titles,
            #     title='',
            #     norm=True
            # )
            # plt.show()


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
    # print(MEANS_LIST)
    print(np.array(MAX_LIST))
    print(np.array(MIN_LIST))

    GLCM_SUB = np.array(GLCM_SUB_LIST)
    print(GLCM_SUB.shape)
    GLCM_SUB_MEANS_LIST = []
    GLCM_SUB_MAX_LIST = []
    GLCM_SUB_MIN_LIST = []
    for j in range(GLCM_SUB.shape[1]):
        GLCM_SUB_MEANS_LIST.append(np.mean(GLCM_SUB[:, j]))
        GLCM_SUB_MAX_LIST.append(np.max(GLCM_SUB[:, j]))
        GLCM_SUB_MIN_LIST.append(np.min(GLCM_SUB[:, j]))
    # print(GLCM_SUB_MEANS_LIST)
    print(np.array(GLCM_SUB_MAX_LIST))
    print(np.array(GLCM_SUB_MIN_LIST))

    IMAGE_FEATURES_XY = np.array(GLCM_FEATURE_VS)
    # print(IMAGE_FEATURES_XY)
    Curve_Name = ['CON', 'DIS', 'HOM', 'ENG', 'COR', 'ASM', 'ENT']
    Texture_XY_df = pd.DataFrame(IMAGE_FEATURES_XY, columns=Curve_Name)
    # 将 charter 列表拆分为两个单独的列
    file_names = [item[0] for item in name_list_XY]  # 文件名列
    features = [item[1] for item in name_list_XY]  # Feature列
    # 向DataFrame添加新列
    Texture_XY_df['文件名'] = file_names
    Texture_XY_df['Feature'] = features

    IMAGE_FEATURES_SUB = np.array(GLCM_SUB)
    # print(GLCM_SUB)
    Curve_Name = ['CON', 'DIS', 'HOM', 'ENG', 'COR', 'ASM', 'ENT']
    Texture_SUB_df = pd.DataFrame(IMAGE_FEATURES_SUB, columns=Curve_Name)
    # 将 charter 列表拆分为两个单独的列
    file_names = [item[0] for item in name_list_SUB]  # 文件名列
    features = [item[1] for item in name_list_SUB]  # Feature列
    # 向DataFrame添加新列
    Texture_SUB_df['文件名'] = file_names
    Texture_SUB_df['Feature'] = features

    TEXTURE_ALL = pd.concat([Texture_XY_df, Texture_SUB_df], ignore_index=True, axis=0)
    TEXTURE_ALL.to_csv(path_folder+'\纹理特征-XY-SUB.csv', index=False)