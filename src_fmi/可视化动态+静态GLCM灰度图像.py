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

    RADIA_STAT_MAX = np.array([12.2933, 1.9019, 0.965, 0.9353, 0.8958, 0.8747, 6.0645]) * 1.1
    RADIA_DYNA_MAX = np.array([19.3887, 3.015, 0.5204, 0.1134, 0.8286, 0.0128, 7.6029]) * 1.1
    RADIA_SUB_STAT_MAX = np.array([-0.2, -0.0337, 0.2232, 0.0972, 0.462, 0.0741, -0.0245])
    RADIA_SUB_STAT_MAX = RADIA_SUB_STAT_MAX + np.abs(0.2 * RADIA_SUB_STAT_MAX)
    RADIA_SUB_DYNA_MAX = np.array([-2.1595, -0.2984, 0.2744, 0.0425, 0.4064, 0.0089, -0.1608])
    RADIA_SUB_DYNA_MAX = RADIA_SUB_DYNA_MAX + np.abs(0.2 * RADIA_SUB_DYNA_MAX)

    RADIA_STAT_MIN = np.array([0.3824, 0.1034, 0.5858, 0.157, 0.4051, 0.0247, 0.6046]) * 0.8
    RADIA_DYNA_MIN = np.array([6.969, 1.6592, 0.3567, 0.0808, 0.5207, 0.0065, 6.9172]) * 0.8
    RADIA_SUB_STAT_MIN = np.array([-14.4434, -1.6563, 0.0076, 0.0047, 0.0697, 0.0087, -0.8995])
    RADIA_SUB_STAT_MIN = RADIA_SUB_STAT_MIN - np.abs(0.2 * RADIA_SUB_STAT_MIN)
    RADIA_SUB_DYNA_MIN = np.array([-18.549, -2.2455, 0.0376, 0.0066, 0.0725, 0.0012, -1.0267])
    RADIA_SUB_DYNA_MIN = RADIA_SUB_DYNA_MIN - np.abs(0.2 * RADIA_SUB_DYNA_MIN)

    GLCM_DYNA_LIST = []
    GLCM_STAT_LIST = []
    GLCM_SUB_DYNA_LIST= []
    GLCM_SUB_STAT_LIST= []
    GLCM_ALL_LIST= []
    name_list = []
    for i in list_path_texture:
        EMPTY_PIC = np.zeros([256, 256])
        EMPTY_PIC.fill(255)
        if i.__contains__('STAT') and i.endswith('.txt'):
            img_stat, depth_data = get_ele_data_from_path(i)
            path_dyna = i.replace('STAT', 'DYNA')
            img_dyna, depth_data = get_ele_data_from_path(path_dyna)
            CHAR_TEMP = i.split('/')[-1].split('\\')[-1].split('.txt')[0]
            print('{} image shape is:{}'.format(CHAR_TEMP, img_stat.shape))
            img_stat = cv2.resize(img_stat, [256, 256], interpolation=cv2.INTER_NEAREST)
            img_dyna = cv2.resize(img_dyna, [256, 256], interpolation=cv2.INTER_NEAREST)

            distance = [2, 4]
            texture_sub_stat = get_glcm_sub(img_stat, distance=distance)
            texture_average_stat, glcm_map_average_stat, _, _ = get_glcm_Features(img_stat, distance=distance, level=16)
            texture_sub_dyna = get_glcm_sub(img_dyna, distance=distance)
            texture_average_dyna, glcm_map_average_dyna, _, _ = get_glcm_Features(img_dyna, distance=distance, level=16)

            GLCM_STAT_LIST.append(texture_average_stat.ravel())
            GLCM_SUB_STAT_LIST.append(texture_sub_stat.ravel())
            GLCM_DYNA_LIST.append(texture_average_dyna.ravel())
            GLCM_SUB_DYNA_LIST.append(texture_sub_dyna.ravel())

            GLCM_ALL_LIST.append(texture_average_stat)
            GLCM_ALL_LIST.append(texture_sub_stat)
            GLCM_ALL_LIST.append(texture_average_dyna)
            GLCM_ALL_LIST.append(texture_sub_dyna)
            name_list.append([CHAR_TEMP, 'STAT'])
            name_list.append([CHAR_TEMP, 'STAT_SUB'])
            name_list.append([CHAR_TEMP, 'DYNA'])
            name_list.append([CHAR_TEMP, 'DYNA_SUB'])

            IMG_LIST = [255-img_stat, glcm_map_average_stat, 255-img_dyna, glcm_map_average_dyna]
            # show_Pic(IMG_LIST, pic_order='81', figure=(5, 42), pic_str=['Img', 'GLCM_Mean', 'Radar', 'GLCM_X', 'Radar', 'GLCM_Y', 'Radar', 'Radar'], title='')
            show_Pic(IMG_LIST, pic_order='41', figure=(3, 11), pic_str=['', '', '', '', '', '', '', ''], title='')

            RADAR_LIST = [(texture_average_stat.ravel() - RADIA_STAT_MIN) / (RADIA_STAT_MAX - RADIA_STAT_MIN),
                          (texture_sub_stat.ravel() - RADIA_SUB_STAT_MIN) / (RADIA_SUB_STAT_MAX - RADIA_SUB_STAT_MIN),
                          (texture_average_dyna.ravel() - RADIA_DYNA_MIN) / (RADIA_DYNA_MAX - RADIA_DYNA_MIN),
                          (texture_sub_dyna.ravel() - RADIA_SUB_DYNA_MIN) / (RADIA_SUB_DYNA_MAX - RADIA_SUB_DYNA_MIN)]

            # attributes = ['CON', 'DIS', 'HOM', 'ENG', 'COR', 'ASM', 'ENT']
            # # sub_titles = ['MEAN', 'X', 'Y', 'SUB']
            # sub_titles = ['', '', '', '']
            # from matplotlib import pyplot as plt
            # fig2, axes2 = draw_radar_chart(
            #     data_list=RADAR_LIST,
            #     radar_str=attributes,
            #     pic_order='41',
            #     figure=(5, 24),
            #     pic_str=sub_titles,
            #     title='',
            #     norm=True
            # )
            # plt.show()
            # # exit(0)

    GLCM_STAT = np.array(GLCM_STAT_LIST)
    GLCM_SUB_STAT = np.array(GLCM_SUB_STAT_LIST)
    GLCM_DYNA = np.array(GLCM_DYNA_LIST)
    GLCM_SUB_DYNA = np.array(GLCM_SUB_DYNA_LIST)
    print(GLCM_STAT.shape, GLCM_SUB_STAT.shape, GLCM_DYNA.shape, GLCM_SUB_DYNA.shape)

    ATTRIBUTE = ['CON', 'DIS', 'HOM', 'ENG', 'COR', 'ASM', 'ENT']
    MEANS_LIST_STAT = []
    MEANS_LIST_DYNA = []
    MEANS_LIST_STAT_SUB = []
    MEANS_LIST_DYNA_SUB = []
    MAX_LIST_STAT = []
    MAX_LIST_DYNA = []
    MAX_LIST_STAT_SUB = []
    MAX_LIST_DYNA_SUB = []
    MIN_LIST_STAT = []
    MIN_LIST_DYNA = []
    MIN_LIST_STAT_SUB = []
    MIN_LIST_DYNA_SUB = []
    for j in range(len(ATTRIBUTE)):
        MEANS_LIST_STAT.append(np.mean(GLCM_STAT[:, j]))
        MEANS_LIST_DYNA.append(np.mean(GLCM_DYNA[:, j]))
        MEANS_LIST_STAT_SUB.append(np.mean(GLCM_SUB_STAT[:, j]))
        MEANS_LIST_DYNA_SUB.append(np.mean(GLCM_SUB_DYNA[:, j]))

        MAX_LIST_STAT.append(np.max(GLCM_STAT[:, j]))
        MAX_LIST_DYNA.append(np.max(GLCM_DYNA[:, j]))
        MAX_LIST_STAT_SUB.append(np.max(GLCM_SUB_STAT[:, j]))
        MAX_LIST_DYNA_SUB.append(np.max(GLCM_SUB_DYNA[:, j]))

        MIN_LIST_STAT.append(np.min(GLCM_STAT[:, j]))
        MIN_LIST_DYNA.append(np.min(GLCM_DYNA[:, j]))
        MIN_LIST_STAT_SUB.append(np.min(GLCM_SUB_STAT[:, j]))
        MIN_LIST_DYNA_SUB.append(np.min(GLCM_SUB_DYNA[:, j]))

    # print('MEANS_LIST_STAT\t\t:{}'.format(np.array(MEANS_LIST_STAT)))
    # print('MEANS_LIST_DYNA\t\t:{}'.format(np.array(MEANS_LIST_DYNA)))
    # print('MEANS_LIST_STAT_SUB\t:{}'.format(np.array(MEANS_LIST_STAT_SUB)))
    # print('MEANS_LIST_DYNA_SUB\t:{}'.format(np.array(MEANS_LIST_DYNA_SUB)))
    print('MAX_LIST_STAT\t\t:{}'.format(np.array(MAX_LIST_STAT)))
    print('MAX_LIST_DYNA\t\t:{}'.format(np.array(MAX_LIST_DYNA)))
    print('MAX_LIST_STAT_SUB\t:{}'.format(np.array(MAX_LIST_STAT_SUB)))
    print('MAX_LIST_DYNA_SUB\t:{}'.format(np.array(MAX_LIST_DYNA_SUB)))
    print('MIN_LIST_STAT\t\t:{}'.format(np.array(MIN_LIST_STAT)))
    print('MIN_LIST_DYNA\t\t:{}'.format(np.array(MIN_LIST_DYNA)))
    print('MIN_LIST_STAT_SUB\t:{}'.format(np.array(MIN_LIST_STAT_SUB)))
    print('MIN_LIST_DYNA_SUB\t:{}'.format(np.array(MIN_LIST_DYNA_SUB)))

    IMAGE_FEATURES_ALL = np.array(GLCM_ALL_LIST)
    print(IMAGE_FEATURES_ALL)
    Curve_Name = ['CON', 'DIS', 'HOM', 'ENG', 'COR', 'ASM', 'ENT']
    Texture_df = pd.DataFrame(IMAGE_FEATURES_ALL, columns=Curve_Name)

    # 将 charter 列表拆分为两个单独的列
    file_names = [item[0] for item in name_list]  # 文件名列
    features = [item[1] for item in name_list]  # Feature列
    # 向DataFrame添加新列
    Texture_df['文件名'] = file_names
    Texture_df['Feature'] = features

    print(Texture_df)
    Texture_df.to_csv(path_folder + '\纹理特征-动静态综合.csv', index=False)