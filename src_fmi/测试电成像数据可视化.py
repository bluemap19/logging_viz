import cv2
import numpy as np

from src_file_op.dir_operation import get_all_file_paths
from src_fmi.fmi_data_read import get_ele_data_from_path
from src_fmi.image_operation import get_glcm_sub, show_Pic, get_glcm_xy, get_glcm_Features
from src_plot.plot_radar import draw_radar_chart

if __name__ == '__main__':
    RADIA_STAT_MAX = np.array([12.2933, 1.9019, 0.965, 0.9353, 0.8958, 0.8747, 6.0645]) * 1.1
    RADIA_DYNA_MAX = np.array([19.3887, 3.015, 0.5204, 0.1134, 0.8286, 0.0128, 7.6029]) * 1.1
    RADIA_STAT_MIN = np.array([0.3824, 0.1034, 0.5858, 0.157, 0.4051, 0.0247, 0.6046]) * 0.8
    RADIA_DYNA_MIN = np.array([6.969, 1.6592, 0.3567, 0.0808, 0.5207, 0.0065, 6.9172]) * 0.8

    path_dyna_test = r'E:\桌面\收集的成像特征数据集\Cheng96_FMI-Type-1-STAT-2046.25-2047-薄夹层砂岩-长英质页岩-中低GR长英质.txt'
    path_stat_test = r'E:\桌面\收集的成像特征数据集\Cheng96_FMI-Type-1-STAT-2046.25-2047-薄夹层砂岩-长英质页岩-中低GR长英质.txt'


    img_dyna_data, _ = get_ele_data_from_path(path_dyna_test)
    img_stat_data, _ = get_ele_data_from_path(path_stat_test)


    img_dyna_data = cv2.resize(img_dyna_data, [256, 256], interpolation=cv2.INTER_NEAREST)
    img_stat_data = cv2.resize(img_stat_data, [256, 256], interpolation=cv2.INTER_NEAREST)


    print(img_dyna_data.shape)
    print(img_stat_data.shape)

    dyna_texture_average, dyna_glcm_map_average, _, _ = get_glcm_Features(img_dyna_data)
    stat_texture_average, stat_glcm_map_average, _, _ = get_glcm_Features(img_stat_data)

    # show_Pic([img_dyna_data, img_stat_data, dyna_glcm_map_average, stat_glcm_map_average], pic_order='22', figure=(12, 12),
    #          pic_str=['IMG_DYNA', 'IMG_STAT', 'glcm_map_dyna', 'glcm_map_stat'], title='')

    RADAR_LIST = [(dyna_texture_average.ravel() - RADIA_STAT_MIN) / (RADIA_STAT_MAX - RADIA_STAT_MIN),
                  (stat_texture_average.ravel() - RADIA_DYNA_MIN) / (RADIA_DYNA_MAX - RADIA_DYNA_MIN),]
    attributes = ['CON', 'DIS', 'HOM', 'ENG', 'COR', 'ASM', 'ENT']
    sub_titles = ['', '']

    from matplotlib import pyplot as plt
    fig2, axes2 = draw_radar_chart(
        data_list=RADAR_LIST,
        radar_str=attributes,
        pic_order='21',
        figure=(5, 12),
        pic_str=sub_titles,
        title='',
        norm=True
    )
    plt.show()

