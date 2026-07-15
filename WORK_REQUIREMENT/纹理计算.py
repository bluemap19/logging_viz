import copy

import numpy as np

from src_fmi.fmi_data_save import fmi_data_save
from src_fmi.fmi_pyrite_content import cal_fmis_pyrite_content, count_small_target_areas
from src_fmi.fmi_segmentation import cal_fmis_segmentation
from src_fmi.image_operation import show_Pic
from src_logging.logging_combine import combine_logging_data
from src_logging.logging_interpolation import ConventionalLogInterpolator
from src_plot.well_logging_viz.data_manager import LoggingDataManager
from src_plot.well_logging_viz.data_visulization import WellLogVisualizer
from src_well_data.data_logging_well import DATA_WELL

# import warnings
# warnings.filterwarnings("ignore", message=".*TripleDES.*")
# warnings.filterwarnings("ignore", message=".*Blowfish.*")

if __name__ == '__main__':
    # 井设置
    path_well = r'F:\logging_workspace\塬22'
    # path_well = r'F:\logging_workspace\姬119H2'
    data_well = DATA_WELL(path_well)

    print(data_well.well_path)
    print(data_well.well_summary())
    print(data_well.well_summary()['paths_fmi'])

    # # 电成像数据读取
    # fmi_dyna, depth_dyna = data_well.get_FMI(key='F:\\logging_workspace\\塬22\\塬22_FMI_DYNA.txt')
    # fmi_stat, depth_stat = data_well.get_FMI(key='F:\\logging_workspace\\塬22\\塬22_FMI_STAT.txt')
    # # fmi_dyna, depth_dyna = data_well.get_FMI(key='F:\\logging_workspace\\姬119H2\\姬119H2_FMI_DYNA.txt')
    # # fmi_stat, depth_stat = data_well.get_FMI(key='F:\\logging_workspace\\姬119H2\\姬119H2_FMI_STAT.txt')
    # print(fmi_dyna.shape, depth_dyna.shape, type(fmi_stat))
    # print(fmi_stat.shape, depth_stat.shape, type(fmi_stat))

    texture_all = data_well.get_FMI_textures(texture_config={
            'level': 16,  # 灰度级别
            'distance': [2, 4],  # 像素距离
            'angles': [0, np.pi / 2],  # 角度方向
            'windows_length': 120,  # 窗口长度
            'windows_step': 10  # 滑动步长
    })
    print(texture_all.describe())

    # texture_all = well.get_FMI_textures(texture_config={
    #         'level': 16,  # 灰度级别
    #         'distance': [2, 4],  # 像素距离
    #         'angles': [0, np.pi / 2],  # 角度方向
    #         'windows_length': 120,  # 窗口长度
    #         'windows_step': 10  # 滑动步长
    # })