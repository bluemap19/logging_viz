import logging
import numpy as np
import pandas as pd
from src_well_data_base.data_logging_FMI import DataFMI
from src_well_data_base.data_logging_well import DATA_WELL


def user_specific_test():
    """
    用户特定测试 - 使用用户提供的文件路径
    """
    print("\n" + "=" * 60)
    print("用户特定测试")
    print("=" * 60)

    # FOLDER_PATH = r'F:\logging_workspace\禄探'
    # FOLDER_PATH = r'F:\logging_workspace\SIMU5-Crack-Hole'
    # FOLDER_PATH = r'F:\logging_workspace\樊页2HF'
    # FOLDER_PATH = r'F:\logging_workspace\樊页3HF'
    # FOLDER_PATH = r'F:\logging_workspace\FY1-12'
    # FOLDER_PATH = r'F:\logging_workspace\FY1-15'
    FOLDER_PATH = r'F:\logging_workspace\simu_beddings'
    WINDOWS_LENGTH = 40
    WINDOWS_STEP = 2

    well = DATA_WELL(FOLDER_PATH)

    summary_temp = well.well_summary()
    for k, val in summary_temp.items():
        print(k, val)

    path_logging_target = well.search_logging_path_list(new_kw=['120', 'TEXTURE', 'logging'])
    print(path_logging_target)
    # path_table_target = well.search_table_path_list(new_kw=['table'])
    # print(path_table_target)
    path_fmi_dyna_target = well.search_fmi_path_list(new_kw=["DYNA"])
    print(path_fmi_dyna_target)
    path_fmi_stat_target = well.search_fmi_path_list(new_kw=['STAT'])
    print(path_fmi_stat_target)

    texture_all = well.get_FMI_textures(texture_config={
            'level': 16,  # 灰度级别
            'distance': [2, 4],  # 像素距离
            'angles': [0, np.pi / 2],  # 角度方向
            'windows_length': WINDOWS_LENGTH,  # 窗口长度
            'windows_step': WINDOWS_STEP  # 滑动步长
    },
    # path_config={
    #     # 'path_dyna': r'F:\logging_workspace\simu_beddings\noise_00\fmi_dyna_simu_bedding.txt',
    #     # 'path_stat': r'F:\logging_workspace\simu_beddings\noise_00\fmi_stat_simu_bedding.txt',
    #     # 'path_dyna': r'F:\logging_workspace\simu_beddings\different_r\fmi_dyna_simu_bedding.txt',
    #     # 'path_stat': r'F:\logging_workspace\simu_beddings\different_r\fmi_stat_simu_bedding.txt',
    # }
    )
    print(texture_all.describe())




if __name__ == '__main__':
    """
    主程序入口
    执行顺序：
    1. 综合测试（使用模拟数据）
    2. 用户特定测试（使用用户提供的文件路径）
    """

    # 设置日志级别
    logging.basicConfig(level=logging.INFO)

    # 执行用户特定测试
    user_specific_test()

    print("=" * 60)