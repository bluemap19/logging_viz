import logging

import numpy as np
import pandas as pd

from src_fmi.fmi_fractal_dimension_extended_calculate import trans_fde_image_to_NMR_type
from src_plot.TEMP_9 import WellLogVisualizer
from src_well_data_base.data_logging_FMI import DataFMI


def user_specific_test():
    """
    用户特定测试 - 使用用户提供的文件路径
    """
    print("\n" + "=" * 60)
    print("用户特定测试")
    print("=" * 60)

    # FOLDER_PATH = r'F:\logging_workspace\禄探'
    FOLDER_PATH = r'C:\Users\purem\Desktop\logging_workspace\禄探'
    # FOLDER_PATH = r'F:\logging_workspace\桃镇1H'

    # 用户提供的测试用例
    test_case_DYNA = {
        'path_fmi': FOLDER_PATH + r'\禄探_DYNA.txt',
        # 'path_fmi': FOLDER_PATH + r'\桃镇1H_DYNA_FULL_TEST.txt',
        'fmi_charter': 'DYNA'
    }
    test_case_STAT = {
        'path_fmi': FOLDER_PATH + r'\禄探_STAT.txt',
        # 'path_fmi': FOLDER_PATH + r'\桃镇1H_STAT_FULL_TEST.txt',
        'fmi_charter': 'STAT'
    }

    print(f"测试文件: {test_case_DYNA['path_fmi']} + {test_case_STAT['path_fmi']}")
    print(f"仪器: {test_case_DYNA['fmi_charter']} + {test_case_STAT['fmi_charter']}")
    print("-" * 50)

    try:
        # 创建FMI处理器实例
        test_FMI_DYNA = DataFMI(
            path_fmi=test_case_DYNA['path_fmi'],
            fmi_charter=test_case_DYNA['fmi_charter']
        )
        test_FMI_STAT = DataFMI(
            path_fmi=test_case_STAT['path_fmi'],
            fmi_charter=test_case_STAT['fmi_charter']
        )

        # # 执行用户要求的操作序列
        # print(">>> 执行空白条带删除...")
        # test_FMI_DYNA.ele_stripes_delete()
        # test_FMI_STAT.ele_stripes_delete()

        print(">>> 获取数据...")
        fmi_data_dyna, depth_data = test_FMI_DYNA.get_data()
        fmi_data_stat, depth_data = test_FMI_STAT.get_data()

        print(f"FMI数据形状: {fmi_data_dyna.shape}")
        print(f"深度数据形状: {depth_data.shape}")

        FMI_FDE_DYNA = test_FMI_DYNA.get_fmi_fde(config_fde={'windows_length': 150, 'windows_step': 50, 'processing_method': 'original'})
        FMI_FDE_STAT = test_FMI_STAT.get_fmi_fde(config_fde={'windows_length': 150, 'windows_step': 50, 'processing_method': 'original'})
        # print(FMI_FDE_DYNA.shape)
        # FMI_FDE_DICT = trans_fde_image_to_NMR_type(FMI_FDE_DYNA)

        visualizer = WellLogVisualizer()
        # 执行可视化
        visualizer.visualize(
            logging_dict=None,
            fmi_dict={  # FMI图像数据
                'depth': test_FMI_DYNA._data_depth,
                'image_data': [test_FMI_DYNA._data_fmi, test_FMI_STAT._data_fmi],
                'title': ['FMI动态', 'FMI静态']
            },
            NMR_dict={  # NMR数据
                'depth': FMI_FDE_DYNA[:, 0],
                'nmr_data': [FMI_FDE_DYNA[:, 1:], FMI_FDE_STAT[:, 1:]],
                'title': ['NMR动态', 'NMR静态']
            },
            NMR_CONFIG={'X_LOG': [False, False], 'NMR_TITLE': ['α-fα-DYNA', 'α-fα-STAT'],
                        'X_LIMIT': [[0, 6.4], [0, 6.4]], 'Y_scaling_factor': 15, 'JUMP_POINT':10},
            # depth_limit_config=[320, 380],                      # 深度限制
            figsize=(12, 10)  # 图形尺寸
        )

        # 显示性能统计
        stats = visualizer.get_performance_stats()
        print("性能统计:", stats)

    except FileNotFoundError:
        print(f"文件不存在: {test_case_DYNA['path_fmi']}")
        print("跳过该测试用例...")
    except Exception as e:
        print(f"测试失败: {e}")
        print("错误详情:", str(e))



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

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)