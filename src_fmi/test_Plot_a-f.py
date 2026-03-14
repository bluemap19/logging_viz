import logging

import numpy as np

from src_fmi.fmi_data_read import get_random_ele_data
from src_plot.TEMP_8 import WellLogVisualizer

if __name__ == '__main__':
    data_img_dyna, data_img_stat, data_depth = get_random_ele_data()
    print(f"数据形状: 动态{data_img_dyna.shape}, 静态{data_img_stat.shape}, 深度{data_depth.shape} form {data_depth[0][0]} -> {data_depth[-1][0]}")

    NMR_DYNA = np.loadtxt("alpha_f_dyna.txt", skiprows=0, delimiter="\t")
    NMR_STAT = np.loadtxt("alpha_f_stat.txt", skiprows=0, delimiter="\t")

    NMR_DYNA_DICT = {}
    NMR_STAT_DICT = {}
    for i in range(NMR_DYNA.shape[0]):
        if i%5 == 0:
            depth = NMR_DYNA[i, 0]
            NMR_DYNA_DICT[f'{depth:.4f}'] = {'NMR_X': np.linspace(0, 6.3, 64), 'NMR_Y':NMR_DYNA[i, 1:]}
            NMR_STAT_DICT[f'{depth:.4f}'] = {'NMR_X': np.linspace(0, 6.3, 64), 'NMR_Y':NMR_STAT[i, 1:]}

    print(NMR_DYNA_DICT)
    print(NMR_STAT_DICT)

    # 使用类接口进行可视化
    print("创建可视化器...")
    visualizer = WellLogVisualizer()
    try:
        # 启用详细日志级别
        logging.getLogger().setLevel(logging.INFO)

        # 执行可视化
        visualizer.visualize(
            logging_dict=None,
            fmi_dict={  # FMI图像数据
                'depth': data_depth,
                'image_data': [data_img_dyna, data_img_stat],
                'title': ['FMI动态', 'FMI静态']
            },
            NMR_dict=[NMR_DYNA_DICT, NMR_STAT_DICT],
            NMR_Config={'X_LOG': [False, False], 'NMR_TITLE': ['α-fα-DYNA', 'α-fα-STAT'],
                        'X_LIMIT': [[0, 6.4], [0, 6.4]], 'Y_scaling_factor': 2.5},
            # depth_limit_config=[320, 380],                      # 深度限制
            figsize=(12, 10)  # 图形尺寸
        )

        # 显示性能统计
        stats = visualizer.get_performance_stats()
        print("性能统计:", stats)

    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()  # 打印完整错误堆栈
    finally:
        # 清理资源
        visualizer.close()