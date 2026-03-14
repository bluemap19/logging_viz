import logging
import cv2
import numpy as np
from tqdm import trange

from src_fmi.fmi_data_read import get_random_ele_data
from src_fmi.fmi_fractal_dimension import adaptive_binarization, edge_detection
from src_fmi.fractal_dimension_extended import cal_pic_fractal_dimension_extended
from src_fmi.fractal_dimension_gxm import multifractal_analysis
from src_fmi.image_operation import show_Pic
from src_plot.TEMP_8 import WellLogVisualizer
from typing import Dict, Any, Tuple

def pic_preprocess(image, processing_method='adaptive_binary'):
    # 步骤2: 图像预处理 - 根据地质特征选择合适方法
    if processing_method == 'adaptive_binary':
        # 适合分析孔隙结构、岩性边界
        processed_image = adaptive_binarization(image, 'otsu_adaptive')
    elif processing_method == 'edge_detection':
        # 适合分析裂缝、层理面等线性特征
        processed_image = edge_detection(image, 'canny_adaptive')
    elif processing_method == 'original':
        # 直接使用原始数据，保留所有信息
        processed_image = image
    else:
        processed_image = image

    return processed_image


def cal_fmi_multidimensional_fractal_spectrum(fmi_dict={}, windows_length=200, windows_step=10, processing_method='adaptive_binary'):
    """
    将NMR分形α-f谱数据转换为类似图像的二维数组
    现在NMR_dict的键是字符串类型（保留4位小数的深度）

    参数:
        NMR_dict: 字典，键为深度字符串(如"1234.5678")，值为包含'NMR_X'和'NMR_Y'的字典

    返回:
        depth_array: 深度值的一维数组
        image_array: 二维数组，行代表深度，列代表α-f谱的强度分布
    """

    # 1. 数据提取，从字典中正确提取深度、电成像图像
    data_depth = fmi_dict['depth'] if 'depth' in fmi_dict else None
    data_img = fmi_dict['image'] if 'image' in fmi_dict else None

    # 数据完整性检查
    if data_depth is None or data_img is None:
        raise ValueError("fmi_dict必须包含'depth', 'image' 两个键")
    # 数据维度一致性检查
    if data_depth.shape[0] != data_img.shape[0]:
        raise ValueError("深度数据与图像数据长度不匹配")
    if data_img.dtype != np.uint8:
        print('trans data type as uint8')
        data_img = data_img.astype(np.uint8)

    data_depth = data_depth.ravel()
    windows_num = data_depth.shape[0] // windows_step + 1

    multi_fractal_dimension_result = {}          # 存储深度-分形维数对

    # 初始化结果数组，用于构建完整深度的处理结果
    fmi_result_weight = np.zeros_like(data_img, dtype=np.float32)
    fmi_result = np.zeros_like(data_img, dtype=np.float32)

    # 滑动窗口遍历电成像数据
    for i in trange(windows_num):
        index_middle = min(i * windows_step, data_img.shape[0]-1)  # 窗口中心深度
        index_start = max(0, index_middle - windows_length // 2)  # 窗口起始索引，防止越界
        index_end = min(data_depth.shape[0], index_middle + windows_length // 2)  # 窗口结束索引，防止越界

        data_image_window = data_img[index_start:index_end]  # 提取当前窗口内的电成像数据
        depth_t = data_depth[index_middle]  # 当前窗口的中心深度

        # # 计算当前窗口的分形维数 fd_param:当前窗口的分形维数值 processed_img:经过预处理后的窗口图像，边缘检测结果，或者是二值化结果
        # processed_img = pic_preprocess(data_image_window, processing_method=processing_method)
        # analysis_results = multifractal_analysis(image_array=processed_img, q_step=0.5)
        # # 存储结果 'NMR_X', 'NMR_Y'
        # multi_fractal_dimension_result[f'{depth_t:.4f}'] = {'NMR_X':analysis_results['singularity_exponents'], 'NMR_Y':(analysis_results['multifractal_spectrum'])}

        # original differential_box edge_detection
        fd, processed_img, analysis_results = cal_pic_fractal_dimension_extended(image=data_image_window, image_shape=[256, 256], method='multifractal', processing_method=processing_method,
                                                                                 multifractal_params={'q_range': np.arange(-5, 5.5, 0.5), 'box_sizes': [2, 4, 8, 16, 32, 64, 128],
                                                                                 })
        result = np.array(analysis_results['f_alpha_spectrum'])
        multi_fractal_dimension_result[f'{depth_t:.4f}'] = {'NMR_X': result[:, 0], 'NMR_Y': result[:, 1]}

        # 将处理结果映射回原始尺寸，用于构建完整深度剖面
        processed_img = cv2.resize(processed_img, data_image_window.shape[::-1], cv2.INTER_LINEAR)
        fmi_result[index_start:index_end] += processed_img
        fmi_result_weight[index_start:index_end] += 1

    # 加权平均处理结果（权重归一化）
    fmi_result = fmi_result / (fmi_result_weight+0.01)

    return fmi_result, multi_fractal_dimension_result


def cal_fmis_fractal_dimension_extended(fmi_dict={}, windows_length=200, windows_step=10, processing_method='adaptive_binary'):
    """
    计算电成像数据的分形维数曲线 - 主控函数

    滑动窗口原理：
    通过窗口在深度方向上滑动，计算每个窗口内电成像图像的分形特征，
    形成随深度变化的分形维数曲线，反映地层复杂度的纵向变化。
    """
    # 1. 数据提取
    # 从字典中正确提取深度、动态图像、静态图像数据
    data_depth = fmi_dict['depth'] if 'depth' in fmi_dict else None
    data_imgs = fmi_dict['fmis'] if 'fmis' in fmi_dict else None
    # output_curves_list = fmi_dict['output_curves_list'] if 'output_curves_list' in fmi_dict else None

    # 数据完整性检查
    if data_depth is None or data_imgs is None:
        raise ValueError("fmi_dict必须包含'depth', 'fmis'键")

    # 数据维度一致性检查
    for i in range(len(data_imgs)):
        if data_depth.shape[0] != data_imgs[i].shape[0]:
            raise ValueError("深度数据与图像数据长度不匹配")

    fmi_multi_fde_list = []
    fmi_result_list = []
    for i in range(len(data_imgs)):
        # 2. 动态成像的分形维数计算
        fmi_result, multi_fractal_dimension_result = cal_fmi_multidimensional_fractal_spectrum(
            fmi_dict={'depth':data_depth, 'image':data_imgs[i]},
            windows_length=windows_length,  # 窗口长度：平衡纵向分辨率和统计可靠性
            windows_step=windows_step,  # 滑动步长：控制计算密度
            # edge_detection adaptive_binary original
            processing_method=processing_method,
        )
        fmi_result_list.append(fmi_result)
        fmi_multi_fde_list.append(multi_fractal_dimension_result)

    return fmi_result_list, fmi_multi_fde_list



def trans_NMR_as_Ciflog_file_type(NMR_dict: Dict[float, Any] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    将NMR分形α-f谱数据转换为类似图像的二维数组

    参数:
        NMR_dict: 字典，键为深度(float)，值为包含'NMR_X'和'NMR_Y'的字典

    返回:
        depth_array: 深度值的一维数组
        image_array: 二维数组，行代表深度，列代表α-f谱的强度分布
    """
    if NMR_dict is None or len(NMR_dict) == 0:
        return np.array([]), np.array([])

    # 1. 对深度进行排序（将字符串键转换为浮点数进行排序）
    depths_str = sorted(NMR_dict.keys(), key=lambda x: float(x))
    depth_array = np.array([float(d) for d in depths_str])  # 返回浮点数深度数组

    # 2. 确定X轴的范围和分辨率
    # 收集所有深度下的X值来确定统一的范围
    all_x_values = []
    for depth_str in depths_str:
        data = NMR_dict[depth_str]
        if 'NMR_X' in data and len(data['NMR_X']) > 0:
            all_x_values.extend(data['NMR_X'])

    if len(all_x_values) == 0:
        return np.array([]), np.array([])

    # 确定X轴的范围和分辨率
    x_min = min(all_x_values)
    x_max = max(all_x_values)

    # 设置合适的列数
    num_columns = 64

    # 创建统一的X轴网格
    x_grid = np.linspace(x_min, x_max, num_columns)

    # 3. 创建图像数组
    num_depths = len(depths_str)
    image_array = np.zeros((num_depths, num_columns))

    # 4. 对每个深度，将谱数据映射到统一的X网格上
    for i, depth_str in enumerate(depths_str):
        data = NMR_dict[depth_str]

        if 'NMR_X' in data and 'NMR_Y' in data:
            x_original = data['NMR_X']
            y_original = data['NMR_Y']

            # 确保数据是有效的
            if len(x_original) > 0 and len(y_original) > 0:
                # 计算当前深度数据在X网格上的索引范围
                min_index_current = int((np.min(x_original) - x_min) / (x_max - x_min) * (num_columns - 1))
                max_index_current = int((np.max(x_original) - x_min) / (x_max - x_min) * (num_columns - 1))

                # 确保索引在有效范围内
                min_index_current = max(0, min_index_current)
                max_index_current = min(num_columns - 1, max_index_current)

                # print(f"深度 {depth_str}: 索引范围 [{min_index_current}, {max_index_current}]")

                # 对原始数据进行排序（按X值）
                sort_idx = np.argsort(x_original)
                x_sorted = x_original[sort_idx]
                y_sorted = y_original[sort_idx]

                # 只在数据存在的范围内进行插值
                if min_index_current <= max_index_current:
                    # 提取数据存在范围内的网格点
                    valid_x_grid = x_grid[min_index_current:max_index_current + 1]

                    # 在有效范围内进行插值
                    interpolated_values = np.interp(valid_x_grid, x_sorted, y_sorted, left=y_sorted[0], right=y_sorted[-1])
                    # 将插值结果赋给图像数组的相应位置
                    image_array[i, min_index_current:max_index_current + 1] = interpolated_values

    return depth_array, image_array

# 把fde的图像格式数据，转换成相对应的谱类数据格式
def trans_fde_image_to_NMR_type(IMG_ARRAY: np.ndarray) -> Dict[str, Any]:
    NMR_DICT = {}
    for i in range(IMG_ARRAY.shape[0]):
        if i % 5 == 0:
            depth = IMG_ARRAY[i, 0]
            NMR_DICT[f'{depth:.4f}'] = {'NMR_X': np.linspace(0, 6.3, 64), 'NMR_Y': IMG_ARRAY[i, 1:]}
            NMR_DICT[f'{depth:.4f}'] = {'NMR_X': np.linspace(0, 6.3, 64), 'NMR_Y': IMG_ARRAY[i, 1:]}

    return NMR_DICT


if __name__ == '__main__':
    """
    主程序执行流程
    """
    # 1. 数据读取 - 获取电成像动态、静态数据和深度信息
    data_img_dyna, data_img_stat, data_depth = get_random_ele_data()
    print(f"数据形状: 动态{data_img_dyna.shape}, 静态{data_img_stat.shape}, 深度{data_depth.shape}")

    # 2. 分形维数计算
    fmi_result_list, fmi_multi_fde_list = cal_fmis_fractal_dimension_extended(
        fmi_dict={
            'depth': data_depth,  # 深度数据
            'fmis': [data_img_dyna.astype(np.uint8), data_img_stat.astype(np.uint8)],  # 电成像数据
        },
        windows_length=200,                 # 窗口长度：平衡纵向分辨率和统计可靠性
        windows_step=100,                    # 滑动步长：控制计算密度
        processing_method='original',       # 图像预处理的方式：自适应二值化突出岩性边界 adaptive_binary  original
    )

    depth_array, image_fde_dyna = trans_NMR_as_Ciflog_file_type(fmi_multi_fde_list[0])
    alpha_f_dyna = np.hstack((depth_array.reshape((-1, 1)), image_fde_dyna.astype(np.float32)))
    np.savetxt('alpha_f_dyna.txt', alpha_f_dyna, delimiter='\t', comments='', fmt='%.4f')

    depth_array, image_fde_stat = trans_NMR_as_Ciflog_file_type(fmi_multi_fde_list[1])
    alpha_f_stat = np.hstack((depth_array.reshape((-1, 1)), image_fde_stat.astype(np.float32)))
    np.savetxt('alpha_f_stat.txt', alpha_f_stat, delimiter='\t', comments='', fmt='%.4f')

    # show_Pic([image_fde_dyna, image_fde_stat])

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
                'image_data': [data_img_dyna, data_img_stat]+fmi_result_list,
                'title': ['FMI动态', 'FMI静态', 'DYNA_PRO', 'STAT_PRO']
            },
            NMR_dict=fmi_multi_fde_list,
            NMR_Config={'X_LOG': [False, False], 'NMR_TITLE': ['α-fα-DYNA', 'α-fα-STAT'], 'X_LIMIT':[[1.2, 4], [1.2, 4]], 'Y_scaling_factor': 2.4},
            # depth_limit_config=[320, 380],                      # 深度限制
            figsize=(12, 10)                                    # 图形尺寸
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

