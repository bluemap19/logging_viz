# -*- coding: utf-8 -*-
"""
================================================================================
电成像测井裂缝区域检测脚本
FMI/EMI Coal Crack/Cleat Detection for Electrical Microresistivity Imaging
================================================================================

功能概述：
---------
本脚本用于检测和量化电成像测井（FMI/EMI）图像中的裂缝/割理区域。
裂缝区域被定义为满足特定几何条件的连通区域（面积、长度、宽度）。

核心功能：
1. 裂缝筛选条件配置 - 支持面积、长度、宽度多条件组合
2. 连通区域检测 - 使用 scipy ndimage 进行连通分量分析
3. 几何特征计算 - 计算每个连通区域的面积、长度（高度）、宽度
4. 裂缝统计 - 统计满足条件的裂缝区域数量和总像素数
5. 滑动窗口计算 - 计算沿深度方向的裂缝含量曲线
6. 可视化 - 彩色标注不同类型的区域

裂缝筛选条件（可配置）：
- 面积条件：大于或小于阈值（可配置）
- 长度条件：y方向最大距离 > 阈值（可配置）
- 宽度条件：x方向最大距离 > 阈值（可配置）
- 逻辑关系：AND（同时满足）或 OR（任一满足）

适用场景：
- 煤系地层裂缝/割理识别
- 电成像测井图像分析
- 煤层渗透性评估
- 地质构造分析

作者：Cuka (OpenClaw Agent)
日期：2026-04-14
================================================================================
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from typing import Tuple, List, Any, Dict
import cv2
import os


# ==================== 全局配置 ====================

# 裂缝筛选默认配置
DEFAULT_CRACK_CONFIG = {
    'area_threshold': 800,       # 面积阈值（像素）
    'area_condition': 'greater',  # 面积条件：'greater'(大于) 或 'less'(小于)
    'length_threshold': 100,      # 长度阈值（y方向，最大高度，像素）
    'width_threshold': 100,       # 宽度阈值（x方向，最大宽度，像素）
    'length_condition': 'greater', # 长度条件：'greater'(大于) 或 'less'(小于)
    'width_condition': 'greater', # 宽度条件：'greater'(大于) 或 'less'(小于)
    'use_and_logic': False       # True: 所有条件同时满足(AND); False: 任一条件满足(OR)
}

# 可视化颜色配置（BGR格式）
COLOR_CONFIG = {
    'background': (0, 0, 0),           # 黑色：背景
    'crack': (0, 255, 0),              # 绿色：裂缝区域
    'non_crack': (0, 0, 255),          # 红色：非裂缝的大目标区域
    'crack_boundary': (255, 255, 0),   # 青色：裂缝边界（用于增强显示）
}


# ==================== 核心函数 ====================

def calculate_region_geometry(labeled_array: np.ndarray, label: int) -> Dict[str, Any]:
    """
    计算单个连通区域的几何特征

    参数:
    ----------
    labeled_array : np.ndarray
        标记后的数组，每个连通区域有唯一的整数标签
    label : int
        要计算的连通区域标签

    返回:
    ----------
    dict: 包含几何特征的字典
        - area: 面积（像素数）
        - length: 长度（y方向最大距离）
        - width: 宽度（x方向最大距离）
        - aspect_ratio: 长宽比
        - center_y: 中心y坐标
        - center_x: 中心x坐标
    """
    # 获取当前标签的掩码
    region_mask = (labeled_array == label)

    # 计算面积（前景像素数）
    area = np.sum(region_mask)

    # 计算边界框（y方向：行，x方向：列）
    rows = np.any(region_mask, axis=1)
    cols = np.any(region_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        # 空区域
        return {
            'area': 0,
            'length': 0,
            'width': 0,
            'aspect_ratio': 0,
            'center_y': 0,
            'center_x': 0
        }

    y_indices, = np.nonzero(rows)
    x_indices, = np.nonzero(cols)

    # 计算长度和宽度
    length = y_indices[-1] - y_indices[0] + 1
    width = x_indices[-1] - x_indices[0] + 1

    # 计算中心坐标
    center_y = (y_indices[0] + y_indices[-1]) / 2
    center_x = (x_indices[0] + x_indices[-1]) / 2

    # 计算长宽比
    aspect_ratio = length / width if width > 0 else 0

    return {
        'area': int(area),
        'length': int(length),
        'width': int(width),
        'aspect_ratio': float(aspect_ratio),
        'center_y': float(center_y),
        'center_x': float(center_x)
    }


def check_crack_conditions(geometry: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """
    检查连通区域是否满足裂缝筛选条件

    参数:
    ----------
    geometry : dict
        区域几何特征字典（由 calculate_region_geometry 返回）
    config : dict
        裂缝筛选配置字典

    返回:
    ----------
    bool: 是否满足裂缝条件
    """
    area = geometry['area']
    length = geometry['length']
    width = geometry['width']

    # 解包配置
    area_threshold = config.get('area_threshold', 800)
    area_condition = config.get('area_condition', 'greater')
    length_threshold = config.get('length_threshold', 100)
    width_threshold = config.get('width_threshold', 100)
    length_condition = config.get('length_condition', 'greater')
    width_condition = config.get('width_condition', 'greater')
    use_and_logic = config.get('use_and_logic', False)

    # 检查面积条件
    if area_condition == 'greater':
        area_ok = area > area_threshold
    else:  # 'less'
        area_ok = area < area_threshold

    # 检查长度条件
    if length_condition == 'greater':
        length_ok = length > length_threshold
    else:  # 'less'
        length_ok = length < length_threshold

    # 检查宽度条件
    if width_condition == 'greater':
        width_ok = width > width_threshold
    else:  # 'less'
        width_ok = width < width_threshold

    # 根据逻辑关系判断
    if use_and_logic:
        # AND: 所有条件同时满足
        is_crack = area_ok and length_ok and width_ok
    else:
        # OR: 任一条件满足
        is_crack = area_ok or length_ok or width_ok

    return is_crack


def detect_crack_regions(
        binary_image: np.ndarray,
        crack_config: Dict[str, Any] = None
) -> Tuple[int, np.ndarray, Dict[str, Any]]:
    """
    检测二值图像中的裂缝区域

    参数:
    ----------
    binary_image : np.ndarray
        输入的二值图像，只包含0(背景)和255(前景)两种像素值
    crack_config : dict, 可选
        裂缝筛选配置，包含：
        - area_threshold: 面积阈值（默认800）
        - area_condition: 面积条件，'greater'或'less'（默认'greater'）
        - length_threshold: 长度阈值（默认100）
        - width_threshold: 宽度阈值（默认100）
        - use_and_logic: 是否使用AND逻辑（默认False，使用OR）

    返回:
    ----------
    total_crack_pixels : int
        所有裂缝区域的总像素数
    crack_mask : np.ndarray
        裂缝区域掩码，与输入图像相同大小，裂缝区域为255，其他为0
    statistics : dict
        统计信息字典
    """
    # 使用默认配置（如果未提供）
    if crack_config is None:
        crack_config = DEFAULT_CRACK_CONFIG.copy()

    # 验证输入
    assert len(binary_image.shape) == 2, f"输入图像必须是2维数组，当前维度: {binary_image.ndim}"

    # 验证是否为二值图像
    unique_values = np.unique(binary_image)
    if not (len(unique_values) == 2 and 0 in unique_values and 255 in unique_values):
        raise ValueError("输入图像必须是二值图像，只包含0和255两种值")

    # 创建二值掩码(0和1)
    binary_mask = (binary_image == 255).astype(np.uint8)

    # 标记连通区域（8连通）
    structure = np.ones((3, 3), dtype=np.int32)
    labeled_array, num_features = ndimage.label(binary_mask, structure=structure)

    print(f"检测到 {num_features} 个连通区域")

    # 初始化结果
    crack_mask = np.zeros_like(binary_image, dtype=np.uint8)
    total_crack_pixels = 0

    # 统计信息
    statistics = {
        'total_regions': num_features,
        'crack_regions': 0,
        'non_crack_regions': 0,
        'crack_pixel_count': 0,
        'non_crack_pixel_count': 0,
        'config': crack_config,
        'region_details': []
    }

    # 遍历所有连通区域
    for label in range(1, num_features + 1):
        # 计算几何特征
        geometry = calculate_region_geometry(labeled_array, label)

        # 检查是否满足裂缝条件
        is_crack = check_crack_conditions(geometry, crack_config)

        # 记录区域信息
        region_info = {
            'label': label,
            'area': geometry['area'],
            'length': geometry['length'],
            'width': geometry['width'],
            'aspect_ratio': geometry['aspect_ratio'],
            'is_crack': is_crack
        }
        statistics['region_details'].append(region_info)

        if is_crack:
            # 标记为裂缝区域
            region_mask = (labeled_array == label)
            crack_mask[region_mask] = 255
            total_crack_pixels += geometry['area']
            statistics['crack_regions'] += 1
            statistics['crack_pixel_count'] += geometry['area']
        else:
            statistics['non_crack_regions'] += 1
            statistics['non_crack_pixel_count'] += geometry['area']

    print(f"裂缝区域数量: {statistics['crack_regions']}")
    print(f"裂缝区域总像素数: {total_crack_pixels}")
    print(f"非裂缝区域数量: {statistics['non_crack_regions']}")

    return total_crack_pixels, crack_mask, statistics


def get_crack_statistics(
        binary_image: np.ndarray,
        crack_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    获取裂缝区域的详细统计信息

    参数:
    ----------
    binary_image : np.ndarray
        输入的二值图像
    crack_config : dict, 可选
        裂缝筛选配置

    返回:
    ----------
    dict: 包含详细统计信息的字典
    """
    if crack_config is None:
        crack_config = DEFAULT_CRACK_CONFIG.copy()

    # 创建二值掩码
    binary_mask = (binary_image == 255).astype(np.uint8)

    # 标记连通区域
    structure = np.ones((3, 3), dtype=np.int32)
    labeled_array, num_features = ndimage.label(binary_mask, structure=structure)

    # 初始化统计信息
    statistics = {
        'total_regions': num_features,
        'crack_regions': 0,
        'non_crack_regions': 0,
        'crack_pixel_count': 0,
        'non_crack_pixel_count': 0,
        'config': crack_config.copy(),
        'crack_details': [],
        'non_crack_details': []
    }

    if num_features > 0:
        # 遍历所有区域
        for label in range(1, num_features + 1):
            geometry = calculate_region_geometry(labeled_array, label)
            is_crack = check_crack_conditions(geometry, crack_config)

            region_info = {
                'label': label,
                'area': geometry['area'],
                'length': geometry['length'],
                'width': geometry['width'],
                'aspect_ratio': geometry['aspect_ratio'],
                'center_y': geometry['center_y'],
                'center_x': geometry['center_x']
            }

            if is_crack:
                statistics['crack_regions'] += 1
                statistics['crack_pixel_count'] += geometry['area']
                statistics['crack_details'].append(region_info)
            else:
                statistics['non_crack_regions'] += 1
                statistics['non_crack_pixel_count'] += geometry['area']
                statistics['non_crack_details'].append(region_info)

    return statistics


def visualize_crack_regions(
        binary_image: np.ndarray,
        crack_mask: np.ndarray,
        crack_config: Dict[str, Any] = None,
        show_labels: bool = False
) -> np.ndarray:
    """
    可视化裂缝检测结果

    参数:
    ----------
    binary_image : np.ndarray
        输入的二值图像
    crack_mask : np.ndarray
        裂缝区域掩码
    crack_config : dict, 可选
        裂缝筛选配置（用于显示配置信息）
    show_labels : bool, 可选
        是否在可视化图像上显示区域标签

    返回:
    ----------
    np.ndarray: 彩色可视化图像（BGR格式）
    """
    if crack_config is None:
        crack_config = DEFAULT_CRACK_CONFIG.copy()

    height, width = binary_image.shape

    # 创建彩色图像用于可视化
    visualization = np.zeros((height, width, 3), dtype=np.uint8)

    # 设置颜色
    # 背景: 黑色
    visualization[:, :, :] = COLOR_CONFIG['background']

    # 非裂缝区域: 红色
    non_crack_areas = (binary_image == 255) & (crack_mask == 0)
    visualization[non_crack_areas] = COLOR_CONFIG['non_crack']

    # 裂缝区域: 绿色
    crack_areas = crack_mask == 255
    visualization[crack_areas] = COLOR_CONFIG['crack']

    # 如果需要显示标签
    if show_labels:
        # 获取所有区域的标签和几何信息
        binary_mask = (binary_image == 255).astype(np.uint8)
        structure = np.ones((3, 3), dtype=np.int32)
        labeled_array, num_features = ndimage.label(binary_mask, structure=structure)

        for label in range(1, num_features + 1):
            geometry = calculate_region_geometry(labeled_array, label)
            is_crack = check_crack_conditions(geometry, crack_config)

            # 在区域中心添加标签
            center_y = int(geometry['center_y'])
            center_x = int(geometry['center_x'])

            # 确保坐标在图像范围内
            if 0 <= center_y < height and 0 <= center_x < width:
                text = f"L{label}" if not is_crack else f"C{label}"
                color = (255, 255, 255)  # 白色文字

                # 添加文本标签
                cv2.putText(visualization, text, (center_x - 10, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return visualization


def visualize_crack_with_boundary(
        original_image: np.ndarray,
        crack_mask: np.ndarray
) -> np.ndarray:
    """
    可视化裂缝检测结果（在原图上叠加显示）

    参数:
    ----------
    original_image : np.ndarray
        原始灰度图像
    crack_mask : np.ndarray
        裂缝区域掩码

    返回:
    ----------
    np.ndarray: 叠加可视化图像
    """
    height, width = original_image.shape

    # 将灰度图转换为BGR
    if len(original_image.shape) == 2:
        bgr_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = original_image.copy()

    # 创建裂缝区域的半透明叠加
    overlay = bgr_image.copy()
    crack_areas = crack_mask == 255
    overlay[crack_areas] = COLOR_CONFIG['crack']  # 绿色裂缝

    # 混合原图和叠加层
    alpha = 0.5  # 透明度
    visualization = cv2.addWeighted(bgr_image, 1 - alpha, overlay, alpha, 0)

    # 绘制裂缝边界
    crack_contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(visualization, crack_contours, -1, COLOR_CONFIG['crack_boundary'], 1)

    return visualization


def cal_fmis_crack_content(
        depth_data: np.ndarray = np.array([]),
        list_fmis: List[np.ndarray] = [],
        config_windows: Dict[str, Any] = {},
        crack_config: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    计算每个深度点的裂缝含量

    参数:
    ----------
    depth_data : np.ndarray
        深度测井数据，一维数组
    list_fmis : List[np.ndarray]
        电成像mask列表，每个mask是二维数组(深度, 其他维度)
    config_windows : Dict[str, Any]
        窗口设置，包含：
        - window_length: 窗口长度（点数）
        - step: 滑动步长（点数）
        - depth_col_name: 深度列列名
        - curves_names_list: 计算得到的裂缝曲线列列名
    crack_config : dict, 可选
        裂缝筛选配置

    返回:
    ----------
    pd.DataFrame
        包含深度和各曲线裂缝含量的数据框
    """
    if crack_config is None:
        crack_config = DEFAULT_CRACK_CONFIG.copy()

    # 解包窗口配置
    window_length = config_windows.get('window_length', 200)
    step = config_windows.get('step', 1)
    depth_col_name = config_windows.get('depth_col_name', 'DEPTH')
    curves = config_windows.get('curves_names_list', ['crack_content'])

    # 参数验证
    if len(list_fmis) != len(curves):
        raise ValueError(f"list_fmis 数量 ({len(list_fmis)}) 与 curves 数量 ({len(curves)}) 不匹配")

    if len(depth_data) == 0:
        raise ValueError("深度数据不能为空")

    for i, fmi in enumerate(list_fmis):
        if len(depth_data) != fmi.shape[0]:
            raise ValueError(f"第{i}个FMI的深度维度({fmi.shape[0]})与深度数据({len(depth_data)})不匹配")

    # 验证窗口参数
    if window_length <= 0:
        raise ValueError("窗口长度必须大于0")
    if window_length > depth_data.shape[0]:
        raise ValueError("窗口长度必须小于图像的长度")
    if step <= 0:
        raise ValueError("步长必须大于0")
    if step > window_length:
        raise ValueError("步长必须小于窗口长度")

    # 计算窗口半长
    half_window = window_length // 2

    # 计算有效的深度点范围
    valid_indices = range(half_window, len(depth_data) - half_window, step)

    # 准备存储结果
    results = []

    print(f"数据总点数: {len(depth_data)}")
    print(f"窗口长度: {window_length}")
    print(f"滑动步长: {step}")
    print(f"有效深度点数: {len(valid_indices)}")

    # 滑动窗口计算
    for center_idx in valid_indices:
        # 计算窗口上下边界
        start_idx = center_idx - half_window
        end_idx = center_idx + half_window

        # 获取中心点深度
        depth_value = depth_data[center_idx]

        # 创建记录字典
        record = {depth_col_name: depth_value}

        # 计算每条曲线的裂缝含量
        for curve_name, fmi in zip(curves, list_fmis):
            # 提取当前窗口的FMI数据
            fmi_window = fmi[start_idx:end_idx]

            # 检测裂缝区域
            _, crack_mask, _ = detect_crack_regions(fmi_window, crack_config)

            # 计算裂缝含量（裂缝像素占比）
            crack_pixels = np.count_nonzero(crack_mask == 255)
            total_pixels = fmi_window.size

            if total_pixels > 0:
                crack_content = crack_pixels / total_pixels
            else:
                crack_content = 0.0

            # 记录结果
            record[curve_name] = crack_content

        results.append(record)

    # 转换为DataFrame
    df_result = pd.DataFrame(results)

    return df_result


# ==================== 测试和示例代码 ====================

def create_test_image_with_cracks(
        image_size: Tuple[int, int] = (500, 500),
        num_large_cracks: int = 3,
        num_small_regions: int = 10,
        seed: int = 42
) -> np.ndarray:
    """
    创建包含裂缝的测试图像

    参数:
    ----------
    image_size : tuple
        图像尺寸 (height, width)
    num_large_cracks : int
        大裂缝数量
    num_small_regions : int
        小目标区域数量
    seed : int
        随机种子

    返回:
    ----------
    np.ndarray: 测试二值图像
    """
    np.random.seed(seed)

    height, width = image_size
    test_image = np.zeros((height, width), dtype=np.uint8)

    # 创建大裂缝（细长形状）
    for i in range(num_large_cracks):
        # 裂缝起始位置
        start_y = np.random.randint(50, height - 150)
        start_x = np.random.randint(50, width - 50)

        # 裂缝长度和宽度
        crack_length = np.random.randint(120, 200)  # 长度 > 100
        crack_width = np.random.randint(5, 15)       # 宽度较小

        # 绘制裂缝（垂直或倾斜）
        angle = np.random.uniform(-0.3, 0.3)  # 倾斜角度

        if angle == 0:
            # 垂直裂缝
            end_y = min(start_y + crack_length, height)
            end_x = start_x + crack_width
            test_image[start_y:end_y, start_x:end_x] = 255
        else:
            # 倾斜裂缝
            for dy in range(crack_length):
                y = start_y + dy
                x_offset = int(dy * angle)
                x = start_x + x_offset
                if 0 <= y < height and 0 <= x < width:
                    test_image[y, x:x + crack_width] = 255

    # 创建小目标区域（面积 < 800）
    for i in range(num_small_regions):
        start_y = np.random.randint(10, height - 20)
        start_x = np.random.randint(10, width - 20)

        # 小区域大小（面积 < 800）
        reg_height = np.random.randint(10, 30)
        reg_width = np.random.randint(10, 30)
        area = reg_height * reg_width

        if area < 800:  # 确保面积小于阈值
            end_y = min(start_y + reg_height, height)
            end_x = min(start_x + reg_width, width)
            test_image[start_y:end_y, start_x:end_x] = 255

    # 创建一些中等大小的区域（不应被识别为裂缝）
    for i in range(5):
        start_y = np.random.randint(10, height - 50)
        start_x = np.random.randint(10, width - 50)

        reg_height = np.random.randint(30, 60)
        reg_width = np.random.randint(30, 60)

        end_y = min(start_y + reg_height, height)
        end_x = min(start_x + reg_width, width)
        test_image[start_y:end_y, start_x:end_x] = 255

    return test_image


def create_test_image_realistic(
        image_size: Tuple[int, int] = (500, 500),
        num_cracks: int = 5,
        seed: int = 42
) -> np.ndarray:
    """
    创建更真实的测试图像（模拟电成像裂缝）

    参数:
    ----------
    image_size : tuple
        图像尺寸
    num_cracks : int
        裂缝数量
    seed : int
        随机种子

    返回:
    ----------
    np.ndarray: 测试二值图像
    """
    np.random.seed(seed)

    height, width = image_size
    test_image = np.zeros((height, width), dtype=np.uint8)

    # 创建不同方向的裂缝
    for i in range(num_cracks):
        # 随机选择裂缝方向
        direction = np.random.choice(['vertical', 'horizontal', 'diagonal'])

        # 裂缝起点
        if direction == 'vertical':
            start_y = np.random.randint(20, height - 120)
            start_x = np.random.randint(width // 4, 3 * width // 4)
            crack_length = np.random.randint(120, 200)
            crack_width = np.random.randint(3, 12)

            for dy in range(crack_length):
                y = start_y + dy
                if y < height:
                    x_end = min(start_x + crack_width, width)
                    test_image[y, start_x:x_end] = 255

        elif direction == 'horizontal':
            start_y = np.random.randint(height // 4, 3 * height // 4)
            start_x = np.random.randint(20, width - 120)
            crack_length = np.random.randint(120, 200)
            crack_width = np.random.randint(3, 12)

            for dx in range(crack_length):
                x = start_x + dx
                if x < width:
                    y_end = min(start_y + crack_width, height)
                    test_image[start_y:y_end, x] = 255

        else:  # diagonal
            start_y = np.random.randint(20, height - 150)
            start_x = np.random.randint(20, width - 150)
            crack_length = np.random.randint(150, 250)
            crack_width = np.random.randint(2, 8)

            for d in range(crack_length):
                y = start_y + d
                x = start_x + int(d * 0.8)
                if y < height and x < width:
                    for w in range(crack_width):
                        if x + w < width:
                            test_image[y, x + w] = 255

    # 添加一些非裂缝的大区域
    for i in range(3):
        start_y = np.random.randint(20, height - 80)
        start_x = np.random.randint(20, width - 80)
        reg_size = np.random.randint(60, 120)

        end_y = min(start_y + reg_size, height)
        end_x = min(start_x + reg_size, width)
        test_image[start_y:end_y, start_x:end_x] = 255

    # 添加一些小的干扰区域
    for i in range(20):
        y = np.random.randint(0, height)
        x = np.random.randint(0, width)
        size = np.random.randint(3, 15)
        test_image[y:y + size, x:x + size] = 255

    return test_image


def run_basic_tests():
    """
    运行基础测试
    """
    print("=" * 70)
    print("裂缝检测基础测试")
    print("=" * 70)

    # 创建测试图像
    print("\n创建测试图像...")
    test_image = create_test_image_with_cracks(
        image_size=(500, 500),
        num_large_cracks=5,
        num_small_regions=10,
        seed=42
    )

    print(f"测试图像尺寸: {test_image.shape}")
    print(f"前景像素总数: {np.sum(test_image == 255)}")

    # 打印配置信息
    print("\n裂缝筛选配置:")
    for key, value in DEFAULT_CRACK_CONFIG.items():
        print(f"  {key}: {value}")

    # 测试1: 检测裂缝区域
    print("\n" + "=" * 50)
    print("测试1: 检测裂缝区域")
    print("=" * 50)

    total_pixels, crack_mask, stats = detect_crack_regions(test_image, DEFAULT_CRACK_CONFIG)

    print(f"\n统计结果:")
    print(f"  总连通区域数: {stats['total_regions']}")
    print(f"  裂缝区域数: {stats['crack_regions']}")
    print(f"  非裂缝区域数: {stats['non_crack_regions']}")
    print(f"  裂缝总像素数: {stats['crack_pixel_count']}")

    # 测试2: 获取详细统计信息
    print("\n" + "=" * 50)
    print("测试2: 获取详细统计信息")
    print("=" * 50)

    detailed_stats = get_crack_statistics(test_image, DEFAULT_CRACK_CONFIG)

    print(f"\n裂缝区域详情 (前5个):")
    for i, detail in enumerate(detailed_stats['crack_details'][:5]):
        print(f"  区域{detail['label']}: 面积={detail['area']}, "
              f"长度={detail['length']}, 宽度={detail['width']}, "
              f"长宽比={detail['aspect_ratio']:.2f}")

    # 测试3: 可视化结果
    print("\n" + "=" * 50)
    print("测试3: 生成可视化图像")
    print("=" * 50)

    # 基本可视化
    viz_image = visualize_crack_regions(test_image, crack_mask, DEFAULT_CRACK_CONFIG)

    # 保存可视化结果
    output_dir = r"C:\Users\Maple\.qclaw\workspace\coal_fmi_seam\results\crack_detection"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "crack_detection_basic.png")
    cv2.imwrite(output_path, viz_image)
    print(f"\n基本可视化已保存: {output_path}")
    print("  绿色: 裂缝区域")
    print("  红色: 非裂缝区域（其他大目标）")
    print("  黑色: 背景")

    # 叠加可视化
    print("\n生成叠加可视化...")
    gray_image = test_image  # 对于二值图像，直接用作灰度图
    overlay_viz = visualize_crack_with_boundary(gray_image, crack_mask)

    overlay_path = os.path.join(output_dir, "crack_detection_overlay.png")
    cv2.imwrite(overlay_path, overlay_viz)
    print(f"叠加可视化已保存: {overlay_path}")


def run_comparison_tests():
    """
    运行对比测试（不同配置下的结果对比）
    """
    print("\n" + "=" * 70)
    print("配置对比测试")
    print("=" * 70)

    # 创建更真实的测试图像
    print("\n创建真实感测试图像...")
    test_image = create_test_image_realistic(
        image_size=(500, 500),
        num_cracks=8,
        seed=123
    )

    print(f"测试图像尺寸: {test_image.shape}")
    print(f"前景像素总数: {np.sum(test_image == 255)}")

    # 配置1: 默认配置（OR逻辑）
    print("\n配置1: 默认配置（OR逻辑）")
    config1 = DEFAULT_CRACK_CONFIG.copy()
    config1['use_and_logic'] = False

    _, crack_mask1, stats1 = detect_crack_regions(test_image, config1)
    print(f"  裂缝区域数: {stats1['crack_regions']}")
    print(f"  裂缝像素数: {stats1['crack_pixel_count']}")

    # 配置2: AND逻辑
    print("\n配置2: AND逻辑（所有条件同时满足）")
    config2 = DEFAULT_CRACK_CONFIG.copy()
    config2['use_and_logic'] = True

    _, crack_mask2, stats2 = detect_crack_regions(test_image, config2)
    print(f"  裂缝区域数: {stats2['crack_regions']}")
    print(f"  裂缝像素数: {stats2['crack_pixel_count']}")

    # 配置3: 仅基于面积（> 800）
    print("\n配置3: 仅基于面积条件")
    config3 = {
        'area_threshold': 800,
        'area_condition': 'greater',
        'length_threshold': 100,
        'width_threshold': 100,
        'length_condition': 'greater',
        'width_condition': 'greater',
        'use_and_logic': False
    }

    _, crack_mask3, stats3 = detect_crack_regions(test_image, config3)
    print(f"  裂缝区域数: {stats3['crack_regions']}")
    print(f"  裂缝像素数: {stats3['crack_pixel_count']}")

    # 生成对比可视化
    output_dir = r"C:\Users\Maple\.qclaw\workspace\coal_fmi_seam\results\crack_detection"
    os.makedirs(output_dir, exist_ok=True)

    # 创建对比图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 原图
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title('Original Binary Image', fontsize=12)
    axes[0].axis('off')

    # 配置1结果
    axes[1].imshow(cv2.cvtColor(visualize_crack_regions(test_image, crack_mask1, config1), cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'OR Logic\n({stats1["crack_regions"]} cracks)', fontsize=12)
    axes[1].axis('off')

    # 配置2结果
    axes[2].imshow(cv2.cvtColor(visualize_crack_regions(test_image, crack_mask2, config2), cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'AND Logic\n({stats2["crack_regions"]} cracks)', fontsize=12)
    axes[2].axis('off')

    # 配置3结果
    axes[3].imshow(cv2.cvtColor(visualize_crack_regions(test_image, crack_mask3, config3), cv2.COLOR_BGR2RGB))
    axes[3].set_title(f'Area Only\n({stats3["crack_regions"]} cracks)', fontsize=12)
    axes[3].axis('off')

    plt.tight_layout()

    comparison_path = os.path.join(output_dir, "crack_detection_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"\n对比图已保存: {comparison_path}")


# 导入matplotlib用于对比图
import matplotlib.pyplot as plt


def run_length_width_tests():
    """
    运行长度和宽度筛选测试
    """
    print("\n" + "=" * 70)
    print("长度和宽度筛选测试")
    print("=" * 70)

    # 创建测试图像
    test_image = create_test_image_realistic(seed=456)

    print(f"\n测试图像尺寸: {test_image.shape}")

    # 测试不同长度阈值
    print("\n测试不同长度阈值:")
    for length_th in [50, 100, 150, 200]:
        config = DEFAULT_CRACK_CONFIG.copy()
        config['length_threshold'] = length_th
        config['use_and_logic'] = True  # 使用AND逻辑

        _, _, stats = detect_crack_regions(test_image, config)
        print(f"  长度阈值={length_th}: 裂缝数={stats['crack_regions']}, "
              f"像素数={stats['crack_pixel_count']}")

    # 测试不同宽度阈值
    print("\n测试不同宽度阈值:")
    for width_th in [5, 10, 20, 50]:
        config = DEFAULT_CRACK_CONFIG.copy()
        config['width_threshold'] = width_th
        config['use_and_logic'] = True

        _, _, stats = detect_crack_regions(test_image, config)
        print(f"  宽度阈值={width_th}: 裂缝数={stats['crack_regions']}, "
              f"像素数={stats['crack_pixel_count']}")


# ==================== 主函数入口 ====================

if __name__ == "__main__":
    # 运行基础测试
    run_basic_tests()

    # 运行对比测试
    run_comparison_tests()

    # 运行长度宽度测试
    run_length_width_tests()

    print("\n" + "=" * 70)
    print("所有测试完成!")
    print("=" * 70)
