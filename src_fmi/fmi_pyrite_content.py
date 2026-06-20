import numpy as np
import pandas as pd
from scipy import ndimage
from typing import Tuple, List, Any, Dict


def count_small_target_areas(
        binary_image: np.ndarray,
        area_threshold: int = 20
) -> Tuple[int, np.ndarray]:
    """
    统计二值图像中小面积目标区域的数量

    参数:
    ----------
    binary_image : np.ndarray
        输入的二值图像，只包含0(背景)和255(前景)两种像素值
    area_threshold : int, 可选
        面积阈值，默认20像素，小于此面积的连通区域被视为目标区域

    返回:
    ----------
    total_target_pixels : int
        所有目标区域的总像素数
    target_mask : np.ndarray
        目标区域掩码，与输入图像相同大小，目标区域为255，其他为0

    注意:
    ----------
    1. 函数假设输入图像已经是二值图像(0和255)
    2. 使用8连通(默认)进行连通区域标记
    3. 返回的目标区域包括所有面积小于阈值的连通区域
    4. 大面积区域(>=阈值)会被忽略
    """
    assert len(binary_image.shape) == 2, f"输入图像必须是2维数组，当前维度: {binary_image.ndim}"

    # 验证输入图像是否为二值图像
    if not np.array_equal(np.unique(binary_image), np.array([0, 255])):
        raise ValueError("输入图像必须是二值图像，只包含0和255两种值")

    # 创建二值掩码(0和1)，255对应1
    binary_mask = (binary_image.astype(np.uint8) == 255)

    # 使用连通分量分析标记图像中的所有连通区域
    # structure参数定义了连通性(这里使用8连通)
    structure = np.ones((3, 3), dtype=np.int32)  # 8连通
    # 对输入的二进制图像（通常是布尔数组）进行连通区域标记。这个函数能够识别图像中相连的“真”区域，并为每个连通区域分配一个唯一的标签。
    labeled_array, num_features = ndimage.label(binary_mask, structure=structure)

    print(f"检测到 {num_features} 个连通区域")

    # 计算每个连通区域的面积(像素数)
    # 通过计算每个标签的像素数量得到
    if num_features > 0:
        # 使用bincount计算每个标签的像素数
        # 注意: 标签0是背景，从1开始是各个连通区域
        areas = np.bincount(labeled_array.flatten())

        # 创建目标区域掩码(初始化为全0)
        target_mask = np.zeros_like(binary_image, dtype=np.uint8)
        total_target_pixels = 0

        # 遍历所有连通区域(从1开始，0是背景)
        for label in range(1, num_features + 1):
            area = areas[label]

            # 判断是否为小面积目标区域
            if area < area_threshold:
                # 获取当前标签对应的像素位置
                region_mask = (labeled_array == label)

                # 将目标区域添加到掩码中
                target_mask[region_mask] = 255

                # 累加目标像素数
                total_target_pixels += area

                # print(f"区域 {label}: 面积 = {area} 像素, 小于阈值 {area_threshold}，标记为目标区域")
            else:
                # print(f"区域 {label}: 面积 = {area} 像素, 大于等于阈值 {area_threshold}，忽略")
                pass

        print(f"目标区域总像素数: {total_target_pixels}")
        return total_target_pixels, target_mask

    else:
        # 没有检测到任何连通区域
        print("未检测到任何连通区域")
        return 0, np.zeros_like(binary_image, dtype=np.uint8)


def get_target_statistics(
        binary_image: np.ndarray,
        area_threshold: int = 20
) -> dict:
    """
    获取目标区域的详细统计信息

    参数:
    ----------
    binary_image : np.ndarray
        输入的二值图像
    area_threshold : int, 可选
        面积阈值，默认20像素

    返回:
    ----------
    dict: 包含目标区域详细统计信息的字典
    """

    # 创建二值掩码
    binary_mask = (binary_image == 255).astype(np.uint8)

    # 标记连通区域
    structure = np.ones((3, 3), dtype=np.int32)
    labeled_array, num_features = ndimage.label(binary_mask, structure=structure)

    # 计算统计信息
    statistics = {
        "total_regions": num_features,
        "area_threshold": area_threshold,
        "target_regions": 0,
        "large_regions": 0,
        "target_pixel_count": 0,
        "large_region_pixel_count": 0,
        "region_details": []
    }

    if num_features > 0:
        # 计算每个区域的面积
        areas = np.bincount(labeled_array.flatten())

        # 遍历所有区域
        for label in range(1, num_features + 1):
            area = areas[label]
            is_target = area < area_threshold

            # 记录区域信息
            region_info = {
                "label": label,
                "area": int(area),
                "is_target": bool(is_target)
            }
            statistics["region_details"].append(region_info)

            # 更新统计
            if is_target:
                statistics["target_regions"] += 1
                statistics["target_pixel_count"] += area
            else:
                statistics["large_regions"] += 1
                statistics["large_region_pixel_count"] += area

    return statistics


def visualize_areas(
        binary_image: np.ndarray,
        area_threshold: int = 20
) -> np.ndarray:
    """
    可视化不同大小的连通区域

    参数:
    ----------
    binary_image : np.ndarray
        输入的二值图像
    area_threshold : int, 可选
        面积阈值，默认20像素

    返回:
    ----------
    np.ndarray: 彩色可视化图像
    """
    import cv2

    # 获取目标区域掩码
    _, target_mask = count_small_target_areas(binary_image, area_threshold)

    # 创建彩色图像用于可视化
    height, width = binary_image.shape
    visualization = np.zeros((height, width, 3), dtype=np.uint8)

    # 设置颜色
    # 大面积区域: 红色
    # 小面积目标区域: 绿色
    # 背景: 黑色

    # 背景保持黑色
    visualization[:, :, :] = 0

    # 标记大面积区域(红色)
    large_areas = (binary_image == 255) & (target_mask == 0)
    visualization[large_areas] = [0, 0, 255]  # BGR格式: 红色

    # 标记目标区域(绿色)
    target_areas = target_mask == 255
    visualization[target_areas] = [0, 255, 0]  # BGR格式: 绿色

    return visualization


def cal_fmis_pyrite_content(
        depth_data: np.ndarray=np.array([]),
        list_fmis: List[np.ndarray] = [],
        config_windows:Dict[str, Any] = {},
):
    """
    计算每个深度点的黄铁矿含量

    参数:
    ----------
    depth_data : np.ndarray
        深度测井数据，一维数组
    list_fmis : List[np.ndarray]
        电成像mask列表，每个mask是二维数组(深度, 其他维度)
    curves : List[str]
        对应的曲线名称列表
    config_windows : Dict[str, Any]
        窗口设置，包含：
        - window_length: 窗口长度（点数）
        - step: 滑动步长（点数）
        - depth_col_name: 深度列列名,String
        - curves_names_list: 计算得到的黄铁矿曲线列列名, List[String] : ['pyrite_fmi']

    返回:
    ----------
    pd.DataFrame
        包含深度和各曲线黄铁矿含量的数据框

    注意:
    ----------
    1. 使用滑动窗口法计算每个深度点的黄铁矿含量
    2. 黄铁矿含量 = 窗口中目标像素占比
    3. 输出深度是每个窗口的中心点深度
    """
    window_length = config_windows.get('window_length', 200)
    step = config_windows.get('step', 1)
    depth_col_name = config_windows.get('depth_col_name', 'DEPTH')
    curves = config_windows.get('curves_names_list', ['pyrite_fmi'])

    # 参数验证
    if len(list_fmis) != len(curves):
        raise ValueError(f"list_fmis 数量 ({len(list_fmis)}) 与 curves 数量 ({len(curves)}) 不匹配")

    if len(depth_data) == 0:
        raise ValueError("深度数据不能为空")

    for i, fmi in enumerate(list_fmis):
        if len(depth_data) != fmi.shape[0]:
            raise ValueError(f"第{i}个FMI的深度维度({fmi.shape[0]})与深度数据({len(depth_data)})不匹配")

    # 验证参数
    if window_length <= 0:
        raise ValueError("窗口长度必须大于0")
    if window_length > depth_data.shape[0]:
        raise ValueError("窗厂必须小于图像的长度，一般推荐为200")
    if step <= 0:
        raise ValueError("步长必须大于0")
    if step > window_length:
        raise ValueError("步长必须小于窗长，一般推荐为1")

    # 计算窗口半长
    half_window = window_length // 2

    # 计算有效的深度点范围，从第half_window个点开始，到第len(depth_data)-half_window-1个点结束，步长为step
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

        # 计算每条曲线的黄铁矿含量
        for curve_name, fmi in zip(curves, list_fmis):
            # 提取当前窗口的FMI数据
            fmi_window = fmi[start_idx:end_idx]

            # 计算黄铁矿含量（目标像素占比 * 100%）
            # 注意：假设目标像素值为255，背景为0
            target_pixels = np.count_nonzero(fmi_window == 255)
            total_pixels = fmi_window.size

            if total_pixels > 0:
                pyrite_content = (target_pixels / total_pixels)
            else:
                pyrite_content = 0.0

            # 记录结果
            record[curve_name] = pyrite_content

        results.append(record)

    # 转换为DataFrame
    df_result = pd.DataFrame(results)

    return df_result

# 示例和测试代码
if __name__ == "__main__":
    # 创建一个测试图像
    # 背景为0，大面积区域为255，小面积目标区域为255
    test_image = np.zeros((100, 100), dtype=np.uint8)

    # 添加一个大面积区域(50x50)
    test_image[10:60, 10:60] = 255

    # 添加几个小面积目标区域
    test_image[5:7, 80:85] = 255  # 2x5 = 10 像素
    test_image[70:75, 70:72] = 255  # 5x2 = 10 像素
    test_image[80:85, 20:25] = 255  # 5x5 = 25 像素(刚好超过阈值)

    print("=" * 50)
    print("测试1: 统计小面积目标区域")
    print("=" * 50)

    # 统计小面积目标区域
    total_pixels, target_mask = count_small_target_areas(test_image, area_threshold=20)

    print(f"\n目标区域总像素数: {total_pixels}")
    print(f"目标区域形状: {target_mask.shape}")

    # 获取详细统计信息
    print("\n" + "=" * 50)
    print("测试2: 获取详细统计信息")
    print("=" * 50)

    stats = get_target_statistics(test_image, area_threshold=20)

    print(f"总连通区域数: {stats['total_regions']}")
    print(f"目标区域数(面积<{stats['area_threshold']}): {stats['target_regions']}")
    print(f"大面积区域数(面积>={stats['area_threshold']}): {stats['large_regions']}")
    print(f"目标区域总像素数: {stats['target_pixel_count']}")
    print(f"大面积区域总像素数: {stats['large_region_pixel_count']}")

    # 可视化结果
    print("\n" + "=" * 50)
    print("测试3: 生成可视化图像")
    print("=" * 50)

    # 注意: 可视化需要OpenCV，如果没有安装，可以跳过
    try:
        viz_image = visualize_areas(test_image, area_threshold=20)

        # 保存可视化结果
        import cv2

        cv2.imwrite("area_visualization.png", viz_image)
        print("可视化图像已保存为 'area_visualization.png'")
        print("红色: 大面积区域")
        print("绿色: 目标区域")
        print("黑色: 背景")
    except ImportError:
        print("OpenCV未安装，跳过可视化步骤")

    # 测试不同阈值的影响
    print("\n" + "=" * 50)
    print("测试4: 不同阈值的影响")
    print("=" * 50)

    thresholds = [10, 20, 30]
    for threshold in thresholds:
        total_pixels, _ = count_small_target_areas(test_image, area_threshold=threshold)
        print(f"阈值={threshold}时，目标区域总像素数: {total_pixels}")