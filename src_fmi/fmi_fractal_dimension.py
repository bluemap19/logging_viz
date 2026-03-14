import logging
import pandas as pd
from src_fmi.fmi_data_read import get_random_ele_data
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from src_fmi.image_operation import show_Pic
from src_logging.logging_interpolation import ConventionalLogInterpolator
from src_plot.TEMP_4 import WellLogVisualizer


def adaptive_binarization(image, method='otsu_adaptive'):
    """
    自适应二值化处理 - 将灰度图像转换为二值图像

    原理说明：
    电成像数据通常包含不同电阻率的地层信息，二值化可以将连续的电导率/电阻率值
    转换为明显的边界，便于后续的分形特征提取。

    参数:
        image: 输入灰度图像
        method: 二值化方法选择

    方法对比:
        - 'otsu': 全局Otsu阈值，适合直方图双峰明显的图像
        - 'adaptive_gaussian': 局部自适应阈值，适合光照不均的图像
        - 'otsu_adaptive': 结合两者优点，先全局确定大致范围，再局部细化
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == 'otsu':
        # 全局Otsu二值化 - 基于图像直方图自动确定最佳阈值
        # 原理：寻找使类间方差最大的阈值，适用于双峰直方图
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == 'adaptive_gaussian':
        # 自适应高斯阈值 - 对每个像素邻域计算局部阈值
        # 原理：使用高斯加权平均，适合电成像中电阻率渐变的地层
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    elif method == 'otsu_adaptive':
        # 结合Otsu的自适应方法 - 综合全局和局部信息
        # 步骤1: 高斯滤波去噪，消除电成像数据中的随机噪声
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # 步骤2: Otsu全局阈值，确定大致的电阻率分界
        _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 步骤3: 自适应局部阈值，捕捉细节变化
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # 步骤4: 逻辑与操作结合两种结果，保留共同识别为特征的区域
        binary = cv2.bitwise_and(otsu_thresh, adaptive)

    return binary


def edge_detection(image, method='canny_adaptive'):
    """
    自适应边缘检测 - 提取电成像中的地层边界和裂缝特征

    原理说明：
    边缘检测可以突出电成像中的地层界面、裂缝、孔洞等不连续特征，
    这些特征对分形维数计算有重要贡献。

    参数:
        image: 输入灰度图像
        method: 边缘检测方法
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯去噪 - 消除电成像测量中的高频噪声，保留真实地质特征
    # 核大小(5,5)和sigma=0是经验值，平衡去噪和边缘保持
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    if method == 'canny_fixed':
        # 固定阈值Canny边缘检测
        edges = cv2.Canny(blurred, 50, 150)

    elif method == 'canny_adaptive':
        # 自适应Canny阈值 - 根据图像统计特性自动确定阈值
        # 原理：使用图像中值作为基准，上下浮动一定比例作为高低阈值
        median = np.median(blurred)  # 中值对异常值不敏感，适合电成像数据
        sigma = 0.33  # 经验系数，控制阈值范围
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        edges = cv2.Canny(blurred, lower, upper)

    elif method == 'sobel':
        # Sobel边缘检测 - 基于一阶梯度
        # 原理：分别计算x和y方向的梯度，合成边缘强度
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)  # x方向梯度
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)  # y方向梯度
        edges = cv2.magnitude(sobelx, sobely)  # 梯度幅值
        edges = np.uint8(edges)

    return edges


def box_counting_dimension(binary_image, box_sizes=None):
    """
    使用盒计数法计算二值图像的分形维数

    分形维数原理：
    分形维数D衡量的是复杂形状的空间填充能力，满足关系：N(ε) ~ ε^(-D)
    其中N(ε)是覆盖图形所需边长为ε的盒子数。

    对于电成像数据，分形维数可以反映：
    - 高维数：复杂的孔隙结构、裂缝发育
    - 低维数：均质、简单的岩性特征
    """
    if box_sizes is None:
        # 盒子尺寸序列，按2的幂次变化，确保多尺度分析
        box_sizes = [2, 4, 8, 16, 32, 64, 128]

    # 确保输入是二值图像(0和1)
    if binary_image.max() > 1:
        binary_image = (binary_image > 0).astype(np.uint8)

    height, width = binary_image.shape
    counts = []  # 存储每个尺度下的盒子计数

    for box_size in box_sizes:
        # 跳过过大的盒子尺寸
        if box_size >= min(height, width):
            continue

        # 计算图像可以划分的盒子网格
        h_boxes = height // box_size
        w_boxes = width // box_size

        count = 0
        # 遍历所有盒子
        for i in range(h_boxes):
            for j in range(w_boxes):
                # 提取当前盒子区域
                box = binary_image[i * box_size:(i + 1) * box_size,
                      j * box_size:(j + 1) * box_size]
                # 如果盒子中包含目标像素(非零)，则计数+1
                if np.any(box > 0):
                    count += 1

        counts.append(count)

    # 线性回归计算分形维数
    if len(counts) < 2:
        return 0  # 数据不足，返回无效值

    # 对盒子尺寸和计数取对数：log(N) = -D * log(ε) + C
    log_sizes = np.log([1 / s for s in box_sizes[:len(counts)]])  # log(1/ε)
    log_counts = np.log(counts)  # log(N)

    # 移除无穷大或NaN值
    valid_idx = np.isfinite(log_counts) & np.isfinite(log_sizes)
    if np.sum(valid_idx) < 2:
        return 0  # 有效数据点不足

    log_sizes = log_sizes[valid_idx].reshape(-1, 1)
    log_counts = log_counts[valid_idx]

    # 线性回归：斜率即为分形维数D
    reg = LinearRegression()
    reg.fit(log_sizes, log_counts)

    return reg.coef_[0]  # 返回分形维数


def differential_box_counting_dimension(image, box_sizes=None):
    """
    使用差分盒计数法计算灰度图像的分形维数

    与盒计数法的区别：
    - 盒计数法：只关心盒子是否包含目标(二值)
    - 差分盒计数法：考虑盒子内的灰度变化，更适合电成像的连续电阻率数据

    原理：N(ε) = Σ(max - min)/ε，其中max和min是盒子内的最大最小灰度值
    """
    if box_sizes is None:
        box_sizes = [2, 4, 8, 16, 32, 64, 128]

    # 转换为灰度图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape
    counts = []

    for box_size in box_sizes:
        if box_size >= min(height, width):
            continue

        h_boxes = height // box_size
        w_boxes = width // box_size

        total_count = 0
        for i in range(h_boxes):
            for j in range(w_boxes):
                # 提取当前盒子
                box = image[i * box_size:(i + 1) * box_size,
                      j * box_size:(j + 1) * box_size]

                if box.size > 0:
                    max_val = np.max(box)
                    min_val = np.min(box)
                    # 关键步骤：计算该盒子覆盖的灰度级数
                    # 反映了该区域电阻率的变化范围
                    n_r = max(1, (max_val - min_val) / box_size)
                    total_count += n_r

        counts.append(total_count)

    # 线性回归计算分形维数（同盒计数法）
    if len(counts) < 2:
        return 0

    log_sizes = np.log([1 / s for s in box_sizes[:len(counts)]])
    log_counts = np.log(counts)

    valid_idx = np.isfinite(log_counts) & np.isfinite(log_sizes)
    if np.sum(valid_idx) < 2:
        return 0

    log_sizes = log_sizes[valid_idx].reshape(-1, 1)
    log_counts = log_counts[valid_idx]

    reg = LinearRegression()
    reg.fit(log_sizes, log_counts)

    return reg.coef_[0]


def cal_pic_fractal_dimension(data_image_window, image_shape=[256, 256],
                              method='differential_box', processing_method='adaptive_binary'):
    """
    计算单个图像窗口的分形维数 - 核心处理函数

    处理流程：
    1. 图像尺寸标准化 → 2. 图像预处理 → 3. 分形维数计算
    """
    # 步骤1: 调整图像尺寸 - 确保不同深度窗口具有可比性
    # INTER_LINEAR插值平衡速度和质量，适合电成像数据
    image = cv2.resize(data_image_window, image_shape, cv2.INTER_LINEAR)

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

    # 步骤3: 分形维数计算 - 根据预处理结果选择算法
    if method == 'box_counting':
        # 适合二值化后的图像
        fd = box_counting_dimension(processed_image)
    elif method == 'differential_box':
        # 适合灰度图像，考虑电阻率变化
        fd = differential_box_counting_dimension(processed_image)
    else:
        fd = differential_box_counting_dimension(processed_image)

    return fd, processed_image


def cal_fmi_fractal_dimension(fmi_dict={}, windows_length=200, windows_step=10,
                              image_shape=[256, 256], method='differential_box',
                              processing_method='adaptive_binary'):
    """
    计算电成像数据的分形维数曲线 - 主控函数

    滑动窗口原理：
    通过窗口在深度方向上滑动，计算每个窗口内电成像图像的分形特征，
    形成随深度变化的分形维数曲线，反映地层复杂度的纵向变化。
    """
    # 1. 数据提取
    # 从字典中正确提取深度、电成像图像
    data_depth = fmi_dict['depth'] if 'depth' in fmi_dict else None
    data_img = fmi_dict['image'] if 'image' in fmi_dict else None
    out_curve_name = fmi_dict['out_curve_name'] if 'out_curve_name' in fmi_dict else None

    # 数据完整性检查
    if data_depth is None or data_img is None:
        raise ValueError("fmi_dict必须包含'depth', 'image' 两个键")
    # 数据维度一致性检查
    if data_depth.shape[0] != data_img.shape[0]:
        raise ValueError("深度数据与图像数据长度不匹配")

    data_depth = data_depth.ravel()
    windows_num = data_depth.shape[0] // windows_step

    df_fd = []  # 存储深度-分形维数对

    # 初始化结果数组，用于构建完整深度的处理结果
    fmi_result_weight = np.zeros_like(data_img, dtype=np.float32)
    fmi_result = np.zeros_like(data_img, dtype=np.float32)

    # 滑动窗口遍历电成像数据
    for i in range(windows_num):
        index_middle = i * windows_step       # 窗口中心深度
        index_start = max(0, index_middle - windows_length // 2)    # 窗口起始索引，防止越界
        index_end = min(data_depth.shape[0], index_middle + windows_length // 2)    # 窗口结束索引，防止越界

        data_image_window = data_img[index_start:index_end]     # 提取当前窗口内的电成像数据
        depth_t = data_depth[index_middle]  # 当前窗口的中心深度

        # 计算当前窗口的分形维数 fd_param:当前窗口的分形维数值 processed_img:经过预处理后的窗口图像，边缘检测结果，或者是二值化结果
        fd_param, processed_img = cal_pic_fractal_dimension(
            data_image_window, image_shape, method, processing_method)

        # 存储结果
        df_fd.append([depth_t, fd_param])

        # 将处理结果映射回原始尺寸，用于构建完整深度剖面
        processed_img = cv2.resize(processed_img, data_image_window.shape[::-1], cv2.INTER_LINEAR)
        fmi_result[index_start:index_end] += processed_img
        fmi_result_weight[index_start:index_end] += 1

    # 加权平均处理结果（这里可能有误，应该是权重归一化）
    fmi_result = fmi_result / fmi_result_weight

    # 转换为DataFrame便于处理
    df_fd = pd.DataFrame(np.array(df_fd), columns=['depth', out_curve_name])

    # 使用插值器将稀疏的窗口计算结果插值到每个深度点
    # 原理：窗口计算得到的是离散点，需要通过插值获得连续曲线
    interpolator = ConventionalLogInterpolator(method='pchip')  # PCHIP保持单调性
    df_fd = interpolator.interpolate_logs(df_fd, target_length=data_img.shape[0], depth_col='depth')

    return df_fd, fmi_result


def cal_fmis_fractal_dimension(fmi_dict={}, windows_length=200, windows_step=10,
                              image_shape=[256, 256], method='differential_box',
                              processing_method='adaptive_binary'):
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
    output_curves_list = fmi_dict['output_curves_list'] if 'output_curves_list' in fmi_dict else None

    # 数据完整性检查
    if data_depth is None or data_imgs is None:
        raise ValueError("fmi_dict必须包含'depth', 'fmis', 'output_curves_list' 键")
    if len(data_imgs) != len(output_curves_list):
        raise ValueError("fmi_dict的'fmis', 'output_curves_list'必须等数量键")

    # 数据维度一致性检查
    for i in range(len(data_imgs)):
        if data_depth.shape[0] != data_imgs[i].shape[0]:
            raise ValueError("深度数据与图像数据长度不匹配")

    df_all = None
    fmi_preprocess_list = []
    for i in range(len(data_imgs)):
        # 2. 动态成像的分形维数计算
        df_fd, fmi_result = cal_fmi_fractal_dimension(
            fmi_dict={'depth':data_depth, 'image':data_imgs[i], 'out_curve_name':output_curves_list[i]},
            windows_length=windows_length,  # 窗口长度：平衡纵向分辨率和统计可靠性
            windows_step=windows_step,  # 滑动步长：控制计算密度
            method=method,
            # edge_detection adaptive_binary
            processing_method=processing_method,
            image_shape=image_shape,
        )
        df_fd = df_fd.rename(columns={'df_param': output_curves_list[i]})
        fmi_preprocess_list.append(fmi_result)
        if df_all is None:
            df_all = df_fd
        else:
            df_all = pd.concat([df_all, df_fd[output_curves_list[i]]], axis=1)

    return df_all, fmi_preprocess_list


if __name__ == '__main__':
    """
    主程序执行流程
    """
    # 1. 数据读取 - 获取电成像动态、静态数据和深度信息
    data_img_dyna, data_img_stat, data_depth = get_random_ele_data()
    print(f"数据形状: 动态{data_img_dyna.shape}, 静态{data_img_stat.shape}, 深度{data_depth.shape}")

    # # 2. 分形维数计算
    # df_fd, processed_imgs = cal_fmi_fractal_dimension(
    #     data_depth,
    #     data_img_stat,  # 使用静态电成像数据（噪声较小）
    #     windows_length=100,  # 窗口长度：平衡纵向分辨率和统计可靠性
    #     windows_step=5,  # 滑动步长：控制计算密度
    #     method='differential_box',  # 差分盒计数法适合灰度电成像
    #     # edge_detection adaptive_binary
    #     processing_method='edge_detection',  # 自适应二值化突出岩性边界
    # )
    # 2. 分形维数计算
    df_fd, fmi_result = cal_fmis_fractal_dimension(
        fmi_dict={
            'depth': data_depth,  # 深度数据
            'fmis': [data_img_dyna, data_img_stat],  # 电成像数据
            'output_curves_list': ['fd_dyna', 'fd_stat'], # 输出曲线配置
        },
        windows_length=200,  # 窗口长度：平衡纵向分辨率和统计可靠性
        windows_step=40,  # 滑动步长：控制计算密度
        method='differential_box',  # 差分盒计数法适合灰度电成像
        # edge_detection adaptive_binary
        processing_method='adaptive_binary',  # 自适应二值化突出岩性边界

    )

    # 3. 结果统计
    print(f"计算完成，共{len(df_fd)}个数据点")
    print(f"动态分形维数范围: {df_fd['fd_dyna'].min():.4f} - {df_fd['fd_dyna'].max():.4f}")
    print(f"静态分形维数范围: {df_fd['fd_stat'].min():.4f} - {df_fd['fd_stat'].max():.4f}")

    # 4. 可视化展示
    print("创建可视化器...")
    visualizer = WellLogVisualizer()
    try:
        # 启用详细日志
        logging.getLogger().setLevel(logging.INFO)

        # 综合可视化：分形维数曲线 + 原始电成像 + 处理结果
        visualizer.visualize(
            data=df_fd,
            depth_col='depth',
            curve_cols=['fd_dyna', 'fd_stat'],  # 分形维数曲线
            type_cols=[],  # 岩性分类（可选）
            fmi_dict={
                'depth': df_fd.depth.values,
                'image_data': [data_img_dyna, data_img_stat, fmi_result[0], fmi_result[1]],
                'title': ['FMI动态', 'FMI静态', '分形预处理结果_DYNA', '分形预处理结果_STAT']
            },
            figsize=(12, 8)
        )

        # 性能统计
        stats = visualizer.get_performance_stats()
        print("性能统计:", stats)

    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理资源
        visualizer.close()