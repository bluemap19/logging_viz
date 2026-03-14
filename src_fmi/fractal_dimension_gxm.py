import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from typing import Dict

import pandas as pd

from src_fmi.fmi_data_read import get_random_ele_data
from src_fmi.image_operation import show_Pic


def multifractal_analysis(image_array: np.ndarray, q_range: float = 5.0, q_step: float = 0.1) -> Dict:
    """
    多维分形分析主函数 - 基于盒子计数法计算图像的多重分形特征

    核心原理：通过不同尺度下的概率测度分布，分析图像的多尺度自相似性
    数学基础：多重分形形式化理论，基于配分函数和Legendre变换

    Args:
        image_array: 输入灰度图像数组，像素值代表局部强度或概率测度
        q_range: q值范围参数，控制分析不同奇异性强度的权重，q∈[-q_range, +q_range]
        q_step: q值采样步长，决定分析的分辨率，步长越小分析越精细
        img_size: image进行缩放的大小，一般默认为256*256，如果使用None，则不进行图像的缩放
        box_sizes: 进行image的盒计数法的盒的配置

    Returns:
        Dict: 包含完整多重分形谱特征的结果字典，关键指标包括：
            - q_values: q参数序列
            - singularity_exponents: 奇异指数α(q)，描述局部奇异性强度
            - multifractal_spectrum: 多重分形谱f(α)，描述具有给定α值的分形子集维数
            - generalized_dimensions: 广义分形维数D_q，统一描述不同q值下的分形特征
            - mass_exponents: 质量指数τ(q)，与广义维数通过τ(q) = (q-1)D_q关联
            - 其他辅助统计量和质量指标
    """
    # =========================================================================
    # 输入数据验证
    # =========================================================================
    if image_array is None or image_array.size == 0:
        raise ValueError("输入图像为空")

    # 检查图像是否全零
    if np.all(image_array == 0):
        raise ValueError("输入图像全为零，无法进行分形分析")

    # 3. 检查数据类型
    if image_array.dtype not in [np.uint8, np.float32, np.float64]:
        print(f"警告: 不支持的图像数据类型 {image_array.dtype}")
        # 尝试转换数据类型
        image_array = image_array.astype(np.float32)

    # 检查图像数值范围
    if np.min(image_array) < 0:
        print("警告: 图像包含负值，将进行绝对值处理")
        image_array = np.abs(image_array)

    # =========================================================================
    # 第一阶段：图像预处理与尺度参数设置
    # =========================================================================

    # 最小像素块大小：决定分析的最小尺度，通常设为2以便二进制分割
    minimum_pixel_block_size = 2

    # 图像标准化：将输入图像调整为统一尺寸(256×256)，确保后续多尺度分析的一致性
    # 使用双三次插值保持图像质量，避免尺度变换引入的人为效应
    processed_image = cv2.resize(image_array, (256, 256), interpolation=cv2.INTER_CUBIC)
    original_rows, original_cols = processed_image.shape

    # 计算最大尺度级数：基于图像尺寸和最小块大小的对数关系
    # 公式：p = floor(log₂(N))，其中N为图像尺寸，确保2^p ≤ N
    maximum_scale_levels = int(math.floor(math.log(original_rows) / math.log(minimum_pixel_block_size)))

    # 标准化图像尺寸：调整为2的幂次方(2^p)，便于二进制多尺度分割
    # 这是盒子计数法的基本要求，确保每个尺度下都能均匀分割
    standardized_image_size = minimum_pixel_block_size ** maximum_scale_levels
    standardized_image = cv2.resize(processed_image, (standardized_image_size, standardized_image_size), interpolation=cv2.INTER_CUBIC)

    # =========================================================================
    # 第二阶段：多尺度序列定义与存储初始化
    # =========================================================================

    # 尺度序列生成：定义分析所用的尺度序列ε = 2^(k+1), k=0,1,...,p-1
    # 每个尺度对应不同的盒子大小，实现从粗到细的多分辨率分析
    scale_sequence = [minimum_pixel_block_size ** (scale_index + 1) for scale_index in range(maximum_scale_levels)]

    # 计算最大盒子数量：最小尺度(ε=2)时的盒子总数，用于预分配存储空间
    # 公式：N_max = (L/ε_min)^2 = (2^p/2)^2 = 2^(2p-2)
    maximum_box_count = (standardized_image_size ** 2) // (minimum_pixel_block_size ** 2)

    # 初始化概率测度矩阵：存储每个尺度下每个盒子的像素强度总和
    # 维度：最大盒子数 × 尺度级数，稀疏存储优化内存使用
    box_probability_matrix = np.zeros((int(maximum_box_count), maximum_scale_levels))

    # 各尺度盒子计数数组：记录每个尺度实际使用的盒子数量
    boxes_count_per_scale = np.zeros(maximum_scale_levels)

    # 图像总强度：用于后续概率归一化的验证，确保质量守恒
    total_image_intensity = np.sum(standardized_image)

    # =========================================================================
    # 第三阶段：多尺度盒子划分与概率测度计算
    # =========================================================================

    # 遍历所有尺度级别，进行多分辨率盒子划分
    for current_scale_index in range(maximum_scale_levels):
        # 计算当前尺度下的理论盒子总数：N(ε) = (L/ε)^2
        boxes_count_per_scale[current_scale_index] = (standardized_image_size ** 2) / (
                    scale_sequence[current_scale_index] ** 2)

        box_counter = 0  # 盒子计数器，记录当前尺度实际处理的盒子数
        current_box_size = int(scale_sequence[current_scale_index])  # 当前尺度对应的盒子边长ε

        # 在图像上滑动窗口，划分盒子网格
        # 步长等于盒子大小，确保无重叠全覆盖
        for start_row in range(0, standardized_image_size - current_box_size + 1, current_box_size):
            for start_col in range(0, standardized_image_size - current_box_size + 1, current_box_size):
                # 边界检查，避免存储越界
                if box_counter < maximum_box_count:
                    # 提取当前盒子区域的图像块
                    image_block = standardized_image[start_row:start_row + current_box_size, start_col:start_col + current_box_size]

                    # 计算盒子内像素强度总和：μ_i(ε) = Σ pixel_value
                    # 这代表在尺度ε下，盒子i所包含的"质量"或概率测度
                    box_probability_matrix[box_counter, current_scale_index] = np.sum(image_block)
                    box_counter += 1

    # =========================================================================
    # 第四阶段：概率归一化处理 - 构建概率测度分布
    # =========================================================================

    # 初始化归一化概率矩阵：将绝对强度转换为相对概率测度
    # p_i(ε) = μ_i(ε) / Σμ_j(ε)，满足Σp_i(ε) = 1
    normalized_probability_matrix = np.zeros((int(maximum_box_count), maximum_scale_levels))

    for scale_index in range(maximum_scale_levels):
        # 当前尺度下的实际盒子数量
        current_scale_box_count = int((standardized_image_size ** 2) / (scale_sequence[scale_index] ** 2))

        # 计算归一化因子：当前尺度下所有盒子的总强度
        # 理论上应与图像总强度一致，验证计算正确性
        normalization_factor = np.sum(box_probability_matrix[:current_scale_box_count, scale_index])

        # 质量守恒验证：归一化因子应等于图像总强度
        if abs(normalization_factor - total_image_intensity) > 1e-5:
            print(f'警告: 检测到归一化因子差异，计算值: {normalization_factor}, 图像总强度: {total_image_intensity}')

        # 对每个盒子进行概率归一化
        for box_index in range(current_scale_box_count):
            if normalization_factor != 0:
                # 归一化概率：p_i(ε) = μ_i(ε) / Σμ_j(ε)
                normalized_probability_matrix[box_index, scale_index] = (
                        box_probability_matrix[box_index, scale_index] / normalization_factor)
            else:
                # 处理零总强度的极端情况（全黑图像）
                normalized_probability_matrix[box_index, scale_index] = 0

    # =========================================================================
    # 第五阶段：q参数序列生成与配分函数计算
    # =========================================================================

    # 生成q值序列：q参数控制对奇异性不同强度的权重
    # q > 0: 强调高概率区域(平滑区域)  q < 0: 强调低概率区域(奇异区域)
    q_values_sequence = np.arange(-q_range, q_range + q_step, q_step)
    q_values_storage = q_values_sequence.copy()  # 存储用于后续计算

    # 初始化结果存储矩阵：
    # multifractal_spectrum_f: 存储f(q,ε)，用于计算多重分形谱
    # singularity_strength_alpha: 存储α(q,ε)，用于计算奇异指数
    # partition_function_values: 存储配分函数χ(q,ε)
    multifractal_spectrum_f = np.zeros((maximum_scale_levels, len(q_values_sequence)))
    singularity_strength_alpha = np.zeros((maximum_scale_levels, len(q_values_sequence)))
    partition_function_values = np.zeros((maximum_scale_levels, len(q_values_sequence)))

    # 遍历所有尺度级别和q值，计算配分函数和相关量
    for scale_index in range(maximum_scale_levels):
        # 当前尺度下的盒子数量
        current_scale_box_count = int((standardized_image_size ** 2) / (scale_sequence[scale_index] ** 2))

        # 遍历q值序列，计算不同权重下的分形特征
        for q_index, current_q_value in enumerate(q_values_sequence):
            # =================================================================
            # 核心计算1：配分函数 χ(q,ε) = Σ [p_i(ε)]^q
            # =================================================================
            # 物理意义：q阶矩的和，反映概率测度在尺度ε下的分布特征
            # 当q=0时，χ(0,ε) = N(ε)，即盒子数量
            # 当q=1时，χ(1,ε) = 1，由概率归一化保证
            partition_function_sum = np.sum([
                normalized_probability_matrix[i, scale_index] ** current_q_value
                for i in range(current_scale_box_count)
                if normalized_probability_matrix[i, scale_index] != 0  # 避免0的q次幂
            ])

            # 初始化累积变量，用于计算加权概率测度相关的统计量
            f_alpha_numerator = 0.0  # 累积f(α)计算项：Σ μ_i(q,ε) log μ_i(q,ε)
            alpha_q_numerator = 0.0  # 累积α(q)计算项：Σ μ_i(q,ε) log p_i(ε)
            weighted_probability_sum = 0.0  # 验证加权概率测度和为1

            # 遍历当前尺度所有盒子，计算加权概率测度μ_i(q,ε)
            for box_index in range(current_scale_box_count):
                if normalized_probability_matrix[box_index, scale_index] > 0:
                    # =========================================================
                    # 核心计算2：加权概率测度 μ_i(q,ε) = [p_i(ε)]^q / χ(q,ε)
                    # =========================================================
                    # 这是多重分形形式化理论的关键概念，将原始概率重新加权
                    # μ_i(q,ε) 强调q所选择的不同奇异性区域
                    weighted_probability_measure = (
                            (normalized_probability_matrix[box_index, scale_index] ** current_q_value)
                            / partition_function_sum
                    )

                    # 累积f(α)计算项：μ_i(q,ε) * log(μ_i(q,ε))
                    # 这对应于Shannon熵的加权形式，描述分布的不确定性
                    if weighted_probability_measure > 0:
                        f_alpha_numerator += weighted_probability_measure * math.log(weighted_probability_measure)

                    # 累积α(q)计算项：μ_i(q,ε) * log(p_i(ε))
                    # 这反映在q权重下，局部奇异性强度的加权平均
                    if normalized_probability_matrix[box_index, scale_index] > 0:
                        alpha_q_numerator += (
                                weighted_probability_measure *
                                math.log(normalized_probability_matrix[box_index, scale_index])
                        )

                    # 验证加权概率测度归一化：Σ μ_i(q,ε) 应该等于1
                    weighted_probability_sum += weighted_probability_measure
                else:
                    weighted_probability_sum +=  0

            # 存储当前尺度和q值下的计算结果
            multifractal_spectrum_f[scale_index, q_index] = f_alpha_numerator
            singularity_strength_alpha[scale_index, q_index] = alpha_q_numerator
            partition_function_values[scale_index, q_index] = partition_function_sum

    # =========================================================================
    # 第六阶段：q=1特殊情况处理 - 避免数学奇异性
    # =========================================================================

    # q=1时配分函数方法出现奇点，需要特殊处理
    # 使用L'Hospital法则，D_1 = lim(ε→0) [Σ p_i log p_i] / log ε
    special_case_q1 = np.zeros(maximum_scale_levels)

    for scale_index in range(maximum_scale_levels):
        current_scale_box_count = int((standardized_image_size ** 2) / (scale_sequence[scale_index] ** 2))
        for box_index in range(current_scale_box_count):
            if normalized_probability_matrix[box_index, scale_index] > 0:
                # 计算p_i log p_i项，用于q=1时的信息维数计算
                special_case_q1[scale_index] += (
                        normalized_probability_matrix[box_index, scale_index] *
                        math.log(normalized_probability_matrix[box_index, scale_index])
                )

    # =========================================================================
    # 第七阶段：尺度对数计算 - 为log-log回归做准备
    # =========================================================================

    # 计算每个尺度的对数：log(ε)，用于后续的尺度缩放分析
    # 多重分形分析的核心是通过log-log关系的斜率提取分形维数
    logarithmic_scales = np.array([math.log(scale_value) for scale_value in scale_sequence])

    # =========================================================================
    # 第八阶段： 广义分形维数D_q计算
    # =========================================================================

    # 广义分形维数D_q统一描述不同q值下的分形特征
    # 公式：D_q = τ(q)/(q-1)，其中τ(q)通过配分函数的尺度行为得到
    generalized_fractal_dimensions = np.zeros(len(q_values_storage))

    for q_index, current_q_value in enumerate(q_values_storage):
        if abs(current_q_value - 1) > 1e-10:  # q ≠ 1 的一般情况
            # =================================================================
            # 核心计算3：通过log-log线性回归求τ(q)
            # =================================================================
            # 理论基础：配分函数满足幂律关系 χ(q,ε) ∝ ε^{τ(q)}
            # 因此 log χ(q,ε) = τ(q) log ε + constant
            # 通过线性回归的斜率得到τ(q)
            regression_line = np.polyfit(
                logarithmic_scales,
                np.log(partition_function_values[:, q_index] + 1e-10),  # 避免log(0)
                1  # 一阶线性拟合
            )

            # D_q = τ(q)/(q-1)，这是广义分形维数的定义
            generalized_fractal_dimensions[q_index] = regression_line[0] / (current_q_value - 1)

        else:  # q = 1 的特殊情况（信息维数）
            # =================================================================
            # 核心计算4：q=1时的信息维数D_1
            # =================================================================
            # 使用L'Hospital法则：D_1 = lim(ε→0) [Σ p_i log p_i] / log ε
            regression_line = np.polyfit(logarithmic_scales, special_case_q1, 1)
            generalized_fractal_dimensions[q_index] = regression_line[0]  # 斜率即为D_1

    # =========================================================================
    # 第九阶段：奇异指数α(q)和多重分形谱f(α)计算
    # =========================================================================

    # 奇异指数α(q)描述局部奇异性强度，多重分形谱f(α)描述分形子集的维数
    # 通过Legendre变换与广义维数关联：f(α) = qα - τ(q)
    singularity_exponents = np.zeros(len(q_values_storage))
    multifractal_spectrum = np.zeros(len(q_values_storage))
    alpha_regression_r2 = np.zeros(len(q_values_storage))  # 拟合优度指标
    f_spectrum_regression_r2 = np.zeros(len(q_values_storage))

    for q_index in range(len(q_values_storage)):
        # =====================================================================
        # 核心计算5：奇异指数α(q) = dτ(q)/dq
        # =====================================================================
        # 计算方法：α(q) = lim(ε→0) [Σ μ_i(q,ε) log p_i(ε)] / log ε
        # 通过α(q,ε)相对于log ε的线性回归斜率得到
        alpha_regression = np.polyfit(logarithmic_scales, singularity_strength_alpha[:, q_index], 1)
        singularity_exponents[q_index] = alpha_regression[0]  # 回归斜率即为α(q)

        # 计算拟合优度R²，评估线性关系的质量
        alpha_fitted_values = np.polyval(alpha_regression, logarithmic_scales)
        alpha_residual_sum_squares = np.sum((singularity_strength_alpha[:, q_index] - alpha_fitted_values) ** 2)
        alpha_total_sum_squares = np.sum((singularity_strength_alpha[:, q_index] - np.mean(singularity_strength_alpha[:, q_index])) ** 2)
        alpha_regression_r2[q_index] = (1 - (alpha_residual_sum_squares / alpha_total_sum_squares) if alpha_total_sum_squares != 0 else 0)

        # =====================================================================
        # 核心计算6：多重分形谱f(α) = qα(q) - τ(q)
        # =====================================================================
        # 计算方法：f(α) = lim(ε→0) [Σ μ_i(q,ε) log μ_i(q,ε)] / log ε
        # 通过f(q,ε)相对于log ε的线性回归斜率得到
        f_spectrum_regression = np.polyfit(logarithmic_scales, multifractal_spectrum_f[:, q_index], 1)
        multifractal_spectrum[q_index] = f_spectrum_regression[0]  # 回归斜率即为f(α)

        # 计算f(α)拟合的R²值
        f_spectrum_fitted_values = np.polyval(f_spectrum_regression, logarithmic_scales)
        f_spectrum_residual_sum_squares = np.sum((multifractal_spectrum_f[:, q_index] - f_spectrum_fitted_values) ** 2)
        f_spectrum_total_sum_squares = np.sum((multifractal_spectrum_f[:, q_index] - np.mean(multifractal_spectrum_f[:, q_index])) ** 2)
        f_spectrum_regression_r2[q_index] = (1 - (f_spectrum_residual_sum_squares / f_spectrum_total_sum_squares) if f_spectrum_total_sum_squares != 0 else 0)

    # =========================================================================
    # 第十阶段：质量指数τ(q)计算与结果整合
    # =========================================================================

    # 质量指数τ(q)与广义维数的关系：τ(q) = (q-1)D_q
    # 这是多重分形形式化理论的基本关系式
    mass_exponent_tau = (q_values_storage - 1) * generalized_fractal_dimensions

    # 构建完整的结果字典，包含所有多重分形特征指标
    analysis_results = {
        'q_values': q_values_storage,  # q参数序列, [-5, 5], step=0.1 (101, )
        'singularity_exponents': singularity_exponents,  # 奇异指数α(q) shape=q_values_storage.shape
        'multifractal_spectrum': multifractal_spectrum,  # 多重分形谱f(α) shape=q_values_storage.shape
        'generalized_dimensions': generalized_fractal_dimensions,  # 广义分形维数D_q shape=q_values_storage.shape
        'mass_exponents': mass_exponent_tau,  # 质量指数τ(q) shape=q_values_storage.shape
        'alpha_regression_quality': alpha_regression_r2,  # α(q)拟合质量 shape=q_values_storage.shape
        'spectrum_regression_quality': f_spectrum_regression_r2,  # f(α)拟合质量 shape=q_values_storage.shape
        'total_scale_levels': maximum_scale_levels,  # 总尺度级数    int 8
        'processed_image_size': standardized_image_size,  # 处理后的图像尺寸
        'scale_values': scale_sequence,  # 尺度序列 ndarray (256, 256)
        'partition_function': partition_function_values,  # 配分函数χ(q,ε)  nadrray shape=(maximum_scale_levels, len(q_values_storage)) (8, 101)
        'weighted_probability_alpha': singularity_strength_alpha,  # 加权概率测度相关量  nadrray shape=(maximum_scale_levels, len(q_values_storage)) (8, 101)
        'weighted_probability_f': multifractal_spectrum_f  # 加权概率测度相关量  nadrray shape=(maximum_scale_levels, len(q_values_storage)) (8, 101)
    }

    return analysis_results


def visualize_multifractal_results(analysis_results: Dict):
    """
    绘制2×2子图展示多维分形分析结果

    Args:
        analysis_results: 多维分形分析结果字典
    """
    q_values = analysis_results['q_values']
    singularity_exponents = analysis_results['singularity_exponents']
    multifractal_spectrum = analysis_results['multifractal_spectrum']
    generalized_dimensions = analysis_results['generalized_dimensions']
    mass_exponents = analysis_results['mass_exponents']

    # 设置图形参数
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    font_size=18

    # 创建2×2子图
    figure, subplot_axes = plt.subplots(2, 2, figsize=(15, 12))

    # 子图1: q-α(q)+f(q)关系
    subplot_axes[0, 0].plot(q_values, singularity_exponents, 'r-o', markersize=3, label='α(q)')
    subplot_axes[0, 0].plot(q_values, multifractal_spectrum, 'g-s', markersize=3, label='f(q)')
    subplot_axes[0, 0].set_xlabel('q', fontsize=font_size)
    subplot_axes[0, 0].set_ylabel('Value', fontsize=font_size)
    subplot_axes[0, 0].set_title('Singularity Exponent α(q) and Multifractal Spectrum f(q) vs q', fontsize=font_size)
    subplot_axes[0, 0].legend()
    subplot_axes[0, 0].grid(True, alpha=0.3)

    # 子图2: α(q)-f(α)关系及其抛物线拟合
    subplot_axes[0, 1].plot(singularity_exponents, multifractal_spectrum, 'bo', markersize=4,
                            label='Multifractal Spectrum')
    if len(singularity_exponents) > 2:
        try:
            # 抛物线拟合多重分形谱
            parabolic_coefficients = np.polyfit(singularity_exponents, multifractal_spectrum, 2)
            alpha_fit_range = np.linspace(min(singularity_exponents), max(singularity_exponents), 100)
            spectrum_fit_values = np.polyval(parabolic_coefficients, alpha_fit_range)
            subplot_axes[0, 1].plot(alpha_fit_range, spectrum_fit_values, 'r-', linewidth=2,
                                    label='Parabolic Fit')
        except Exception as e:
            print(f"抛物线拟合失败: {e}")
    subplot_axes[0, 1].set_xlabel('Singularity Exponent α(q)', fontsize=font_size)
    subplot_axes[0, 1].set_ylabel('Multifractal Spectrum f(α)', fontsize=font_size)
    subplot_axes[0, 1].set_title('Multifractal Spectrum f(α) vs Singularity Exponent α', fontsize=font_size)
    subplot_axes[0, 1].legend()
    subplot_axes[0, 1].grid(True, alpha=0.3)

    # 子图3: 广义分形维数 D_q vs q
    subplot_axes[1, 0].plot(q_values, generalized_dimensions, 'm-^', markersize=4, linewidth=2)
    subplot_axes[1, 0].set_xlabel('q', fontsize=font_size)
    subplot_axes[1, 0].set_ylabel('Generalized Dimension D(q)', fontsize=font_size)
    subplot_axes[1, 0].set_title('Generalized Fractal Dimension D(q) vs q', fontsize=font_size)
    subplot_axes[1, 0].grid(True, alpha=0.3)

    # 子图4: 质量指数 τ(q) vs q
    subplot_axes[1, 1].plot(q_values, mass_exponents, 'c-d', markersize=4, linewidth=2)
    subplot_axes[1, 1].set_xlabel('q', fontsize=font_size)
    subplot_axes[1, 1].set_ylabel('Mass Exponent τ(q)', fontsize=font_size)
    subplot_axes[1, 1].set_title('Mass Exponent τ(q) vs q', fontsize=font_size)
    subplot_axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# 测试用例
if __name__ == "__main__":
    # 创建测试图像（256×256随机纹理）
    # test_texture = cv2.imread(r"C:\Users\Maple\Documents\MATLAB\multifractal-last modified\output1.jpg", cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    test_texture, test_texture_stat, depth = get_random_ele_data()
    test_texture = test_texture.astype(np.uint8)
    test_texture = cv2.resize(test_texture, (512, 512))
    print(test_texture.shape, type(test_texture))
    show_Pic([test_texture])

    print("开始执行多维分形分析...")
    fractal_results = multifractal_analysis(test_texture)

    print(f"分析完成！总尺度级数: {fractal_results['total_scale_levels']}")
    print(f"q值分析范围: {fractal_results['q_values'][0]:.1f} 到 {fractal_results['q_values'][-1]:.1f}")

    # 输出关键分形维数指标
    q_near_zero = np.argmin(np.abs(fractal_results['q_values']))
    q_near_one = np.argmin(np.abs(fractal_results['q_values'] - 1))
    q_near_two = np.argmin(np.abs(fractal_results['q_values'] - 2))

    print(f"容量维数 D(0) = {fractal_results['generalized_dimensions'][q_near_zero]:.4f}")
    print(f"信息维数 D(1) = {fractal_results['generalized_dimensions'][q_near_one]:.4f}")
    print(f"关联维数 D(2) = {fractal_results['generalized_dimensions'][q_near_two]:.4f}")
    print(f"奇异谱宽度 Δα = {np.max(fractal_results['singularity_exponents']) - np.min(fractal_results['singularity_exponents']):.4f}")

    print(fractal_results['singularity_exponents'], fractal_results['multifractal_spectrum'])

    # 可视化分析结果
    visualize_multifractal_results(fractal_results)
