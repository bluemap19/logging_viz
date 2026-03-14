import matplotlib.pyplot as plt
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
from src_fmi.fmi_data_read import get_random_ele_data
from src_fmi.fmi_fractal_dimension import adaptive_binarization, edge_detection, box_counting_dimension, \
    differential_box_counting_dimension
from matplotlib.gridspec import GridSpec
import numpy as np
import cv2
from scipy import stats
import matplotlib.pyplot as plt


def multifractal_analysis(image, q_range=np.arange(-10, 11, 1), box_sizes=None):
    """
    计算图像的多维分形（多重分形）谱

    参数:
        image: 输入灰度图像
        q_range: 矩阶数范围，q>1强调高强度区域，q<1强调低强度区域
        box_sizes: 盒子尺寸序列
        method: 计算方法 ('standard'或'fixed_mass')

    返回:
        f_alpha_spectrum: 多重分形谱 f(α)
        alpha_q: 奇异性指数序列
        tau_q: 质量指数函数
        D_q: 广义分形维数谱
    """

    # 转换为灰度图像并归一化测度
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 将图像灰度值转换为概率测度（总和为1）
    measure = image.astype(np.float64)
    measure = measure - np.min(measure)  # 确保非负
    measure_total = np.sum(measure)
    if measure_total > 0:
        measure = measure / measure_total  # 归一化为概率测度
    else:
        measure = np.ones_like(measure) / measure.size  # 如果全零，则均匀分布

    height, width = measure.shape

    if box_sizes is None:
        # 盒子尺寸序列，覆盖多个数量级
        box_sizes = [2, 4, 8, 16, 32, 64, 128]

    # 初始化存储数组
    partition_functions = {q: [] for q in q_range}  # 配分函数 χ(q, ε)
    partition_functions_q1 = []

    # print(f"开始多重分形分析，q范围: {q_range[0]} 到 {q_range[-1]}")

    # 步骤1: 计算不同尺度下的配分函数
    for box_size in box_sizes:
        if box_size >= min(height, width):
            continue

        h_boxes = height // box_size
        w_boxes = width // box_size

        for q in q_range:
            partition_sum = 0.0

            for i in range(h_boxes):
                for j in range(w_boxes):
                    # 提取当前盒子
                    box_measure = measure[i * box_size:(i + 1) * box_size, j * box_size:(j + 1) * box_size]

                    if np.sum(box_measure) > 0:
                        # 计算盒子的测度和
                        mu = np.sum(box_measure)

                        # 添加数值稳定性处理
                        if mu <= 1e-15:  # 忽略极小的测度值
                            continue

                        # 对于不同的q值，采用不同的数值稳定处理方法
                        if abs(q - 1) < 1e-10:  # q ≈ 1
                            # q=1时的特殊处理，避免数值问题
                            partition_sum += mu * np.log(mu) if mu > 1e-10 else 0
                            # partition_sum += np.log(mu)/np.log(box_size)
                        else:
                            partition_sum += mu ** q
            # q=1时，Dq计算需要使用特殊的partition_sum，partition_sum=sum(μ*log(μ))，其数值远小于0，需要单独拎出来进行处理
            if abs(q-1) < 1e-10:
                partition_functions_q1.append(partition_sum)

            # 这个主要是为了绘制图好看，绘制q-partition_sum的散点图-拟合结果图
            if partition_sum < 0:
                partition_sum = 1e-10

            partition_functions[q].append(partition_sum)

    # 步骤2: 计算质量指数 τ(q)
    tau_q = {}
    D_q = {}  # 广义分形维数
    reg_results = {}

    for q in q_range:
        if len(partition_functions[q]) < 2:
            continue

        # 对配分函数取对数  ln(1/[2, 4, 6, 8, 10, 12, 14, 16, 32])
        log_sizes = np.log([s for s in box_sizes[:len(partition_functions[q])]])

        # 数值稳定性处理：确保配分函数均为正数
        log_chi = []
        valid_chi = []
        valid_sizes = []
        for i, chi_val in enumerate(partition_functions[q]):
            if abs(q-1) < 1e-10:
                log_chi.append(partition_functions_q1[i])
                valid_chi.append(partition_functions_q1[i])
                valid_sizes.append(log_sizes[i])
            else:
                if chi_val > 1e-15:  # 只处理正值，把所有的正直取出来，进行后面的线性拟合，用来计算不同q下的分形
                    log_chi.append(np.log(chi_val))
                    valid_chi.append(chi_val)
                    valid_sizes.append(log_sizes[i])

        if len(valid_chi) < 2:
            continue

        log_sizes_valid = np.array(valid_sizes).reshape(-1, 1)
        log_chi_valid = np.array(log_chi)

        # 线性回归计算τq以及Dq
        try:
            # 线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes_valid.ravel(), log_chi_valid.ravel())

            # 计算广义分形维数
            if abs(q - 1.0) < 1e-10:    # q == 1
                # q=1时，D_q[q]为log(box_sizes)-partition_sum的斜率，但是q-τq为一条y=0的直线，各算各的，不能当做一回事一起计算
                tau_q[q] = 0.0
                intercept = 0.0
                D_q[q] = slope
            else:
                tau_q[q] = slope
                D_q[q] = tau_q[q] / (q - 1)

            # 存储回归质量信息
            reg_results[q] = {
                'r_squared': r_value ** 2,
                'slope': tau_q[q],
                'intercept': intercept,
            }

        except Exception as e:
            print(f"q={q}时回归计算失败: {e}")
            continue

    # 步骤3: 计算奇异性指数 α(q) 和多重分形谱 f(α)
    if len(tau_q) < 3:
        print("有效数据点不足，无法计算多重分形谱")
        return None, None

    # # 步骤3: 通过数值微分计算奇异性指数 α(q) 和多重分形谱 f(α)
    # 按q值排序
    q_values_sorted = sorted(tau_q.keys())
    tau_values = np.array([tau_q[q] for q in q_values_sorted])

    # 数值微分计算α(q)，添加平滑处理
    try:
        if len(q_values_sorted) >= 3:
            # 使用中心差分法，更稳定的数值微分
            alpha_q = np.zeros_like(tau_values)
            n = len(q_values_sorted)

            # 端点使用前向/后向差分
            alpha_q[0] = (tau_values[1] - tau_values[0]) / (q_values_sorted[1] - q_values_sorted[0])
            alpha_q[-1] = (tau_values[-1] - tau_values[-2]) / (q_values_sorted[-1] - q_values_sorted[-2])

            # 内部点使用中心差分
            for i in range(1, n - 1):
                alpha_q[i] = (tau_values[i + 1] - tau_values[i - 1]) / (q_values_sorted[i + 1] - q_values_sorted[i - 1])

            # 计算 f(α) = q * α(q) - τ(q)
            f_alpha = np.array(q_values_sorted) * alpha_q - tau_values

            # 构建多重分形谱，过滤无效值
            valid_indices = np.isfinite(alpha_q) & np.isfinite(f_alpha)
            alpha_q_valid = alpha_q[valid_indices]
            f_alpha_valid = f_alpha[valid_indices]

            if len(alpha_q_valid) == 0:
                return None, None
            f_alpha_spectrum = list(zip(alpha_q_valid, f_alpha_valid))

            # 使用f(α)谱的最大值作为代表性分形维数
            f_alpha_values = [point[1] for point in f_alpha_spectrum]
            fd = np.max(f_alpha_values) if f_alpha_values else 0

            multifractal_result = {
                'f_alpha_spectrum': f_alpha_spectrum,
                'alpha_q': alpha_q,
                'tau_q': tau_q,
                'D_q': D_q,
                'regression_results': reg_results,
                'spectrum_width': np.max(alpha_q) - np.min(alpha_q) if len(alpha_q) > 0 else 0,
                'q_range': q_range,
                'box_sizes': box_sizes,
                'Z_q_data': partition_functions  # 保存原始的配分函数数据
            }
            return fd, multifractal_result
        else:
            return None, None

    except Exception as e:
        print(f"计算多重分形谱时出错: {e}")
        return None, None



def cal_pic_fractal_dimension_extended(image, image_shape=[256, 256],
                                       method='differential_box',
                                       processing_method='adaptive_binary',
                                       multifractal_params=None):
    """
    扩展的分形维数计算函数 - 优化版

    改进点:
    1. 增加输入验证
    2. 改进错误处理
    3. 优化多重分形参数默认值
    4. 增加结果验证
    """

    # 输入验证
    if image is None or image.size == 0:
        raise ValueError("输入图像数据为空")

    # 调整图像尺寸
    if image.shape[:2] != tuple(image_shape):
        image = cv2.resize(image, tuple(image_shape), interpolation=cv2.INTER_AREA)
    else:
        image = image

    # 图像预处理
    if processing_method == 'adaptive_binary':
        image = adaptive_binarization(image, 'otsu_adaptive')
    elif processing_method == 'edge_detection':
        image = edge_detection(image, 'canny_adaptive')


    try:
        # 分形分析
        if method == 'multifractal':
            # 多重分形分析
            if multifractal_params is None:
                multifractal_params = {
                    'q_range': np.arange(-5, 6, 1),
                    'box_sizes': [2, 4, 8, 16, 32, 64]  # 更合理的尺度序列
                }

            try:
                fd, multifractal_result = multifractal_analysis(image, q_range=multifractal_params['q_range'], box_sizes=multifractal_params['box_sizes'])
                return fd, image, multifractal_result
            except Exception as e:
                print(f"多重分形分析失败: {e}")
                # 降级到普通分形分析
                fd = differential_box_counting_dimension(image)
                multifractal_result = None
        else:
            # 单分形分析
            if method == 'box_counting': #盒计数法
                fd = box_counting_dimension(image)
            else:  # differential_box
                fd = differential_box_counting_dimension(image)
            multifractal_result = None

        return fd, image, multifractal_result

    except Exception as e:
        print(f"分形分析过程中出错: {e}")
        raise




def plot_multifractal_spectrum(multifractal_result, title="多重分形谱"):
    """
        绘制多重分形分析结果 - 六子图版本

        参数:
            multifractal_result: 多重分形结果字典
            title: 图像标题
        """
    if multifractal_result is None:
        print("多重分形结果为空，无法绘图")
        return

    # 创建图形时启用constrained_layout，这是解决tight_layout警告的推荐方法
    fig = plt.figure(figsize=(22, 14), constrained_layout=True)
    fig.suptitle(f'{title}', fontsize=16, fontweight='bold')

    # 使用GridSpec创建2x3布局
    gs = GridSpec(2, 3, figure=fig, wspace=0.05, hspace=0.05)

    # 子图1: 配分函数与尺度关系 (0,0)
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_partition_function(ax1, multifractal_result)

    # 子图2: 质量指数τ(q) (0,1)
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_tau_q(ax2, multifractal_result)

    # 子图3: 广义分形维数D(q) (0,2)
    ax3 = fig.add_subplot(gs[0, 2])
    _plot_D_q(ax3, multifractal_result)

    # 子图4: q-alpha_q和q-f_alpha双Y轴图 (1,0)
    ax4 = fig.add_subplot(gs[1, 0])
    _plot_q_alpha_f_alpha(ax4, multifractal_result)

    # 子图5: 多重分形谱f(α) (1,1)
    ax5 = fig.add_subplot(gs[1, 1])
    _plot_f_alpha_spectrum(ax5, multifractal_result)

    # 子图6: 空白图，可添加统计信息 (1,2)
    ax6 = fig.add_subplot(gs[1, 2])
    _plot_statistical_info(ax6, multifractal_result)

    plt.show()

    # 输出数值统计信息
    _print_statistical_info(multifractal_result)


def _plot_q_alpha_f_alpha(ax, multifractal_result):
    """
    绘制q-alpha_q和q-f_alpha的关系图（共享Y轴版本）

    参数:
        ax: 子图坐标轴
        multifractal_result: 多重分形结果字典，包含以下键:
            - 'alpha_q': 奇异性指数序列，反映不同q值下的奇异性强度
            - 'f_alpha_spectrum': 多重分形谱数据
            - 'tau_q'或'q_range': 用于获取q值范围

    功能说明:
        1. 在同一坐标系中绘制两条曲线：
           - q-α(q): 反映矩阶数q与奇异性指数α的关系
           - q-f(α): 反映矩阶数q与多重分形谱f(α)的关系
        2. 通过共享Y轴便于比较两条曲线的变化趋势
        3. 添加关键统计信息和图例说明
    """
    # 数据验证：检查必要的数据是否存在
    if ('alpha_q' not in multifractal_result or
            'f_alpha_spectrum' not in multifractal_result):
        ax.text(0.5, 0.5, 'q-α和q-f(α)数据不可用\n请检查输入数据的完整性',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('(d) q-α(q) 和 q-f(α) 关系', fontsize=12, fontweight='bold')
        ax.set_xlabel('矩阶数 q')
        ax.set_ylabel('数值')
        return

    try:
        # 提取alpha_q数据：奇异性指数序列
        # α(q)反映测度分布的奇异性，α值越大表示该区域的测度分布越奇异（集中）
        alpha_q = multifractal_result['alpha_q']
        if not hasattr(alpha_q, '__len__') or len(alpha_q) == 0:
            ax.text(0.5, 0.5, 'alpha_q数据无效或为空',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            return

        # 提取f_alpha_spectrum数据：多重分形谱
        # f(α)表示具有奇异性指数α的子集的分形维数
        spectrum = multifractal_result['f_alpha_spectrum']
        if isinstance(spectrum, list) and len(spectrum) > 0:
            if isinstance(spectrum[0], (list, tuple)):
                # 标准格式：[(alpha1, f1), (alpha2, f2), ...]
                f_alpha_values = [point[1] for point in spectrum]
            else:
                # 备用格式：直接是f(α)值列表
                f_alpha_values = spectrum
        else:
            f_alpha_values = []

        # 获取q值序列：矩阶数范围
        # q>1强调高强度区域，q<1强调低强度区域，q=1为信息维数
        if 'tau_q' in multifractal_result:
            q_values = sorted(multifractal_result['tau_q'].keys())
        elif 'q_range' in multifractal_result:
            q_values = multifractal_result['q_range']
        else:
            # 默认生成与alpha_q长度相同的q序列
            q_values = list(range(len(alpha_q)))

        # 数据长度一致性检查
        min_len = min(len(q_values), len(alpha_q), len(f_alpha_values))
        if min_len < 2:
            ax.text(0.5, 0.5, '有效数据点不足，至少需要2个点',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            return

        # 截取有效数据段
        q_values = q_values[:min_len]
        alpha_q_valid = alpha_q[:min_len]
        f_alpha_valid = f_alpha_values[:min_len]

        # 创建图形和坐标轴
        ax.clear()  # 清空原有内容

        # 绘制q-α(q)曲线：蓝色实线带圆形标记
        # α(q)曲线通常呈现单调递减趋势，反映不同q值下的奇异性强度变化
        line1 = ax.plot(q_values, alpha_q_valid, 'bo-', linewidth=2, markersize=6, label='α(q)', alpha=0.8)

        # 绘制q-f(α)曲线：红色实线带方形标记
        # f(α)曲线呈凸函数形状，最大值对应最主要的分形维数
        line2 = ax.plot(q_values, f_alpha_valid, 'rs-', linewidth=2, markersize=6, label='f(α)', alpha=0.8)

        # 设置图形标题和坐标轴标签
        ax.set_title('(d) q-α(q) 和 q-f(α) 关系（共享Y轴）', fontsize=12, fontweight='bold')
        ax.set_xlabel('矩阶数 q', fontsize=10)
        ax.set_ylabel('数值', fontsize=10)

        # 添加网格线便于读数
        ax.grid(True, alpha=0.3, linestyle='--')

        # 添加图例说明
        ax.legend(loc='best', fontsize=10, framealpha=0.8)

        # 设置刻度线朝内
        ax.tick_params(axis='both', direction='in', which='both')

        # 计算并显示关键统计信息
        alpha_min, alpha_max = min(alpha_q_valid), max(alpha_q_valid)
        f_min, f_max = min(f_alpha_valid), max(f_alpha_valid)
        delta_alpha = alpha_max - alpha_min

        # 信息框内容：显示关键参数范围
        info_text = (f'α范围: [{alpha_min:.3f}, {alpha_max:.3f}]\n'
                     f'f(α)范围: [{f_min:.3f}, {f_max:.3f}]\n'
                     f'谱宽度 Δα: {delta_alpha:.3f}')

        # 在左上角添加统计信息框
        ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # 标记特殊q值点（如果存在）
        special_q_points = [q for q in [0, 1, 2] if q in q_values]
        for q_special in special_q_points:
            idx = q_values.index(q_special)
            ax.plot(q_special, alpha_q_valid[idx], 'go', markersize=8,
                    label=f'q={q_special}' if q_special == special_q_points[0] else "")
            ax.plot(q_special, f_alpha_valid[idx], 'go', markersize=8)

    except Exception as e:
        # 异常处理：显示错误信息
        error_msg = f'绘图过程中发生错误: {str(e)}'
        print(error_msg)
        ax.text(0.5, 0.5, error_msg,
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=10, color='red')

def _plot_statistical_info(ax, multifractal_result):
    """
    在空白子图中显示统计信息

    参数:
        ax: 子图坐标轴
        multifractal_result: 多重分形结果
    """
    ax.set_title('(f) 统计信息摘要', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')  # 隐藏坐标轴

    if multifractal_result is None:
        ax.text(0.5, 0.5, '无有效数据',
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, transform=ax.transAxes)
        return

    # 收集统计信息
    info_lines = ["多重分形分析统计摘要", "=" * 20]

    # 基本q值信息
    if 'tau_q' in multifractal_result and multifractal_result['tau_q']:
        tau_q = multifractal_result['tau_q']
        q_values = sorted(tau_q.keys())
        info_lines.append(f"q值数量: {len(q_values)}")
        info_lines.append(f"q范围: [{min(q_values):.1f}, {max(q_values):.1f}]")

    # 广义分形维数
    if 'D_q' in multifractal_result and multifractal_result['D_q']:
        D_q = multifractal_result['D_q']
        D_values = list(D_q.values())
        if D_values:
            info_lines.append(f"D(q)范围: [{min(D_values):.3f}, {max(D_values):.3f}]")
            if 0 in D_q:
                info_lines.append(f"容量维 D$_{0}$: {D_q[0]:.3f}")
            if 1 in D_q:
                info_lines.append(f"信息维 D$_{1}$: {D_q[1]:.3f}")
            if 2 in D_q:
                info_lines.append(f"关联维 D$_{2}$: {D_q[2]:.3f}")

    # 多重分形谱信息
    if 'f_alpha_spectrum' in multifractal_result and multifractal_result['f_alpha_spectrum']:
        spectrum = multifractal_result['f_alpha_spectrum']
        if isinstance(spectrum, list) and len(spectrum) > 0:
            if isinstance(spectrum[0], (list, tuple)):
                alpha_values = [point[0] for point in spectrum]
                f_values = [point[1] for point in spectrum]
            else:
                alpha_values = multifractal_result.get('alpha_q', [])
                f_values = spectrum

            if alpha_values and f_values:
                delta_alpha = max(alpha_values) - min(alpha_values)
                f_max = max(f_values)
                info_lines.append(f"谱宽度 Δα: {delta_alpha:.3f}")
                info_lines.append(f"最大f(α): {f_max:.3f}")

    # 回归质量信息
    if 'regression_results' in multifractal_result and multifractal_result['regression_results']:
        reg_results = multifractal_result['regression_results']
        r2_values = [info.get('r_squared', 0) for info in reg_results.values()
                     if info.get('r_squared') is not None]
        if r2_values:
            avg_r2 = np.mean(r2_values)
            info_lines.append(f"平均R$^{2}$: {avg_r2:.3f}")

    # 盒子尺寸信息
    if 'box_sizes' in multifractal_result:
        box_sizes = multifractal_result['box_sizes']
        info_lines.append(f"盒子尺寸数: {len(box_sizes)}")
        info_lines.append(f"尺度范围: {min(box_sizes)}-{max(box_sizes)}")

    # 在子图中显示文本
    y_pos = 0.95
    line_height = 0.06

    for i, line in enumerate(info_lines):
        if i == 0:  # 标题
            weight = 'bold'
            size = 11
        elif i == 1:  # 分隔线
            weight = 'normal'
            size = 9
        else:  # 普通信息
            weight = 'normal'
            size = 9

        ax.text(0.05, y_pos, line, transform=ax.transAxes,
                fontsize=size, fontweight=weight, verticalalignment='top')
        y_pos -= line_height

def _plot_partition_function(ax, multifractal_result):
    """
    绘制配分函数与尺度关系图

    对于q≠1: 绘制 log(ε) vs log(Z(q,ε))
    对于q=1: 绘制 log(ε) vs Z(1,ε) (特殊情况)
    """
    # 检查必要的数据是否存在
    if 'Z_q_data' not in multifractal_result or 'box_sizes' not in multifractal_result:
        ax.text(0.5, 0.5, '配分函数数据不可用', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('(a) 配分函数尺度关系', fontsize=12, fontweight='bold')
        # ax.set_xlabel('log(1/ε)')
        ax.set_xlabel('log(ε)')
        ax.set_ylabel('log(Z(q,ε))')
        return

    Z_q_data = multifractal_result['Z_q_data']
    box_sizes = multifractal_result['box_sizes']
    q_range = multifractal_result.get('q_range', list(Z_q_data.keys()))

    if not box_sizes or len(q_range) == 0:
        ax.text(0.5, 0.5, '配分函数数据不完整', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12)
        return

    # 计算 log(ε)
    log_epsilon = np.log(np.array(box_sizes))

    # 为不同的q值选择颜色
    # colors = plt.cm.viridis(np.linspace(0, 1, len(representative_q)))
    # colors = plt.cm.plasma(np.linspace(0, 1, len(representative_q)))
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(representative_q)))
    colors = plt.cm.jet(np.linspace(0, 1, len(q_range)))

    for i, q in enumerate(q_range):
        if q not in Z_q_data:
            continue

        Z_q_values = Z_q_data[q]

        # 确保Z_q_values长度与box_sizes一致
        if len(Z_q_values) != len(box_sizes):
            # 截断或填充以匹配长度
            min_len = min(len(Z_q_values), len(box_sizes))
            Z_q_values = Z_q_values[:min_len]
            log_epsilon_plot = log_epsilon[:min_len]
        else:
            log_epsilon_plot = log_epsilon

        # 数值有效性检查
        Z_q_values = np.array(Z_q_values)  # 确保转换为numpy数组
        valid_mask = np.isfinite(Z_q_values)
        if abs(q - 1) < 1e-10:
            # q=1: 允许负值，只检查有限性
            valid_mask = valid_mask
        else:
            # 其他q: 需要正值
            valid_mask = valid_mask & (Z_q_values > 0)

        if np.sum(valid_mask) < 2:  # 需要至少2个有效点
            continue

        log_epsilon_valid = log_epsilon_plot[valid_mask]
        Z_q_valid = np.array(Z_q_values)[valid_mask]

        # 特殊处理q=1的情况
        if abs(q - 1) < 1e-10:
            # q=1: 直接绘制Z(1,ε)，不取对数
            y_values = Z_q_valid
            plot_label = f"q={q:.1f}"

            # 绘制数据点
            ax.scatter(log_epsilon_valid, y_values, color=colors[i],
                       alpha=0.6, s=50, label=plot_label)

            # 绘制拟合线（如果回归结果可用）
            if 'regression_results' in multifractal_result and q in multifractal_result['regression_results']:
                reg_info = multifractal_result['regression_results'][q]
                y_fit = reg_info['slope'] * log_epsilon_valid + reg_info['intercept']
                ax.plot(log_epsilon_valid, y_fit, color=colors[i],
                        linewidth=2, linestyle='--', alpha=0.8)

            # 设置y轴标签
            ax.set_ylabel('Z(1,ε)')
        else:
            # 一般情况: 绘制log(Z(q,ε))
            y_values = np.log(Z_q_valid)
            plot_label = f"q={q:.1f}"

            # 绘制数据点
            ax.scatter(log_epsilon_valid, y_values, color=colors[i], alpha=0.6, s=10, label=plot_label)

            # 绘制拟合线
            if 'regression_results' in multifractal_result and q in multifractal_result['regression_results']:
                reg_info = multifractal_result['regression_results'][q]
                y_fit = reg_info['slope'] * log_epsilon_valid + reg_info['intercept']
                ax.plot(log_epsilon_valid, y_fit, color=colors[i],
                        linewidth=2, linestyle='--', alpha=0.8)

            # 设置y轴标签
            ax.set_ylabel('log(Z(q,ε))')

    ax.set_title('(a) 配分函数尺度关系', fontsize=12, fontweight='bold')
    ax.set_xlabel('log(ε)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, ncol=5)
    # # 设置legend不可见，这个legend太长了，q值太多了
    legend = ax.legend()
    legend.set_visible(False)
    # 设置刻度线朝内
    ax.tick_params(axis='both', direction='in', which='both')
    # # 更详细的设置
    # ax.tick_params(axis='both', direction='in', which='both',
    #                length=6, width=1, color='black',  # 刻度线长度、宽度和颜色
    #                top=True, bottom=True, left=True, right=True)  # 在哪些边显示刻度线
    # # 只设置x轴刻度线朝内
    # ax.tick_params(axis='x', direction='in')
    # # 只设置y轴刻度线朝内
    # ax.tick_params(axis='y', direction='in')

    # #### 这部分当q太多时，不进行显示，这个是显示q斜率-legend的信息
    # # 添加拟合质量信息
    # if 'regression_results' in multifractal_result:
    #     r2_text = r''
    #     r2_values = []
    #     for key in multifractal_result['regression_results'].keys():
    #         info = multifractal_result['regression_results'][key]
    #         # print(info, key)
    #         if 'r_squared' in info:
    #             r2_values.append(info["r_squared"])
    #             r2_text += f'q:{int(key):>2d}, r2:{info["r_squared"]:.2f}'
    #             if len(r2_values)%3==0:
    #                 r2_text += '\n'
    #             else:
    #                 r2_text += '  '
    #
    #     if r2_values:
    #         avg_r2 = np.mean(r2_values)
    #         # ax.text(0.98, 0.02, r2_text+f'平均R$^{2}$: {avg_r2:.4f}', transform=ax.transAxes, fontsize=9,
    #         #         verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
    #         ax.text(0.8, 0.02, r2_text + f'平均R$^{2}$: {avg_r2:.4f}', transform=ax.transAxes, fontsize=9,
    #                 verticalalignment='bottom', horizontalalignment='center',  # 修改此参数为居中对齐
    #                 bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.3))


def _plot_tau_q(ax, multifractal_result):
    """绘制质量指数τ(q)"""
    if 'tau_q' not in multifractal_result or not multifractal_result['tau_q']:
        ax.text(0.5, 0.5, 'τ(q)数据不可用',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('(b) 质量指数 τ(q)', fontsize=12, fontweight='bold')
        ax.set_xlabel('q')
        ax.set_ylabel('τ(q)')
        return

    tau_q = multifractal_result['tau_q']
    q_values = sorted(tau_q.keys())
    tau_values = [tau_q[q] for q in q_values]

    # 绘制τ(q)曲线
    ax.plot(q_values, tau_values, 'bo-', linewidth=2, markersize=6,
            label='τ(q)', alpha=0.8)

    # # 标记特殊点
    special_q = []
    target_qs = [0, 1, 2]
    for target_q in target_qs:
        # 检查是否存在足够接近的q值（容差1e-6）
        matches = [q for q in q_values if abs(q - target_q) < 1e-6]
        if matches:
            special_q.append(matches[0])


    for q in special_q:
        ax.plot(int(q), tau_q[q], 'ro', markersize=8, label=f'q={int(q):d}')

    ax.set_title('(b) 质量指数 τ(q)', fontsize=12, fontweight='bold')
    ax.set_xlabel('矩阶数 q', fontsize=10)
    ax.set_ylabel('τ(q)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    # 设置刻度线朝内
    ax.tick_params(axis='both', direction='in', which='both')

    # 添加τ(q)特性说明
    if len(tau_values) > 1:
        min_tau = min(tau_values)
        max_tau = max(tau_values)
        tau_range = max_tau - min_tau
        ax.text(0.98, 0.02, f'τ范围: {tau_range:.4f}\n{min_tau:.4f} - {max_tau:.4f}', transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.3))


def _plot_D_q(ax, multifractal_result):
    """绘制广义分形维数D(q)"""
    if 'D_q' not in multifractal_result or not multifractal_result['D_q']:
        ax.text(0.5, 0.5, 'D(q)数据不可用',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('(c) 广义分形维数 D(q)', fontsize=12, fontweight='bold')
        ax.set_xlabel('q')
        ax.set_ylabel('D(q)')
        return

    D_q = multifractal_result['D_q']
    q_values = sorted(D_q.keys())
    D_values = [D_q[q] for q in q_values]

    # 绘制D(q)曲线
    ax.plot(q_values, D_values, 'go-', linewidth=2, markersize=6, label='D(q)', alpha=0.8)

    special_q = []
    target_qs = [0, 1, 2]
    for target_q in target_qs:
        # 检查是否存在足够接近的q值（容差1e-6）
        matches = [q for q in q_values if abs(q - target_q) < 1e-6]
        if matches:
            special_q.append(matches[0])

    color_q = {0:'r', 1:'orange', 2:'yellow'}

    # 标记重要的维数值
    for q in special_q:
        ax.axhline(y=D_q[q], color=color_q[int(q)], linestyle='--', alpha=0.6, label=f'D$_{int(q):d}$={D_q[q]:.4f}')

    ax.set_title('(c) 广义分形维数 D(q)', fontsize=12, fontweight='bold')
    ax.set_xlabel('矩阶数 q', fontsize=10)
    ax.set_ylabel('D(q)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    # 设置刻度线朝内
    ax.tick_params(axis='both', direction='in', which='both')

    # 添加D(q)统计信息
    if len(D_values) > 0:
        D_min, D_max = min(D_values), max(D_values)
        # ax.text(0.02, 0.02, f'D范围: [{D_min:.4f}, {D_max:.4f}]', transform=ax.transAxes, fontsize=9, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax.text(0.02, 0.02, f'D范围: [{D_min:.4f}, {D_max:.4f}]', transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='left', bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.3))


def _plot_f_alpha_spectrum(ax, multifractal_result):
    """绘制多重分形谱f(α)"""
    if ('f_alpha_spectrum' not in multifractal_result or
            not multifractal_result['f_alpha_spectrum']):
        ax.text(0.5, 0.5, 'f(α)谱数据不可用',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('(d) 多重分形谱 f(α)', fontsize=12, fontweight='bold')
        ax.set_xlabel('α')
        ax.set_ylabel('f(α)')
        return

    spectrum = multifractal_result['f_alpha_spectrum']

    # 检查spectrum的数据结构
    if isinstance(spectrum, list) and len(spectrum) > 0:
        # 标准格式: [(alpha1, f1), (alpha2, f2), ...]
        nd_spectrum = np.array(spectrum)
        alpha_values = nd_spectrum[:, 0]
        f_alpha_values = nd_spectrum[:, 1]
        # alpha_values = [point[0] for point in spectrum]
        # f_alpha_values = [point[1] for point in spectrum]
    elif 'alpha_q' in multifractal_result and 'alpha_spectrum' in multifractal_result:
        # 备用格式: 使用alpha_q-alpha_spectrum从multifractal_result中取值
        alpha_values = multifractal_result['alpha_q']
        if isinstance(alpha_values, list):
            alpha_values = np.array(alpha_values).ravel()
        f_alpha_values = multifractal_result['alpha_spectrum']
        if isinstance(f_alpha_values, list):
            f_alpha_values = np.array(f_alpha_values).ravel()
    else:
        ax.text(0.5, 0.5, 'f(α)谱格式不支持,请按照 \{alpha_q:list;alpha_spectrum:list\} 格式\n初始化multifractal_result[f_alpha_spectrum]',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        return

    # 绘制f(α)谱线
    ax.plot(alpha_values, f_alpha_values, 'mo-', linewidth=2, markersize=6, label='f(α)', alpha=0.8)

    # 标记谱宽度和最大值
    if len(alpha_values) > 0:
        alpha_min, alpha_max = min(alpha_values), max(alpha_values)
        f_max = max(f_alpha_values)
        alpha_at_fmax = alpha_values[np.argmax(f_alpha_values)]

        # 标记谱宽度
        ax.axvline(x=alpha_min, color='gray', linestyle=':', alpha=0.5, label=f'α_min={alpha_min:.4f}')
        ax.axvline(x=alpha_max, color='gray', linestyle=':', alpha=0.5, label=f'α_max={alpha_max:.4f}')

        # 标记最大值点
        ax.plot(alpha_at_fmax, f_max, 'r*', markersize=12, label=f'峰值(α={alpha_at_fmax:.4f})')

        # 填充谱线下方的区域
        ax.fill_between(alpha_values, f_alpha_values, alpha=0.2, color='purple')

    ax.set_title('(d) 多重分形谱 f(α)', fontsize=12, fontweight='bold')
    ax.set_xlabel('奇异性指数 α', fontsize=10)
    ax.set_ylabel('f(α)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=8)
    # 设置刻度线朝内
    ax.tick_params(axis='both', direction='in', which='both')

    # 添加谱特征信息
    if len(alpha_values) > 1:
        delta_alpha = alpha_max - alpha_min
        # asymmetry = (alpha_max - alpha_at_fmax) / (alpha_at_fmax - alpha_min) if alpha_at_fmax > alpha_min else 1

        info_text = f'Δα = {delta_alpha:.4f}\nα$_{0}$ = {alpha_at_fmax:.4f}\nf_max = {f_max:.4f}'
        ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.3))


def _print_statistical_info(multifractal_result):
    """打印多重分形分析的统计信息"""
    print("\n" + "=" * 60)
    print("多重分形分析统计信息")
    print("=" * 60)

    if 'tau_q' in multifractal_result and multifractal_result['tau_q']:
        tau_q = multifractal_result['tau_q']
        q_values = sorted(tau_q.keys())
        tau_values = [tau_q[q] for q in q_values]
        print(f"有效q值数量: {len(q_values)}")
        print(f"q范围: [{min(q_values)}, {max(q_values)}]")
        print(f"τ_q范围: [{min(tau_values)}, {max(tau_values)}]")
        print('τ_q:', type(tau_q), tau_q)
        print("=" * 60)

    if 'D_q' in multifractal_result and multifractal_result['D_q']:
        D_q = multifractal_result['D_q']
        D_values = list(D_q.values())
        print(f"广义分形维数范围: [{min(D_values):.4f}, {max(D_values):.4f}]")
        if 0 in D_q:
            print(f"容量维数 D₀ = {D_q[0]:.4f}")
        if 1 in D_q:
            print(f"信息维数 D₁ = {D_q[1]:.4f}")
        if 2 in D_q:
            print(f"关联维数 D₂ = {D_q[2]:.4f}")
        print('D_q:', type(D_q), D_q)
        print("=" * 60)

    if 'f_alpha_spectrum' in multifractal_result and multifractal_result['f_alpha_spectrum']:
        spectrum = multifractal_result['f_alpha_spectrum']

        # 提取alpha值
        if isinstance(spectrum, list) and len(spectrum) > 0 and isinstance(spectrum[0], (list, tuple)):
            alpha_values = [point[0] for point in spectrum]
            f_values = [point[1] for point in spectrum]
        elif 'alpha_q' in multifractal_result:
            alpha_values = multifractal_result['alpha_q']
            if hasattr(spectrum, '__len__') and not isinstance(spectrum[0], (list, tuple)):
                f_values = spectrum
            else:
                f_values = [1.0] * len(alpha_values)
        else:
            alpha_values = []
            f_values = []

        if alpha_values and f_values:
            delta_alpha = max(alpha_values) - min(alpha_values)
            alpha_max_f = alpha_values[np.argmax(f_values)] if alpha_values else 0
            print(f"奇异性谱宽度 Δα = {delta_alpha:.4f}")
            print(f"谱峰值位置 α₀ = {alpha_max_f:.4f}")
            print(f"最大分形维数 f_max = {max(f_values):.4f}")
            print('spectrum:', type(spectrum), spectrum)
        print("=" * 60)

    if 'regression_results' in multifractal_result and multifractal_result['regression_results']:
        regression_results = multifractal_result['regression_results']
        # 使用get方法安全访问，如果键不存在返回None
        r2_values = [info.get('r_squared') for info in regression_results.values()]
        # 过滤掉None值
        r2_values = [r2 for r2 in r2_values if r2 is not None]

        avg_r2 = np.mean(r2_values) if r2_values else 0
        print(f"平均回归确定系数 R² = {avg_r2:.4f}")
        print('regression_results:', type(regression_results), regression_results)
        print("=" * 60)

    if 'box_sizes' in multifractal_result:
        print(f"使用的盒子尺寸: {multifractal_result['box_sizes']}")
        print("=" * 60)

    if 'Z_q_data' in multifractal_result and multifractal_result['Z_q_data']:
        print('Z_q_data or partition_sum is:', type(multifractal_result['Z_q_data']))
        Z_q_data = multifractal_result['Z_q_data']
        q_list = Z_q_data.keys()
        for q in q_list:
            print(q, ':',Z_q_data[q])

# 使用示例
def demo_multifractal_analysis():
    """
    演示多重分形分析的使用方法
    """
    # 生成一个测试图像（可以用您的电成像数据替换）
    data_img_dyna, data_img_stat, data_depth = get_random_ele_data()
    print(data_img_dyna.shape, data_img_stat.shape, data_depth.shape)       # (500, 250) (500, 250) (500, 1)

    # data_img_stat = cv2.imread(r'C:\Users\Maple\Documents\MATLAB\multifractal-last modified\output2.jpg', cv2.IMREAD_GRAYSCALE)
    test_image = data_img_stat.astype(np.uint8)

    print("\n=== 多重分形分析 ===")
    fd, processed_image, multifractal_result = cal_pic_fractal_dimension_extended(
        test_image,
        image_shape=[256, 256],
        method='multifractal',
        processing_method='original',
        # processing_method='adaptive_binary',
        multifractal_params={
            # 'q_range': np.arange(-5, 5.5, 0.5),
            'q_range': np.arange(-5, 5.5, 0.5),
            # 'q_range': np.arange(-5, 6, 1),
            # 'q_range': np.arange(-12, 13, 2),
            'box_sizes': [2, 4, 8, 16, 32, 64],
            # 'box_sizes': list(np.arange(2, 32, 2)),
        },
    )

    plot_multifractal_spectrum(
        multifractal_result,
    )

if __name__ == "__main__":
    demo_multifractal_analysis()