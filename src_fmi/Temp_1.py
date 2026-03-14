import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import sys
from typing import Dict, Tuple


def multifractal_analysis(image_array: np.ndarray, q_range: float = 5.0, q_step: float = 0.1) -> Dict:
    """
    多维分形分析主函数

    Args:
        image_array: 输入图像数组(灰度图)
        q_range: q值范围，从-q_range到+q_range
        q_step: q值步长

    Returns:
        包含所有分形计算结果的字典
    """
    # 参数设置
    mpixel = 2  # 最小像素块大小

    # 调整图像尺寸为2的幂次方
    data1 = cv2.resize(image_array, (256, 256), interpolation=cv2.INTER_CUBIC)
    rows, cols = data1.shape

    # 计算最大尺度级数
    p = int(math.floor(math.log(rows) / math.log(mpixel)))
    pixn = mpixel ** p
    data = cv2.resize(data1, (pixn, pixn), interpolation=cv2.INTER_CUBIC)

    # 尺度数组
    r = [mpixel ** (i + 1) for i in range(p)]
    max_boxes = (pixn ** 2) // (mpixel ** 2)

    # 初始化变量
    P = np.zeros((int(max_boxes), p))
    boxes = np.zeros(p)
    object_sum = np.sum(data)

    # 多尺度盒子划分与概率计算
    for j in range(p):
        boxes[j] = (pixn ** 2) / (r[j] ** 2)
        count = 0
        step = int(r[j])

        for m in range(0, pixn - step + 1, step):
            for n in range(0, pixn - step + 1, step):
                if count < max_boxes:
                    block = data[m:m + step, n:n + step]
                    P[count, j] = np.sum(block)
                    count += 1

    # 概率归一化处理
    npL = np.zeros((int(max_boxes), p))
    for l in range(p):
        nboxes = int((pixn ** 2) / (r[l] ** 2))
        norm = np.sum(P[:nboxes, l])

        if abs(norm - object_sum) > 1e-10:
            print(f'警告: 归一化因子差异较大, norm: {norm}, object_sum: {object_sum}')

        for i in range(nboxes):
            npL[i, l] = P[i, l] / norm if norm != 0 else 0

    # 生成q值序列
    q_list = np.arange(-q_range, q_range + q_step, q_step)
    qval = q_list.copy()

    # 初始化结果矩阵
    fql = np.zeros((p, len(q_list)))
    aql = np.zeros((p, len(q_list)))
    qql = np.zeros((p, len(q_list)))

    # 配分函数计算
    for l in range(p):
        nboxes = int((pixn ** 2) / (r[l] ** 2))

        for count, q in enumerate(q_list):
            qsum = np.sum([npL[i, l] ** q for i in range(nboxes) if npL[i, l] != 0])

            fqnum, aqnum, smuiqL = 0.0, 0.0, 0.0
            for i in range(nboxes):
                if npL[i, l] != 0:
                    muiqL = (npL[i, l] ** q) / qsum
                    fqnum += muiqL * math.log(muiqL) if muiqL > 0 else 0
                    aqnum += muiqL * math.log(npL[i, l]) if npL[i, l] > 0 else 0
                    smuiqL += muiqL

            fql[l, count] = fqnum
            aql[l, count] = aqnum
            qql[l, count] = qsum

    # q=1特殊情况处理
    psum = np.zeros(p)
    for l in range(p):
        nboxes = int((pixn ** 2) / (r[l] ** 2))
        for i in range(nboxes):
            if npL[i, l] != 0:
                psum[l] += npL[i, l] * math.log(npL[i, l])

    # 尺度对数计算
    logl = np.array([math.log(r_val) for r_val in r])

    # 广义分形维数Dq计算
    dq = np.zeros(len(qval))
    for i, q in enumerate(qval):
        if abs(q - 1) > 1e-10:
            line = np.polyfit(logl, np.log(qql[:, i] + 1e-10), 1)
            dq[i] = line[0] / (q - 1)
        else:
            line = np.polyfit(logl, psum, 1)
            dq[i] = line[0]

    # 奇异指数α(q)和奇异谱f(α)计算
    aq, fq = np.zeros(len(qval)), np.zeros(len(qval))
    ar2, fr2 = np.zeros(len(qval)), np.zeros(len(qval))

    for i in range(len(qval)):
        # α(q)计算
        line_a = np.polyfit(logl, aql[:, i], 1)
        aq[i] = line_a[0]
        yfit_a = np.polyval(line_a, logl)
        sse_a = np.sum((aql[:, i] - yfit_a) ** 2)
        sst_a = np.sum((aql[:, i] - np.mean(aql[:, i])) ** 2)
        ar2[i] = 1 - (sse_a / sst_a) if sst_a != 0 else 0

        # f(α)计算
        line_f = np.polyfit(logl, fql[:, i], 1)
        fq[i] = line_f[0]
        yfit_f = np.polyval(line_f, logl)
        sse_f = np.sum((fql[:, i] - yfit_f) ** 2)
        sst_f = np.sum((fql[:, i] - np.mean(fql[:, i])) ** 2)
        fr2[i] = 1 - (sse_f / sst_f) if sst_f != 0 else 0

    # 质量指数τ(q)计算
    tau_q = -(qval - 1) * dq

    # 构建结果字典
    results = {
        'q_values': qval,           # ndarray -5.0->5.0 step=0.1
        'alpha_q': aq,              # alpha(q) ndarray shape=q_values
        'f_alpha': fq,              #
        'D_q': dq,
        'tau_q': tau_q,
        'alpha_r2': ar2,
        'f_r2': fr2,
        'scale_levels': p,
        'image_size': pixn,
        'r_values': r
    }

    return results


def plot_multifractal_results(results: Dict):
    """
    绘制2×2子图展示多维分形分析结果
    """
    qval = results['q_values']
    aq = results['alpha_q']
    fq = results['f_alpha']
    dq = results['D_q']
    tau_q = results['tau_q']

    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 子图1: q-alpha(q)+f(q)
    axes[0, 0].plot(qval, aq, 'r-o', markersize=3, label='α(q)')
    axes[0, 0].plot(qval, fq, 'g-s', markersize=3, label='f(q)')
    axes[0, 0].set_xlabel('q')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('α(q) and f(q) vs q')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 子图2: alpha(q)-f(q)及其拟合结果
    axes[0, 1].plot(aq, fq, 'bo', markersize=4, label='f(α) vs α')
    if len(aq) > 2:
        try:
            # 抛物线拟合
            parabola_fit = np.polyfit(aq, fq, 2)
            aq_fit = np.linspace(min(aq), max(aq), 100)
            fq_fit = np.polyval(parabola_fit, aq_fit)
            axes[0, 1].plot(aq_fit, fq_fit, 'r-', linewidth=2, label='Parabolic Fit')
        except:
            pass
    axes[0, 1].set_xlabel('α(q)')
    axes[0, 1].set_ylabel('f(α)')
    axes[0, 1].set_title('Multifractal Spectrum f(α) vs α')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 子图3: q-Dq
    axes[1, 0].plot(qval, dq, 'm-^', markersize=4, linewidth=2)
    axes[1, 0].set_xlabel('q')
    axes[1, 0].set_ylabel('Dq')
    axes[1, 0].set_title('Generalized Fractal Dimension Dq vs q')
    axes[1, 0].grid(True)

    # 子图4: q-τq
    axes[1, 1].plot(qval, tau_q, 'c-d', markersize=4, linewidth=2)
    axes[1, 1].set_xlabel('q')
    axes[1, 1].set_ylabel('τ(q)')
    axes[1, 1].set_title('Mass Exponent τ(q) vs q')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


# 测试用例
if __name__ == "__main__":
    # 生成测试图像（替代实际图像加载）
    # test_image = cv2.imread(r"F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\LG7-12_423_5191.7520_5192.3770_stat.png", cv2.IMREAD_GRAYSCALE)
    test_image = cv2.imread(r"C:\Users\Maple\Documents\MATLAB\multifractal-last modified\output1.jpg", cv2.IMREAD_GRAYSCALE)
    test_image = test_image.astype(np.uint8)

    print("开始多维分形分析...")
    results = multifractal_analysis(test_image)

    print(f"分析完成！尺度级数: {results['scale_levels']}")
    print(f"q值范围: {results['q_values'][0]:.1f} 到 {results['q_values'][-1]:.1f}")

    # 关键分形维数值
    idx0 = np.argmin(np.abs(results['q_values']))
    idx1 = np.argmin(np.abs(results['q_values'] - 1))
    idx2 = np.argmin(np.abs(results['q_values'] - 2))

    print(f"容量维数 D0 = {results['D_q'][idx0]:.4f}")
    print(f"信息维数 D1 = {results['D_q'][idx1]:.4f}")
    print(f"关联维数 D2 = {results['D_q'][idx2]:.4f}")
    print(f"谱宽度 Δα = {np.max(results['alpha_q']) - np.min(results['alpha_q']):.4f}")

    # 绘制结果
    plot_multifractal_results(results)
