import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import pandas as pd

# =========================================================================
# 第一部分：数据加载与预处理
# =========================================================================
# 清空命令窗口（Python中没有直接等价，但可以打印分隔线）
print("=" * 50)
print("开始多重分形分析")
print("=" * 50)


# # 加载目标图像（当前使用的图像）
data1 = cv2.imread(r"C:\Users\Maple\Documents\MATLAB\multifractal-last modified\output1.jpg", cv2.IMREAD_GRAYSCALE)# 读取图像并转换为numpy数组
# # data1 = cv2.resize(data1, (256, 256), interpolation=cv2.INTER_CUBIC)
# data1 = cv2.resize(data1, (256, 256), interpolation=cv2.INTER_LINEAR)
# data1 = pd.read_csv(r"C:\Users\Maple\Desktop\Temp.csv")
# data1 = data1.values

# 获取图像尺寸
cols, rows = data1.shape  # 注意：numpy的shape返回(行, 列)，与Matlab的(列, 行)相反

# 设置最小像素块大小（必须为2，因为使用二分法进行多尺度分析）
mpixel = 2

# 计算最大尺度级数p：基于图像尺寸和最小像素块大小的对数计算
# p = fix(log(图像行数)/log(2))，确保尺度划分合理
p = int(math.floor(math.log(rows) / math.log(mpixel)))

# 计算标准化后的图像尺寸（2的p次方，便于多尺度分析）
pixn = mpixel ** p
width = pixn  # 图像处理宽度

# 最大尺度值（用于后续计算）
maxx = mpixel ** p

# =========================================================================
# 第二部分：初始化变量和矩阵
# =========================================================================
# 定义尺度数组r：存储每个尺度级别对应的盒子大小
r = np.zeros(p)
for i in range(p):
    r[i] = mpixel ** (i+1)

# 计算最大盒子数量（最小尺度时的盒子数）
max_boxes = (maxx ** 2) // (mpixel ** 2)

# 初始化概率矩阵P：存储每个尺度下每个盒子中的像素和
# 维度：最大盒子数 × 尺度级数p
P = np.zeros((int(max_boxes), p))

# 初始化盒子数量数组：存储每个尺度级别对应的盒子总数
boxes = np.zeros(p)

# 重新定义数据矩阵：将原始图像数据裁剪或扩展到标准尺寸
data = np.zeros((pixn, pixn))
for i in range(pixn):
    for j in range(pixn):
        if i < data1.shape[0] and j < data1.shape[1]:
            data[i, j] = data1[i, j]  # 复制像素值
        else:
            data[i, j] = 0  # 超出原图范围的部分补零

# 计算图像中目标像素的总和（用于后续归一化验证）
object_sum = np.sum(data)

# =========================================================================
# 第三部分：多尺度盒子划分与概率计算
# =========================================================================

# 计算每个尺度级别对应的盒子总数
for j in range(p):
    boxes[j] = (maxx ** 2) / (r[j] ** 2)

# 多尺度盒子统计：在不同尺度下划分图像并计算每个盒子中的像素和
for j in range(p):
    count = 0  # 盒子计数器重置

    # 遍历所有可能的盒子起始位置（步长为当前尺度对应的盒子大小）
    # 注意：Python的range是左闭右开，所以需要+1
    step = int(mpixel ** (j + 1))  # j从0开始，所以需要j+1
    for m in range(0, width - step + 1, step):
        for n in range(0, width - step + 1, step):
            count += 1  # 盒子计数增加

            # 计算当前盒子(m:n区域)内所有像素值的总和
            # 注意：Python索引是左闭右开，所以需要m:m+step而不是m:m+step-1
            block = data[m:m + step, n:n + step]
            sums = np.sum(block)

            # 将像素和存入概率矩阵P
            if count <= max_boxes:
                P[count - 1, j] = sums  # Python索引从0开始

# =========================================================================
# 第四部分：概率归一化处理
# =========================================================================

# 初始化归一化概率矩阵
npL = np.zeros((int(max_boxes), p))

# 对每个尺度级别进行归一化
for l in range(p):  # l从0到p-1
    # 计算当前尺度下的盒子总数
    nboxes = (maxx ** 2) // (mpixel ** (2 * (l + 1)))  # l从0开始，所以需要l+1

    # 计算当前尺度下所有盒子的像素和总和（归一化分母）
    norm = np.sum(P[0:int(nboxes), l])

    # 验证归一化因子的正确性（应该等于总目标像素数）
    if abs(norm - object_sum) > 1e-10:  # 使用容差比较浮点数
        print('error: 归一化因子计算错误')
        print(f'norm: {norm}, object_sum: {object_sum}')

    # 对每个盒子进行概率归一化：盒子像素和 / 总像素和
    for i in range(int(nboxes)):
        if norm != 0:
            npL[i, l] = P[i, l] / norm
        else:
            npL[i, l] = 0

# =========================================================================
# 第五部分：多重分形分析 - 配分函数计算
# =========================================================================

# 设置q值范围参数
qran = 5  # q值的范围：从 -qran 到 +qran
q_step = 0.1  # q值的采样步长

# 生成q值序列
q_list_values = np.arange(-qran, qran + q_step, q_step)

# 初始化结果存储矩阵
fql = np.zeros((p, len(q_list_values)))  # 存储f(q,l)值
aql = np.zeros((p, len(q_list_values)))  # 存储a(q,l)值
qql = np.zeros((p, len(q_list_values)))  # 存储配分函数值
qval = np.zeros(len(q_list_values))  # 存储q值序列

# 对每个尺度级别l进行计算
for l in range(p):
    count = -1  # q值索引计数器（从-1开始，因为后面会先+1）

    # 当前尺度下的盒子总数
    nboxes = int((width ** 2) / (mpixel ** 2))

    # 遍历q值范围（从-qran到+qran，步长为depth）
    for q in np.arange(-qran, qran + q_step, q_step):
        qsum = 0.0  # 配分函数值初始化
        newqsum = 0.0  # 备用变量（当前未使用）

        count += 1  # 更新计数器

        # 计算配分函数：sum(p_i^q)
        for i in range(nboxes):
            if (npL[i, l] != 0):  # 避免对0进行幂运算
                qsum += npL[i, l] ** q

        # 初始化累积变量
        fqnum = 0.0  # 用于计算f(q)
        aqnum = 0.0  # 用于计算a(q)
        smuiqL = 0.0  # 用于验证概率归一化

        # 计算加权概率和相关信息
        for i in range(nboxes):
            if (npL[i, l] != 0):
                # 计算加权概率：μ_i(q) = p_i^q / sum(p_j^q)
                muiqL = (npL[i, l] ** q) / qsum

                # 累加f(q)计算所需项：μ_i(q) * log(μ_i(q))
                fqnum += (muiqL * math.log(muiqL)) if muiqL > 0 else 0

                # 累加a(q)计算所需项：μ_i(q) * log(p_i)
                aqnum += (muiqL * math.log(npL[i, l])) if npL[i, l] > 0 else 0

                # 累加加权概率和（应该等于1）
                smuiqL += muiqL

        # 验证加权概率和是否为1（允许浮点数误差）
        if abs(smuiqL - 1) > 1e-10:
            print('error: 加权概率和不为1')
            print(f'smuiqL: {smuiqL}')

        # 存储结果
        fql[l, count] = fqnum  # 存储f(q,l)
        aql[l, count] = aqnum  # 存储a(q,l)
        qval[count] = q  # 存储当前q值
        qql[l, count] = qsum  # 存储配分函数值

# =========================================================================
# 第六部分：特殊处理q=1的情况（避免除零错误）
# =========================================================================

# 初始化q=1时的特殊计算数组
psum = np.zeros(p)

# 对每个尺度级别计算q=1时的特殊项
for l in range(p):
    nboxes = int((width ** 2) / (mpixel ** 2))
    for i in range(nboxes):
        for j in range(len(qval)):
            if abs(qval[j] - 1) < q_step / 2:  # 浮点数比较，使用容差
                if (npL[i, l] != 0):
                    # 计算p_i * log(p_i)，用于q=1时的维数计算
                    psum[l] += npL[i, l] * math.log(npL[i, l])

# =========================================================================
# 第七部分：尺度对数计算
# =========================================================================

# 计算每个尺度级别的对数尺度值log(ε)
logl = np.zeros(p)
for l in range(p):
    logl[l] = math.log(mpixel ** (l + 1))  # log(盒子大小)，l从0开始所以用l+1

# =========================================================================
# 第八部分：广义分形维数Dq计算
# =========================================================================

# 初始化广义分形维数数组
dq = np.zeros(len(qval))

# 对每个q值计算对应的广义分形维数 Dq
for i in range(len(qval)):
    if abs(qval[i] - 1) > 1e-10:  # q ≠ 1
        # 当q≠1时：Dq = 斜率 / (q-1)
        # 通过对数线性回归求斜率：log(χ(q,ε)) ~ log(ε)
        # 使用numpy的polyfit进行线性拟合
        line = np.polyfit(logl, np.log(qql[:, i] + 1e-10), 1)  # 添加小常数避免log(0)
        dq[i] = line[0] / (qval[i] - 1)  # 计算广义分形维数
    else:
        # 当q=1时特殊处理：D1 = 斜率
        # 使用之前计算的psum进行线性回归
        line = np.polyfit(logl, psum, 1)  # 一阶多项式拟合
        dq[i] = line[0]  # 斜率即为分形维数

# =========================================================================
# 第九部分：奇异指数α(q)和奇异谱f(α)计算
# =========================================================================

# 初始化结果数组
aq = np.zeros(len(qval))  # 奇异指数α(q)
fq = np.zeros(len(qval))  # 奇异谱f(α)
ar2 = np.zeros(len(qval))  # α(q)拟合的R²值
fr2 = np.zeros(len(qval))  # f(α)拟合的R²值

# 计算奇异指数α(q) = dτ(q)/dq
for i in range(len(qval)):
    # 对每个q值，用logl和aql进行线性回归求α(q)
    line = np.polyfit(logl, aql[:, i], 1)  # 一阶多项式拟合
    aq[i] = line[0]  # 斜率即为α(q)

    # 计算拟合优度R²
    yfit = np.polyval(line, logl)  # 拟合值
    sse = np.sum((aql[:, i] - yfit) ** 2)  # 残差平方和
    sst = np.sum((aql[:, i] - np.mean(aql[:, i])) ** 2)  # 总平方和
    if sst != 0:
        ar2[i] = 1 - (sse / sst)  # 确定系数R²
    else:
        ar2[i] = 0

# 计算奇异谱f(α) = q*α(q) - τ(q)
for i in range(len(qval)):
    # 对每个q值，用logl和fql进行线性回归求f(α)
    line = np.polyfit(logl, fql[:, i], 1)  # 一阶多项式拟合
    fq[i] = line[0]  # 斜率即为f(α)

    # 计算拟合优度R²
    yfit = np.polyval(line, logl)  # 拟合值
    sse = np.sum((fql[:, i] - yfit) ** 2)  # 残差平方和
    sst = np.sum((fql[:, i] - np.mean(fql[:, i])) ** 2)  # 总平方和
    if sst != 0:
        fr2[i] = 1 - (sse / sst)  # 确定系数R²
    else:
        fr2[i] = 0

# =========================================================================
# 第十部分：结果可视化
# =========================================================================

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 图1：α(q)和f(q)随q值的变化
plt.figure(figsize=(10, 6))
plt.plot(qval, aq, 'r:o', label='alpha(q)')
plt.plot(qval, fq, 'g:o', label='f(q)')
plt.legend(loc='best')
plt.xlabel('q', fontsize=14)
plt.ylabel('Value', fontsize=14)
plt.title('α(q) and f(q) vs q', fontsize=14)
plt.grid(True)
plt.show()

# 图2：多重分形奇异谱f(α) vs α
plt.figure(figsize=(10, 6))
plt.plot(aq, fq, 'r:o')
plt.xlabel('alpha(q)', fontsize=14)
plt.ylabel('f(q)', fontsize=14)
plt.title('Multifractal Spectrum f(α) vs α', fontsize=14)
plt.grid(True)
plt.show()

# 图3：奇异谱的抛物线拟合
line = np.polyfit(aq, fq, 2)  # 二阶多项式（抛物线）拟合
pfit = np.polyval(line, aq)  # 拟合值计算
plt.figure(figsize=(10, 6))
plt.plot(aq, fq, 'ro', label='f(α)')
plt.plot(aq, pfit, 'g-', linewidth=2, label='Parabolic fit to f(α)')
# supported values are 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
plt.legend(loc='best')
plt.xlabel('alpha(q)', fontsize=14)
plt.ylabel('f(q)', fontsize=14)
plt.title('Multifractal Spectrum with Parabolic Fit', fontsize=14)
plt.grid(True)
plt.show()

# =========================================================================
# 第十一部分：验证和辅助图形
# =========================================================================

# 重新生成完整的q值序列（用于验证）
qqq = np.zeros(len(qval))
for i in range(len(qval)):
    qqq[i] = -qran + q_step * i  # 从-qran开始，步长depth

# 验证广义分形维数计算结果
ddd = np.zeros(len(qval))
for i in range(len(qval)):
    if abs(qqq[i] - 1) > 1e-10:  # 避免除零
        # 使用公式验证：Dq = (q*α(q) - f(α)) / (q-1)
        ddd[i] = (qqq[i] * aq[i] - fq[i]) / (qqq[i] - 1)
    else:
        ddd[i] = dq[i]  # q=1时使用之前计算的值

# 图4：广义分形维数Dq vs q
plt.figure(figsize=(10, 6))
plt.plot(qqq, dq, 'b-', linewidth=2)
plt.xlabel('q', fontsize=12)
plt.ylabel('Dq', fontsize=12)
plt.title('Generalized Fractal Dimension Dq vs q', fontsize=14)
plt.grid(True)
plt.show()

# 计算质量指数τ(q) = (q-1)*Dq
t = np.zeros(len(qval))
for i in range(len(qval)):
    qqq[i] = -qran + q_step * i  # 确保q值序列正确
    t[i] = (1 - qqq[i]) * dq[i]  # τ(q) = (q-1)*Dq

# 图5：质量指数τ(q) vs q
plt.figure(figsize=(10, 6))
plt.plot(qqq, t, 'm-', linewidth=2)
plt.xlabel('q', fontsize=12)
plt.ylabel('τ(q)', fontsize=12)
plt.title('Mass Exponent τ(q) vs q', fontsize=14)
plt.grid(True)
plt.show()

# =========================================================================
# 程序结束
# =========================================================================

# 显示计算完成信息
print('多重分形分析完成！')
print(f'尺度级数: {p}')
print(f'q值范围: {-qran:.1f} 到 {qran:.1f}，步长: {q_step:.1f}')
print(f'总计算点数: {len(qval)}')

# 输出关键分形维数值
print('\n关键分形维数:')
# 查找最接近0,1,2的q值索引
idx0 = np.argmin(np.abs(qqq))
idx1 = np.argmin(np.abs(qqq - 1))
idx2 = np.argmin(np.abs(qqq - 2))

print(f'容量维数 D0 = {dq[idx0]:.4f}')
print(f'信息维数 D1 = {dq[idx1]:.4f}')
print(f'关联维数 D2 = {dq[idx2]:.4f}')

# 显示多重分形谱特征
print('\n多重分形谱特征:')
print(f'谱宽度 Δα = {np.max(aq) - np.min(aq):.4f}')
print(f'最大f(α) = {np.max(fq):.4f}')

print("=" * 50)
print("程序执行完毕")
print("=" * 50)