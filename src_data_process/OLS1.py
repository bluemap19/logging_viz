import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import least_squares


# 最小二乘法（ Ordinary Least Square，OLS）
def nonlinear_fitting(df, formula_func, initial_guess=(0, 0, 0), bounds=([-np.inf, -np.inf], [np.inf, np.inf])):
    """
    使用最小二乘法拟合非线性方程 R1/R2 = (TEMP1-X1)/(TEMP2-X2)

    参数:
    df : DataFrame - 包含列 ['DEPTH', 'R1', 'R2', 'TEMP1', 'TEMP2']
    formula_func : 函数 - 残差计算函数（需与目标公式匹配）
    initial_guess : 元组 - (X1, X2) 的初始猜测值，默认(0,0)

    返回:
    result : OptimizeResult - 包含拟合参数和状态信息
    """

    # 定义最小二乘残差函数
    def residuals(params):
        return formula_func(df, *params)


    # bounds, 2元组的序列, 设置参数的上下界
    # method(可选), 字符串, 选择优化算法
        # 'trf': Trust Region Reflective (默认)、支持边界约束、适用于有界问题、性能稳健
        # 'lm': Levenberg-Marquardt 不支持边界、适用于小规模无约束问题、计算速度快
        # 'dogbox': dogleg算法 支持边界约束、适用于中等规模问题
    # ftol(可选), 浮点数, 残差相对变化的终止阈值, 默认: 1e-8,
        # 终止条件:(cost(prev) - cost(curr)) / cost(curr) < ftol
        # 意义: 当成本函数（残差平方和）的相对变化小于此值时停止
    # xtol(可选), 浮点数, 参数相对变化的终止阈值, 默认: 1e-8
        # 终止条件: norm(delta_params) / norm(params) < xtol
        # 意义: 当参数的相对变化小于此值时停止
    # gtol(可选), 浮点数, 梯度范数的终止阈值, 默认: 1e-8
        # 终止条件: norm(gradient) < gtol
        # 意义: 当梯度范数小于此值时停止
    # x_scale(可选), 浮点数或类似数组, 作用: 参数缩放因子
        # 格式:标量：所有参数使用相同缩放; 数组：每个参数单独缩放; 'jac': 自动缩放（基于雅可比矩阵）
        # 目的:平衡不同量级参数的影响，提高数值稳定性
        # loss(可选) ,类型: 可调用的或字符串, 作用: 损失函数类型
        # 选项:'linear': 标准最小二乘, 'soft_l1': 平滑L1损失（对异常值更鲁棒）,
        # 'huber': Huber损失（鲁棒回归）, 'cauchy': Cauchy损失（高度鲁棒）, 'arctan': Arctan损失（最鲁棒）

    for method in ['trf', 'dogbox', 'lm', ]:
        try:
            if method == 'lm':
                # 执行非线性最小二乘拟合
                result = least_squares(
                    fun=residuals,
                    x0=initial_guess,
                    method=method,
                    verbose=0  # 查看优化过程
                )

            else:
                # 执行非线性最小二乘拟合
                result = least_squares(
                    fun=residuals,
                    x0=initial_guess,
                    method=method,
                    bounds=bounds,
                    verbose=0  # 查看优化过程
                )

            if result.success:
                return result

        except Exception as e:
            print(f"Method {method} failed: {str(e)}")

    # 所有方法都失败时的处理
    raise RuntimeError("所有优化方法均失败")


# # 定义与目标公式匹配的残差计算函数
# def target_formula(df, XTm, XTr, XR):
#     """计算目标公式的预测值: (TEMP1 - X1)/(TEMP2 - X2)"""
#     return (df['TEMP_measured'] - XTm) / (df['TEMP_real'] - XTr) - np.log2((df['R_measured'] - XR)) / np.log2(df['R_real'])

def target_formula(df, A, B, C):
    return C*(df['R_measured']-A*np.power(df['TEMP_measured'], B))-df['R_real']


# 生成示例数据
def generate_sample_data(size=100, true_params=(25.0, 15.0), noise_level=0.1):
    """生成模拟测试数据"""
    np.random.seed(42)
    depth = np.linspace(1000, 5000, size)
    temp1 = 80 + 0.02 * depth + np.random.normal(0, 5, size)
    temp2 = 70 + 0.015 * depth + np.random.normal(0, 5, size)

    # 使用真实参数生成电阻率
    X1_true, X2_true = true_params
    r_ratio = (temp1 - X1_true) / (temp2 - X2_true)

    # 添加噪声
    r1 = r_ratio * (1 + np.random.normal(0, noise_level, size))
    r2 = np.ones(size) * (1 + np.random.normal(0, noise_level, size))

    return pd.DataFrame({
        'DEPTH': depth,
        'R1': r1,
        'R2': r2,
        'TEMP1': temp1,
        'TEMP2': temp2
    })


# 测试示例
if __name__ == "__main__":
    # 生成模拟数据（真实参数 X1=25.0, X2=15.0）
    sample_df = generate_sample_data(size=500, true_params=(25.0, 15.0))
    print(sample_df.describe())

    # 执行拟合
    fit_result = nonlinear_fitting(sample_df, target_formula, initial_guess=(20, 10, 1000))

    # 打印结果
    print("\n拟合结果:")
    print(f"X1 = {fit_result.x[0]:.4f} (真值: 25.0)")
    print(f"X2 = {fit_result.x[1]:.4f} (真值: 15.0)")
    print(f"X3 = {fit_result.x[2]:.4f} (真值: 15.0)")
    print(f"残差平方和: {fit_result.cost:.6f}")
    print(f"状态: {fit_result.message}")

    df_result = target_formula(sample_df, X1=fit_result.x[0], X2=fit_result.x[1], X3=fit_result.x[1])
    print(df_result.describe())