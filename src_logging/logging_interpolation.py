import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, CubicSpline, PchipInterpolator
from sklearn.preprocessing import StandardScaler
import warnings

from src_plot.TEMP_4 import WellLogVisualizer


class ConventionalLogInterpolator:
    """
    常规测井曲线插值器

    功能: 对常规测井曲线（GR、RT、NPHI、DEN等）进行高分辨率插值
          提高数据密度和分辨率，保持地质特征

    主要特性:
    - 支持线性、三次样条、PCHIP等多种插值方法
    - 自动处理缺失值和异常值
    - 保持原始深度范围和地质趋势
    - 提供插值质量评估和可视化
    - 专门优化常规测井曲线特性
    """

    def __init__(self, method='pchip', fill_value='extrapolate'):
        """
        初始化常规测井插值器

        参数:
        - method: 插值方法 ('linear', 'cubic', 'pchip')
        - fill_value: 外推值处理方式
        """
        self.method = method
        self.fill_value = fill_value
        self.scaler = StandardScaler()
        self.is_fitted = False

        # 常规测井曲线类型及其特性
        self.log_types = {
            'GR': {'unit': 'API', 'range': (0, 200), 'trend': 'gradual'},  # 伽马曲线
            'RT': {'unit': 'ohm.m', 'range': (0.01, 10000), 'trend': 'log'},  # 电阻率
            'RM': {'unit': 'ohm.m', 'range': (0.01, 10000), 'trend': 'log'},  # 电阻率
            'Rxo': {'unit': 'ohm.m', 'range': (0.01, 10000), 'trend': 'log'},  # 电阻率
            'NPHI': {'unit': 'v/v', 'range': (-0.15, 0.6), 'trend': 'gradual'},  # 中子孔隙度
            'DEN': {'unit': 'g/cc', 'range': (1.8, 3.0), 'trend': 'gradual'},  # 密度
            'DT': {'unit': 'us/ft', 'range': (40, 140), 'trend': 'gradual'},  # 声波时差
            'CALI': {'unit': 'inch', 'range': (6, 20), 'trend': 'abrupt'},  # 井径
            'SP': {'unit': 'mV', 'range': (-100, 100), 'trend': 'gradual'},  # 自然电位
        }

    def interpolate_logs(self, df, depth_col='DEPTH', target_length=10000,
                         curve_cols=None, return_quality=False):
        """
        对常规测井曲线进行插值

        参数:
        - df: 输入DataFrame，包含深度列和测井曲线列
        - depth_col: 深度列名
        - target_length: 目标数据点数量
        - curve_cols: 需要插值的曲线列名列表，为None时自动选择
        - return_quality: 是否返回插值质量评估

        返回:
        - 插值后的DataFrame
        - 可选: 插值质量评估字典
        """
        # 1. 数据验证和预处理
        self._validate_input(df, depth_col, target_length)

        # 2. 准备数据
        depth_original = df[depth_col].values
        if curve_cols is None:
            curve_cols = self._auto_select_curves(df, depth_col)

        # 3. 生成新的深度序列 (保持原始范围，提高密度)
        depth_new = self._generate_new_depth(depth_original, target_length)

        # 4. 对每条曲线进行插值
        interpolated_data = {}
        quality_metrics = {}

        for col in curve_cols:
            curve_original = df[col].values
            interpolated_curve, metrics = self._interpolate_single_log(
                depth_original, curve_original, depth_new, col
            )
            interpolated_data[col] = interpolated_curve
            quality_metrics[col] = metrics

        # 5. 创建结果DataFrame
        result_df = pd.DataFrame({depth_col: depth_new})
        for col in curve_cols:
            result_df[col] = interpolated_data[col]

        # 6. 后处理和质控
        result_df = self._postprocess_interpolation(result_df, depth_col, curve_cols)

        self.is_fitted = True

        if return_quality:
            return result_df, quality_metrics
        else:
            return result_df

    def _validate_input(self, df, depth_col, target_length):
        """
        验证输入数据的有效性

        参数:
        - df: 输入DataFrame
        - depth_col: 深度列名
        - target_length: 目标长度

        异常:
        - 如果数据无效，抛出相应异常
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("输入必须是pandas DataFrame")

        if depth_col not in df.columns:
            raise ValueError(f"深度列 '{depth_col}' 不存在于DataFrame中")

        if len(df) < 2:
            raise ValueError("数据点太少，至少需要2个点进行插值")

        if target_length <= len(df):
            raise ValueError(f"目标长度 {target_length} 必须大于原始数据长度 {len(df)}")

        # 检查深度列是否单调递增（常规测井要求）
        depth_values = df[depth_col].values
        depth_diff = np.diff(depth_values)

        if np.any(depth_diff <= 0):
            # 尝试排序
            sorted_indices = np.argsort(depth_values)
            if not np.all(np.diff(depth_values[sorted_indices]) > 0):
                raise ValueError("深度列必须严格单调递增")
            else:
                warnings.warn("深度列未排序，已自动排序", UserWarning)

    def _auto_select_curves(self, df, depth_col):
        """
        自动选择数值型测井曲线列

        参数:
        - df: 输入DataFrame
        - depth_col: 深度列名

        返回:
        - 测井曲线列名列表
        """
        # 获取所有数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # 排除深度列
        if depth_col in numeric_cols:
            numeric_cols.remove(depth_col)
        else:
            print('Error Depth config as :{}'.format(numeric_cols))
            raise ValueError("Depth col not in dataframe")

        if not numeric_cols:
            raise ValueError("未找到数值型测井曲线列")

        return numeric_cols

    def _generate_new_depth(self, depth_original, target_length):
        """
        生成新的深度序列

        参数:
        - depth_original: 原始深度序列
        - target_length: 目标长度

        返回:
        - 新的深度序列
        """
        depth_min = np.min(depth_original)
        depth_max = np.max(depth_original)

        # 生成等间隔深度序列
        depth_new = np.linspace(depth_min, depth_max, target_length)

        return depth_new

    def _interpolate_single_log(self, depth_original, curve_original, depth_new, col_name):
        """
        对单条测井曲线进行插值

        参数:
        - depth_original: 原始深度序列
        - curve_original: 原始曲线值
        - depth_new: 新深度序列
        - col_name: 曲线列名

        返回:
        - 插值后的曲线值
        - 插值质量指标
        """
        # 1. 选择适合该曲线类型的插值方法
        method_used = self._select_method_for_log(col_name, len(depth_original))

        # 2. 执行插值
        if len(depth_original) < 2:
            # 数据点不足，返回常数值
            interpolated = np.full_like(depth_new, np.nanmean(curve_original))
            metrics = {'method': 'constant', 'valid_points': len(depth_original)}
        else:
            # 创建插值器
            interpolator = self._create_interpolator(depth_original, curve_original, method_used)

            # 执行插值
            interpolated = interpolator(depth_new)

            # 计算插值质量指标
            metrics = self._calculate_interpolation_quality(
                depth_original, curve_original, depth_new, interpolated, method_used
            )

        return interpolated, metrics

    def _select_method_for_log(self, col_name, n_points):
        """
        根据曲线类型选择最合适的插值方法

        参数:
        - col_name: 曲线名称
        - n_points: 有效数据点数量

        返回:
        - 推荐的插值方法
        """
        # 默认使用初始化时设置的方法
        method = self.method

        # 根据曲线特性调整方法
        if col_name in self.log_types:
            log_trend = self.log_types[col_name]['trend']

            if log_trend == 'abrupt':  # 突变型曲线（如井径）
                # 使用PCHIP保持突变特征
                if n_points >= 4:
                    method = 'pchip'
                else:
                    method = 'linear'
            elif log_trend == 'log':  # 对数型曲线（如电阻率）
                # 使用三次样条平滑插值
                if n_points >= 4:
                    method = 'cubic'
                else:
                    method = 'linear'
            else:  # 渐变型曲线（大多数常规测井曲线）
                # 使用默认方法
                method = self.method

        # 根据数据点数量调整方法
        if n_points < 4 and method in ['cubic', 'pchip']:
            method = 'linear'  # 数据点太少，降级为线性插值

        return method

    def _create_interpolator(self, depth, curve, method):
        """
        创建插值器

        参数:
        - depth: 深度序列
        - curve: 曲线值
        - method: 插值方法

        返回:
        - 插值器对象
        """
        if method == 'linear':
            return interp1d(depth, curve, kind='linear', fill_value=self.fill_value, bounds_error=False)
        elif method == 'cubic':
            return CubicSpline(depth, curve, extrapolate=True)
        elif method == 'pchip':
            return PchipInterpolator(depth, curve, extrapolate=True)
        else:
            raise ValueError(f"不支持的插值方法: {method}")


    def _postprocess_interpolation(self, df, depth_col, curve_cols):
        """
        插值后处理

        参数:
        - df: 插值后的DataFrame
        - depth_col: 深度列名
        - curve_cols: 曲线列名列表

        返回:
        - 后处理后的DataFrame
        """
        # 确保深度列单调递增
        df = df.sort_values(by=depth_col)

        # 对特定曲线类型进行后处理
        for col in curve_cols:
            if col in self.log_types:
                log_info = self.log_types[col]

                # 确保值在合理范围内
                if 'range' in log_info:
                    min_val, max_val = log_info['range']
                    df[col] = np.clip(df[col], min_val, max_val)

                # 对电阻率等对数型曲线进行特殊处理
                if log_info.get('trend') == 'log':
                    # 确保正值
                    df[col] = np.maximum(df[col], 0.01)

        return df

    def _calculate_interpolation_quality(self, depth_orig, curve_orig, depth_new, curve_new, method):
        """
        计算插值质量指标

        参数:
        - depth_orig: 原始深度
        - curve_orig: 原始曲线值
        - depth_new: 新深度
        - curve_new: 插值后曲线值
        - method: 使用的插值方法

        返回:
        - 质量指标字典
        """
        metrics = {
            'method': method,
            'original_points': len(depth_orig),
            'interpolated_points': len(depth_new),
            'density_improvement': len(depth_new) / len(depth_orig)
        }

        # 计算在原始点处的插值误差
        if len(depth_orig) > 1:
            interpolator = self._create_interpolator(depth_orig, curve_orig, method)
            curve_interp_at_orig = interpolator(depth_orig)
            residuals = curve_orig - curve_interp_at_orig

            metrics.update({
                'rmse': np.sqrt(np.mean(residuals ** 2)),
                'mae': np.mean(np.abs(residuals)),
                'max_error': np.max(np.abs(residuals)),
                'r_squared': 1 - np.var(residuals) / np.var(curve_orig) if np.var(curve_orig) > 0 else 0
            })

        return metrics



if __name__ == "__main__":

    # 单独使用示例
    print("\n" + "=" * 60)
    print("单独使用示例")
    print("=" * 60)

    # 创建示例数据
    np.random.seed(42)
    depth = np.linspace(1000, 2000, 250)

    # 模拟常规测井曲线
    gr = 50 + 30 * np.sin(depth / 100) + 8 * np.random.normal(size=250)
    rt = 10 * np.exp(0.001 * depth) + 3 * np.random.lognormal(0, 0.2, 250)
    nphi = 0.3 + 0.1 * np.sin(depth / 200) + 0.04 * np.random.normal(size=250)

    df_example = pd.DataFrame({
        'DEPTH': depth,
        'GR': gr,
        'RT': rt,
        'NPHI': nphi
    })

    # 使用插值器
    interpolator = ConventionalLogInterpolator(method='pchip')
    df_result = interpolator.interpolate_logs(df_example, target_length=5000)

    print(f"示例完成: {len(df_example)}点 -> {len(df_result)}点")

    # 使用类接口进行可视化
    print("创建可视化器...")
    visualizer = WellLogVisualizer()
    try:
        # 启用详细日志
        logging.getLogger().setLevel(logging.INFO)

        visualizer.visualize(
            data=df_result,
            depth_col='DEPTH',
            curve_cols=['GR', 'RT', 'NPHI'],
            type_cols=[],
            # type_cols=['LITHOLOGY', 'FACIES'],
            # legend_dict={
            #     0: '砂岩',
            #     1: '页岩',
            #     2: '石灰岩',
            #     3: '白云岩'
            # },
            fmi_dict=None,
            # depth_limit_config=[320, 380],  # 只显示320-380米段
            figsize=(8, 8)
        )

        # 显示性能统计
        stats = visualizer.get_performance_stats()
        print("性能统计:", stats)

    except Exception as e:
        print(f"可视化过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # 清理资源
        visualizer.close()