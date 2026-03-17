import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Dict, Optional
import warnings
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MultiVariateLinearRegressor:
    """
    多因变量多元线性回归模型类

    实现多目标线性回归：
    y1 = α1*x1 + α2*x2 + α3*x3 + A
    y2 = β1*x1 + β2*x2 + β3*x3 + B

    使用最小二乘法（正规方程）进行参数估计
    """

    def __init__(self, fit_intercept: bool = True):
        """
        初始化多变量线性回归模型

        参数:
        fit_intercept: 是否拟合截距项
        """
        self.fit_intercept = fit_intercept
        self.coef_matrix = None  # 系数矩阵
        self.intercept_matrix = None  # 截距矩阵
        self.x_cols = None  # 特征列名
        self.y_cols = None  # 目标列名
        self.is_fitted = False  # 模型是否已训练
        self.training_metrics = None  # 训练集评估指标

    def fit(self, df: pd.DataFrame, x_cols: List[str], y_cols: List[str]) -> 'MultiVariateLinearRegressor':
        """
        训练多变量线性回归模型

        参数:
        df: 包含特征和目标的DataFrame
        x_cols: 特征列名列表
        y_cols: 目标列名列表

        返回:
        self: 训练好的模型实例
        """
        # 输入验证
        self._validate_input(df, x_cols, y_cols)

        if x_cols is None or len(x_cols) == 0:
            raise ValueError('x_cols is empty')
        if y_cols is None or len(y_cols) == 0:
            raise ValueError('y_cols is empty')

        # 保存列名
        self.x_cols = x_cols
        self.y_cols = y_cols

        # 提取特征和目标矩阵
        X = df[x_cols].values.astype(float)
        Y = df[y_cols].values.astype(float)

        n_samples, n_features = X.shape
        n_targets = len(y_cols)

        # 添加截距项（如果需要）
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(n_samples), X])
        else:
            X_design = X

        try:
            # 使用正规方程求解：θ = (X^T X)^(-1) X^T Y
            XTX = X_design.T @ X_design
            XTX_inv = np.linalg.pinv(XTX)  # 使用伪逆提高稳定性
            XTY = X_design.T @ Y
            theta = XTX_inv @ XTY

            # 分离系数和截距
            if self.fit_intercept:
                self.intercept_matrix = theta[0:1, :].T
                self.coef_matrix = theta[1:, :].T
            else:
                self.intercept_matrix = np.zeros((n_targets, 1))
                self.coef_matrix = theta.T

            self.is_fitted = True

            # 计算训练集评估指标
            train_predictions = self.predict(df)
            self.training_metrics = self._calculate_evaluation_metrics(df[y_cols].values, train_predictions.values, self.y_cols)

            print("模型训练完成！")
            self._print_training_summary()

        except np.linalg.LinAlgError as e:
            raise ValueError("矩阵不可逆，可能存在多重共线性问题。可以尝试：\n"
                             "1. 移除高度相关的特征\n"
                             "2. 使用正则化方法\n"
                             "3. 增加数据量") from e

        return self

    def set_fit_paras(self, x_cols: List[str], y_cols: List[str], coef_matrix, intercept_matrix) -> 'MultiVariateLinearRegressor':
        assert x_cols is not None and y_cols is not None and coef_matrix is not None and intercept_matrix is not None

        self.is_fitted = True
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.coef_matrix = coef_matrix
        self.intercept_matrix = intercept_matrix
        return self

    def predict(self, df: pd.DataFrame, x_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        使用训练好的模型进行预测

        参数:
        df: 包含特征数据的DataFrame

        返回:
        predictions: 预测值的DataFrame
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")

        if x_cols is None or len(x_cols) == 0:
            x_cols = self.x_cols

        # 检查特征列是否存在
        for col in x_cols:
            if col not in df.columns:
                raise ValueError(f"特征列 '{col}' 在测试数据中不存在")

        if len(x_cols) == len(self.x_cols):
            X = df[x_cols].values.astype(float)
        else:
            print(f'X_COLS:{x_cols}输入的列数不对 应该是{len(x_cols)}列:{self.x_cols}')
            exit(0)

        # 进行预测: Y_pred = X @ coef_matrix.T + intercept_matrix.T
        predictions = X @ self.coef_matrix.T + self.intercept_matrix.T

        return pd.DataFrame(predictions, columns=self.y_cols, index=df.index)

    def score(self, df: pd.DataFrame, x_cols: Optional[List[str]] = None, y_cols: Optional[List[str]] = None) -> Dict:
        """
        计算模型在测试集上的评估指标

        参数:
        df: 测试数据（包含真实目标值）
        y_cols: 目标列名（默认为训练时使用的目标列）

        返回:
        metrics: 包含各种评估指标的字典
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        if y_cols is None or len(y_cols) == 0:
            y_cols = self.y_cols
        if x_cols is None or len(x_cols) == 0:
            x_cols = self.x_cols

        self._validate_input(df=df, x_cols=x_cols, y_cols=y_cols)

        # 进行预测
        predictions = self.predict(df, x_cols=x_cols)

        # 提取真实值
        y_true = df[y_cols]

        # 计算评估指标
        metrics = self._calculate_evaluation_metrics(y_true.values, predictions.values, y_cols)

        return metrics

    def _calculate_evaluation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> Dict:
        """
        计算各种评估指标

        参数:
        y_true: 真实值数组
        y_pred: 预测值数组
        target_names: 目标变量名称列表

        返回:
        metrics: 评估指标字典
        """
        # 确保形状一致
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if y_true.shape != y_pred.shape:
            raise ValueError("真实值和预测值的形状不一致")

        n_targets = y_true.shape[1] if y_true.ndim > 1 else 1

        # 如果是一维数组，确保是列向量
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)

        metrics = {}

        # MSE (均方误差)
        mse = np.mean((y_true - y_pred) ** 2, axis=0)
        metrics['MSE'] = dict(zip(target_names, mse))

        # MAE (平均绝对误差)
        mae = np.mean(np.abs(y_true - y_pred), axis=0)
        metrics['MAE'] = dict(zip(target_names, mae))

        # RMSE (均方根误差)
        rmse = np.sqrt(mse)
        metrics['RMSE'] = dict(zip(target_names, rmse))

        # R² (决定系数)
        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))  # 添加小常数防止除零
        metrics['R²'] = dict(zip(target_names, r2))

        return metrics

    def get_params(self) -> Dict:
        """获取模型参数"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        return {
            'coefficients': dict(zip(self.y_cols, self.coef_matrix.tolist())),
            'intercepts': dict(zip(self.y_cols, self.intercept_matrix.flatten().tolist())),
            'feature_names': self.x_cols,
            'target_names': self.y_cols
        }

    def visualize(self, df: pd.DataFrame, figsize: tuple = (15, 10), x_cols:List[str] = None, y_cols:List[str] = None, ):
        """
        可视化预测结果

        参数:
        df: 包含真实目标值的数据
        figsize: 图形大小
        target_cols: 要可视化的目标列（默认全部）
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        if x_cols is None or len(x_cols) == 0:
            x_cols = self.x_cols
        if y_cols is None or len(y_cols) == 0:
            y_cols = self.y_cols
        self._validate_input(df=df, x_cols=x_cols, y_cols=y_cols)

        # 进行预测
        predictions = self.predict(df, x_cols=x_cols)
        y_true = df[y_cols].values

        n_targets = len(y_cols)
        n_cols = min(3, n_targets)  # 每行最多3个子图
        n_rows = (n_targets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # 处理子图数组的维度
        if n_targets == 1:
            axes = np.array([axes])
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # 扁平化axes数组以便迭代
        axes_flat = axes.flatten()

        # 计算整体评估指标
        overall_metrics = self.score(df, x_cols, y_cols)

        for i, target in enumerate(y_cols):
            if i >= len(axes_flat):
                break

            ax = axes_flat[i]
            y_true_col = df[target].values
            y_pred_col = predictions[target].values

            # 绘制散点图
            ax.scatter(y_true_col, y_pred_col, alpha=0.6, s=30)

            # 绘制理想拟合线
            min_val = min(y_true_col.min(), y_pred_col.min())
            max_val = max(y_true_col.max(), y_pred_col.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)

            # 获取该目标的R²
            r2 = overall_metrics['R²'][target]

            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            ax.set_title(f'{target} (R² = {r2:.4f})')
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(n_targets, len(axes_flat)):
            axes_flat[i].set_visible(False)

        plt.tight_layout()
        plt.show()

        # 打印评估指标
        print("\n模型评估指标:")
        print("-" * 50)
        for metric_name, values in overall_metrics.items():
            print(f"{metric_name}:")
            for target, value in values.items():
                print(f"  {target}: {value:.6f}")
            print()

    def visualize_residuals(self, df: pd.DataFrame, figsize: tuple = (15, 10), x_cols:Optional[List[str]]=None, y_cols: Optional[List[str]] = None):
        """
        可视化残差图

        参数:
        df: 包含真实目标值的数据
        figsize: 图形大小
        target_cols: 要可视化的目标列（默认全部）
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")

        if x_cols is None or len(x_cols) == 0:
            x_cols = self.x_cols
        if y_cols is None or len(y_cols) == 0:
            y_cols = self.y_cols
        self._validate_input(df=df, x_cols=x_cols, y_cols=y_cols)

        predictions = self.predict(df, x_cols=x_cols)
        residuals = df[y_cols].values - predictions[y_cols].values

        n_targets = len(y_cols)
        n_cols = min(3, n_targets)
        n_rows = (n_targets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # 处理子图数组的维度
        if n_targets == 1:
            axes = np.array([axes])
        if n_rows * n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        axes_flat = axes.flatten()

        for i, target in enumerate(y_cols):
            if i >= len(axes_flat):
                break

            ax = axes_flat[i]
            ax.scatter(predictions[target], residuals[:, i], alpha=0.6)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('预测值')
            ax.set_ylabel('残差')
            ax.set_title(f'{target} 残差图')
            ax.grid(True, alpha=0.3)

        for i in range(n_targets, len(axes_flat)):
            axes_flat[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def _validate_input(self, df: pd.DataFrame, x_cols: List[str], y_cols: List[str]):
        """验证输入数据"""
        if df is None or df.empty:
            raise ValueError("数据框不能为空")

        if not x_cols:
            raise ValueError("特征列列表不能为空")

        if not y_cols:
            raise ValueError("目标列列表不能为空")

        # 检查列是否存在
        for col in x_cols + y_cols:
            if col not in df.columns:
                raise ValueError(f"列 '{col}' 在数据框中不存在")

        # 检查数据类型
        for col in x_cols + y_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                warnings.warn(f"列 '{col}' 不是数值类型，尝试转换...")
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    raise ValueError(f"无法将列 '{col}' 转换为数值类型")

    def _print_training_summary(self):
        """打印训练摘要"""
        print("=" * 60)
        print("模型训练摘要")
        print("=" * 60)
        print(f"特征数量: {len(self.x_cols)}")
        print(f"目标变量数量: {len(self.y_cols)}")
        print(f"特征名称: {self.x_cols}")
        print(f"目标名称: {self.y_cols}")
        print("\n模型参数:")
        print("-" * 30)

        params = self.get_params()
        for target in self.y_cols:
            print(f"{target}:")
            coefs = params['coefficients'][target]
            intercept = params['intercepts'][target]

            equation = f"    {target} = "
            terms = []
            for feature, coef in zip(self.x_cols, coefs):
                terms.append(f"{coef:.4f}*{feature}")
            equation += " + ".join(terms)
            equation += f" + {intercept:.4f}"
            print(equation)

        print("\n训练集评估指标:")
        print("-" * 30)
        for metric_name, values in self.training_metrics.items():
            print(f"{metric_name}:")
            for target, value in values.items():
                print(f"  {target}: {value:.6f}")
        print("=" * 60)



if __name__ == '__main__':
    # 设置随机种子保证结果可重现
    np.random.seed(42)

    # 创建测试数据（添加一些真实的线性关系）
    n_samples = 200

    # 创建特征
    x1 = np.random.normal(0, 1, n_samples)
    x2 = np.random.normal(0, 1, n_samples)
    x3 = np.random.normal(0, 1, n_samples)
    x4 = np.random.normal(0, 1, n_samples)

    # 创建有真实线性关系的目标变量（添加一些噪声）
    y1 = 2.5 * x1 + 1.5 * x2 - 0.8 * x3 + 1.0 + 1.2*np.random.normal(0, 1, n_samples)
    y2 = -1.2 * x1 + 3.0 * x2 + 0.5 * x3 + 0.5 + 2.1*np.random.normal(0, 1, n_samples)
    y3 = 0.8 * x1 - 1.5 * x2 + 2.0 * x3 - 0.5 + 3.5*np.random.normal(0, 1, n_samples)

    data_test = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'x3': x3,
        'x4': x4,
        'y1': y1,
        'y2': y2,
        'y3': y3,
    })

    print("数据统计描述:")
    print(data_test.describe())
    print("\n" + "=" * 60 + "\n")

    # 测试1: 使用类接口
    print("测试1: 使用类接口")
    print("-" * 40)

    # 创建模型实例
    model = MultiVariateLinearRegressor(fit_intercept=True)

    # 训练模型（使用x1, x2, x3预测y1, y2）
    model.fit(data_test, x_cols=['x1', 'x2', 'x3'], y_cols=['y1', 'y2'])

    # 进行预测
    predictions = model.predict(data_test)

    # 计算评估指标
    test_metrics = model.score(data_test)

    # 可视化结果
    print("\n生成可视化图表...")
    model.visualize(data_test, figsize=(12, 5))
    model.visualize_residuals(data_test, figsize=(12, 5))

