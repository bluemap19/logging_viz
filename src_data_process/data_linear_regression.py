import numpy as np
import pandas as pd


def multi_linear_Regressor(df:pd.DataFrame = None,
                           x_cols: list=['x1', 'x2', 'x3'],
                           y_cols: list=['y1', 'y2'],):
    """
    多因变量多元线性回归模型
    y1 = α1*x1 + α2*x2 + α3*x3 + A
    y2 = β1*x1 + β2*x2 + β3*x3 + B
    类似的多因式回归，回归的目标是使其
    y1 y2 的计算值，与 y1 y2 的实际值的响应 的MSE或者是MAE误差逐渐减小，请你使用牛顿迭代法或者是其他计算方法实现其功能，要求是最后返回的是参数矩阵：
    系数矩阵[[α1, α2, α3], [β1, β2, β3]]
    截距矩阵 [[A], [B]]
    请你实现该功能，并添加极其详细的注释
    :param df:
    :param x_cols:
    :param y_cols:
    :return:
    """
    # 输入数据验证
    if df is None:
        raise ValueError("数据框不能为空")

    # 检查必需的列是否存在
    for col in x_cols + y_cols:
        if col not in df.columns:
            raise ValueError(f"列 '{col}' 在数据框中不存在")

    # 提取特征矩阵X和目标矩阵Y
    # X的形状: (样本数, 特征数)
    # Y的形状: (样本数, 目标变量数)
    X = df[x_cols].values.astype(float)
    Y = df[y_cols].values.astype(float)

    # 获取样本数和特征数
    n_samples, n_features = X.shape
    n_targets = len(y_cols)

    # 添加偏置项（截距项）到特征矩阵
    # 在X矩阵左侧添加一列1，用于计算截距
    X_with_intercept = np.column_stack([np.ones(n_samples), X])

    # 使用最小二乘法求解多元线性回归
    # 正规方程: θ = (X^T * X)^(-1) * X^T * Y
    # 其中θ包含所有系数和截距

    try:
        # 计算X^T * X
        XTX = np.dot(X_with_intercept.T, X_with_intercept)

        # 计算(X^T * X)的逆矩阵
        # 使用伪逆提高数值稳定性，防止矩阵奇异的情况
        XTX_inv = np.linalg.pinv(XTX)

        # 计算X^T * Y
        XTY = np.dot(X_with_intercept.T, Y)

        # 计算参数矩阵θ: (n_features+1) × n_targets
        # 每一列对应一个目标变量的参数
        theta = np.dot(XTX_inv, XTY)

    except np.linalg.LinAlgError as e:
        raise ValueError("矩阵不可逆，可能存在多重共线性问题") from e

    # 分离截距项和系数项
    # theta矩阵的第一行是截距项，其余行是系数
    intercept_matrix = theta[0:1, :].T  # 形状: (n_targets, 1)
    coef_matrix = theta[1:, :].T  # 形状: (n_targets, n_features)

    return coef_matrix, intercept_matrix


def calculate_predictions(df, x_cols, y_cols, coef_matrix, intercept_matrix):
    """
    使用训练好的模型进行预测

    参数:
    df: 测试数据
    x_cols: 特征列
    y_cols: 目标列
    coef_matrix: 系数矩阵
    intercept_matrix: 截距矩阵

    返回:
    predictions: 预测值DataFrame
    """
    X = df[x_cols].values
    predictions = np.dot(X, coef_matrix.T) + intercept_matrix.T
    return pd.DataFrame(predictions, columns=y_cols)


def calculate_metrics(df, y_cols, predictions):
    """
    计算模型评估指标

    参数:
    df: 原始数据
    y_cols: 目标列
    predictions: 预测值

    返回:
    metrics: 包含MSE和MAE的字典
    """
    Y_true = df[y_cols].values
    Y_pred = predictions.values

    # 计算均方误差(MSE)
    mse = np.mean((Y_true - Y_pred) ** 2, axis=0)

    # 计算平均绝对误差(MAE)
    mae = np.mean(np.abs(Y_true - Y_pred), axis=0)

    return {
        'MSE': dict(zip(y_cols, mse)),
        'MAE': dict(zip(y_cols, mae))
    }


if __name__ == '__main__':

    data_test = pd.DataFrame({
        'x1': np.random.random(size=(200, )),
        'x2': np.random.random(size=(200, )),
        'x3': np.random.random(size=(200, )),
        'x4': np.random.random(size=(200, )),
    })

    data_test[['y1', 'y2', 'y3']] = pd.DataFrame({
        'y1': 0.3*data_test['x1']+0.2*data_test['x2']+0.3*data_test['x3']+np.random.random(size=(200, )),
        'y2': 0.3*data_test['x1']+0.2*data_test['x3']+0.3*data_test['x4']+np.random.random(size=(200, )),
        'y3': 0.3*data_test['x2']+0.2*data_test['x3']+0.3*data_test['x4']+np.random.random(size=(200, )),
    })

    print(data_test.describe())
    print("\n" + "=" * 50 + "\n")

    # 测试函数
    try:
        # 使用前3个特征预测前2个目标变量
        coef_matrix, intercept_matrix = multi_linear_Regressor(
            df=data_test,
            x_cols=['x1', 'x2', 'x3'],
            y_cols=['y1']
        )

        print("系数矩阵 (coef_matrix):")
        print(coef_matrix)
        print("\n截距矩阵 (intercept_matrix):")
        print(intercept_matrix)
        print("\n" + "=" * 50 + "\n")

        # 进行 预测
        predictions = calculate_predictions(
            data_test,
            ['x1', 'x2', 'x3'],
            ['y1'],
            coef_matrix,
            intercept_matrix
        )

        # 计算 评估 指标
        metrics = calculate_metrics(data_test, ['y1'], predictions)

        print("模型评估指标:")
        for metric_name, values in metrics.items():
            print(f"{metric_name}:")
            for target, value in values.items():
                print(f"  {target}: {value:.6f}")

        print("\n" + "=" * 50 + "\n")

        # 显示前5个样本的实际值和预测值对比
        print("前5个样本的预测对比:")
        comparison = pd.DataFrame({
            'y1_实际': data_test['y1'].head(),
            'y1_预测': predictions['y1'].head(),
        })
        print(comparison)

    except Exception as e:
        print(f"错误: {e}")