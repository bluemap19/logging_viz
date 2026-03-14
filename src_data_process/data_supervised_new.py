import os
import pandas as pd
import numpy as np
import pickle
import warnings
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, LabelEncoder
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, classification_report, cohen_kappa_score, matthews_corrcoef
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline


class class_supervised_classification:
    """
    监督分类类 - 集成多种机器学习算法进行多类别分类

    主要特性：
    1. 支持多种机器学习算法
    2. 完整的训练-评估-预测流程
    3. 详细的模型性能评估报告
    4. 模型持久化功能

    示例用法:
    >>> clf = class_supervised_classification(cluster_num=3)
    >>> clf.fit(df, cols_x=['feat1', 'feat2'], col_y='label', norm=True)
    >>> results = clf.fit_result(X_test, y_test, save_report=True)
    >>> predictions = clf.predict(new_data)
    """

    # 默认算法配置
    DEFAULT_ALGORITHMS = {
        'MLP': {
            'hidden_layer_sizes': (16, 16, 8),
            'alpha': 1e-4,
            'learning_rate': 'adaptive',
            'max_iter': 500,
            'early_stopping': True,
            'random_state': 42
        },
        'KNN': {
            'n_neighbors': None  # 将在初始化时动态设置
        },
        'SVM': {
            'kernel': 'rbf',
            'probability': True,
            'class_weight': 'balanced',
            'random_state': 42
        },
        'Naive Bayes': {},  # 高斯朴素贝叶斯通常不需要特殊参数
        'Random Forest': {
            'n_estimators': 10,
            'class_weight': 'balanced',
            'random_state': 42
        },
        'GBM': {
            'n_estimators': 10,
            'subsample': 0.8,
            'random_state': 42
        }
    }

    def __init__(
            self,
            algorithms: Dict[str, Dict] = None,
            cluster_num: int = 10,
            path_saved_models: str = None
    ):
        """
        初始化监督分类器

        参数:
        algorithms: 算法配置字典，包含算法名称及对应参数配置
        cluster_num: 分类类别数量，必须 ≥ 2
        path_saved_models: 预训练模型加载路径
        """
        # 参数验证
        if cluster_num < 2:
            raise ValueError("分类类别数量必须大于等于2")

        # 设置算法配置
        if algorithms is None:
            self.algorithms = self.DEFAULT_ALGORITHMS.copy()
        else:
            # 使用默认配置作为基础，然后更新用户提供的配置
            self.algorithms = self.DEFAULT_ALGORITHMS.copy()
            for algo_name, algo_params in algorithms.items():
                self.update_algorithm_params(algo_name, algo_params)
                # if algo_name in self.algorithms:
                #     # 更新现有算法的参数
                #     self.algorithms[algo_name].update(algo_params)
                # else:
                #     # 添加新算法
                #     self.algorithms[algo_name] = algo_params

        # 动态设置KNN的邻居数（如果未设置或需要更新）
        if 'KNN' in self.algorithms and (self.algorithms['KNN'].get('n_neighbors') is None or
                                         self.algorithms['KNN'].get('n_neighbors') > cluster_num):
            self.algorithms['KNN']['n_neighbors'] = min(cluster_num, 10)

        self.cluster_num = cluster_num

        # 初始化类属性
        self.models = {}  # 训练好的模型字典
        self.normalizer = None  # 归一化器对象
        self.feature_columns = []  # 训练使用的特征列名
        self.target_column = None  # 目标变量列名
        self.is_trained = False  # 训练状态标识
        self.class_labels = {}  # 类别标签映射
        self.normalization_method = None  # 归一化方法
        self.X_train = None  # 训练集特征
        self.X_test = None  # 测试集特征
        self.y_train = None  # 训练集标签
        self.y_test = None  # 测试集标签

        # 如果提供了模型路径，尝试加载模型
        if path_saved_models is not None and os.path.exists(path_saved_models):
            self.load_model(path_saved_models)


    def normalize(self, Data: pd.DataFrame, cols_x: List[str] = None, way_norm: str = 'z-score') -> pd.DataFrame:
        """
        数据归一化函数

        参数:
        Data: 输入数据，pandas DataFrame格式
        cols_x: 需要归一化的列名列表，如果为None则归一化所有数值列
        way: 归一化方法，'min-max'或'z-score'

        返回:
        归一化后的DataFrame
        """
        if not isinstance(Data, pd.DataFrame):
            raise TypeError("Data必须是pandas DataFrame")

        if Data.empty:
            raise ValueError("输入数据不能为空")

        # 确定需要归一化的列
        if cols_x is None:
            # 自动选择数值型列
            numeric_cols = Data.columns.tolist()[:-1]
            if len(numeric_cols) == 0:
                raise ValueError("未找到数值型列进行归一化")
            cols_x = numeric_cols
        else:
            # 验证指定的列是否存在
            missing_cols = [col for col in cols_x if col not in Data.columns]
            if missing_cols:
                raise ValueError(f"以下列不存在: {missing_cols}")

        # 创建归一化器
        if way_norm == 'min-max':
            self.normalizer = MinMaxScaler()
        elif way_norm == 'z-score':
            self.normalizer = StandardScaler()
        elif way_norm == 'robust':
            from sklearn.preprocessing import RobustScaler
            self.normalizer = RobustScaler()
        elif way_norm == 'max-abs':
            from sklearn.preprocessing import MaxAbsScaler
            self.normalizer = MaxAbsScaler()
        elif way_norm == 'none' or way_norm is None:
            # 不进行归一化
            self.normalizer = None
            self.normalization_method = 'None'
            return Data.copy()
        else:
            raise ValueError(f"不支持的归一化方法: {way_norm}，支持的方法: min-max, z-score, robust, max-abs, none")

        self.normalization_method = way_norm

        # 执行归一化
        Data_normalized = Data.copy()
        Data_normalized[cols_x] = self.normalizer.fit_transform(Data[cols_x])
        print(f'使用{way_norm}进行列{cols_x}的归一化')

        return Data_normalized

    def _create_classifier_pipeline(self, name: str, params: Dict):
        """
        创建分类器管道，支持自定义归一化

        参数:
        name: 算法名称
        params: 算法参数
        use_normalization: 是否使用归一化

        返回:
        配置好的分类器管道
        """
        # 创建分类器
        if name == 'MLP':
            classifier = MLPClassifier(**params)
        elif name == 'KNN':
            knn_params = params.copy()
            if knn_params.get('n_neighbors') is None:
                knn_params['n_neighbors'] = min(self.cluster_num, 10)
            classifier = KNeighborsClassifier(**knn_params)
        elif name == 'SVM':
            classifier = SVC(**params)
        elif name == 'Naive Bayes':
            classifier = GaussianNB(**params)
        elif name == 'Random Forest':
            classifier = RandomForestClassifier(**params)
        elif name == 'GBM':
            classifier = GradientBoostingClassifier(**params)
        else:
            raise ValueError(f"未知算法: {name}")

        return classifier

    def fit(
            self,
            Data: pd.DataFrame,
            cols_x: List[str] = None,
            col_y: str = None,
            norm: str = None,
            test_size: float = 0.3,
            random_state: int = 42
    ) -> Dict[str, Any]:
        """
        模型训练函数 - 只进行模型训练，不进行评价

        参数:
        Data: 训练数据，包含特征和标签
        cols_x: 特征列名列表，如果为None则使用除col_y外的所有列
        col_y: 目标变量列名，如果为None则使用最后一列
        norm: 是否进行数据归一化
        test_size: 测试集比例
        random_state: 随机种子

        返回:
        包含训练基本信息的字典
        """
        # 数据验证
        if not isinstance(Data, pd.DataFrame):
            raise TypeError("Data必须是pandas DataFrame")

        if Data.empty:
            raise ValueError("输入数据不能为空")

        # 确定特征列和目标列
        if col_y is None:
            self.target_column = Data.columns[-1]
            if cols_x is None:
                self.feature_columns = Data.columns[:-1].tolist()  # 排除最后一列（目标列）
            else:
                self.feature_columns = cols_x
        else:
            self.target_column = col_y
            if cols_x is None:
                self.feature_columns = [col for col in Data.columns if col != col_y]  # 排除目标列
            else:
                self.feature_columns = cols_x

        # 验证列存在性
        if self.target_column not in Data.columns:
            raise ValueError(f"目标列 '{self.target_column}' 不存在")

        missing_feature_cols = [col for col in self.feature_columns if col not in Data.columns]
        if missing_feature_cols:
            raise ValueError(f"以下特征列不存在: {missing_feature_cols}")

        # 检查类别数量
        unique_classes = Data[self.target_column].nunique()
        if unique_classes < 2:
            raise ValueError("目标变量必须至少包含2个类别")
        if unique_classes != self.cluster_num:
            warnings.warn(f"数据中的类别数量({unique_classes})与初始化参数cluster_num({self.cluster_num})不一致")

        # 进行正则化的预处理
        if norm is not None:
            # 只对特征列进行归一化
            Data_normalized = self.normalize(Data, self.feature_columns, norm)

            # 确保归一化后的数据包含目标列
            if self.target_column not in Data_normalized.columns:
                Data_normalized[self.target_column] = Data[self.target_column]

            Data = Data_normalized

        # 数据预处理
        X = Data[self.feature_columns]
        y = Data[self.target_column]

        # 记录类别标签映射
        self.class_labels = {i: label for i, label in enumerate(y.astype('category').cat.categories)}

        # 训练测试集分割（分层抽样）
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # 转换目标变量为整数
        self.y_train = self.y_train.astype(int)
        self.y_test = self.y_test.astype(int)

        # 模型训练（不进行评价）
        self.models = {}
        trained_models = []

        for name, params in self.algorithms.items():
            try:
                # 使用新的管道创建函数
                clf = self._create_classifier_pipeline(name, params)

                # 训练模型
                clf.fit(self.X_train, self.y_train)
                self.models[name] = clf
                trained_models.append(name)
                print(f"算法 {name} 训练完成")

            except Exception as e:
                warnings.warn(f"算法 {name} 训练失败: {str(e)}")
                continue

        # 检查训练结果
        if not self.models:
            raise RuntimeError("所有算法训练失败，请检查数据和参数配置")

        self.is_trained = True

        # 返回训练基本信息
        return {
            'models_trained': trained_models,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'class_labels': self.class_labels,
            'train_size': len(self.X_train),
            'test_size': len(self.X_test)
        }

    def fit_result(
            self,
            Data: pd.DataFrame = None,
            x_cols: List[str] = None,
            y_col: str = None,
            save_report: bool = False,
            report_path: str = "model_evaluation_report.xlsx"
    ) -> Dict[str, Dict[str, Any]]:
        """
        模型训练结果评估函数 - 进行全面的模型评价和结果输出

        参数:
        X_test: 测试集特征，如果为None则使用训练时分割的测试集
        y_test: 测试集标签，如果为None则使用训练时分割的测试集标签
        save_report: 是否保存详细评估报告
        report_path: 报告保存路径

        返回:
        包含所有模型评估结果的字典
        """
        # 检查模型是否已训练
        self._check_trained()

        # # 确定测试数据
        if Data is None:
            X_eval = self.X_test
            y_eval = self.y_test
        else:
            # 使用提供的DataFrame进行验证
            if not isinstance(Data, pd.DataFrame):
                raise TypeError("Data必须是pandas DataFrame")

            if Data.empty:
                raise ValueError("输入数据不能为空")

            if y_col is None:
                y_col = self.target_column
            if y_col not in Data.columns:
                raise ValueError(f"目标列 '{y_col}' 在数据中不存在")

            if x_cols is None:
                if not set(self.feature_columns).issubset(set(Data.columns)):
                    missing_cols = set(self.feature_columns) - set(Data.columns)
                    raise ValueError(f"以下特征列在数据中不存在: {missing_cols}")
                x_cols = self.feature_columns

            X_eval = Data[x_cols]
            y_eval = Data[y_col].astype(int)

        # 准备结果容器
        results = {}

        # 对每个模型进行评估
        for model_name, clf in self.models.items():
            try:
                print(f"\n正在评估模型: {model_name}")

                # 模型预测
                y_pred = clf.predict(X_eval)

                # 计算评估指标
                model_metrics = self._calculate_classification_metrics(y_eval, y_pred)

                # 保存结果
                results[model_name] = model_metrics

                # 打印简要 结果
                print(f" - 准确率: {model_metrics['accuracy']:.4f}")
                print(f" - 加权F1分数: {model_metrics['f1_weighted']:.4f}")

            except Exception as e:
                warnings.warn(f"模型 {model_name} 评估失败: {str(e)}")
                continue

        # 保存详细报告（解耦版本）
        if save_report and results:
            self._save_evaluation_report(results, report_path)

        return results

    def _calculate_classification_metrics(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        计算分类评估指标

        参数:
        y_true: 真实标签数组
        y_pred: 预测标签数组

        返回:
        包含各项评估指标的字典
        """
        # 基本指标计算
        accuracy = accuracy_score(y_true, y_pred)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # 高级指标计算
        cohen_kappa = cohen_kappa_score(y_true, y_pred)
        matthews_corr = matthews_corrcoef(y_true, y_pred)

        # 各类别指标
        n_classes = len(np.unique(y_true))
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)

        # 分类报告
        report = classification_report(
            y_true, y_pred,
            output_dict=True,
            zero_division=0
        )

        return {
            'accuracy': accuracy,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'cohen_kappa': cohen_kappa,
            'matthews_corrcoef': matthews_corr,
            'n_classes': n_classes,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'classification_report': report,
        }

    def predict(self, Data: pd.DataFrame, cols_x: List[str] = None, norm: str = None) -> pd.DataFrame:
        """
        使用训练好的模型进行预测

        参数:
        Data: 待预测数据，需与训练数据特征一致
        cols_x: 特征列名列表，如果为None则使用训练时的特征列
        norm: 是否进行归一化，如果为None则使用训练时的设置

        返回:
        预测结果 DataFrame ，每列对应不同模型的预测结果
        """
        # 检查模型是否已训练
        self._check_trained()

        # 数据验证
        if not isinstance(Data, pd.DataFrame):
            raise TypeError("Data必须是pandas DataFrame")

        if Data.empty:
            raise ValueError("输入数据不能为空")

        # 确定特征列
        if cols_x is None:
            cols_x = self.feature_columns

        # 验证特征列存在性
        missing_cols = [col for col in cols_x if col not in Data.columns]
        if missing_cols:
            raise ValueError(f"以下特征列不存在: {missing_cols}")

        # 归一化处理 - 修复版本
        X_predict = Data[cols_x].copy()  # 创建特征数据的副本

        ############ 将数据进行正则化
        if norm is not None:
            if norm == self.normalization_method and self.normalizer is not None:
                # 直接尝试转换
                X_predict_normalized = self.normalizer.transform(X_predict)
                X_predict = pd.DataFrame(X_predict_normalized, columns=cols_x, index=Data.index)
            else:
                warnings.warn(f"预测时使用的归一化方法'{norm}'与训练时的方法'{self.normalization_method}'不一致，这可能导致性能下降")
                temp_normalized = self.normalize(Data, cols_x, norm)
                X_predict = temp_normalized[cols_x]

        # 初始化结果容器
        predictions = pd.DataFrame(index=Data.index)

        # 遍历模型进行预测
        for model_name, clf in self.models.items():
            try:
                y_pred = clf.predict(X_predict)
                predictions[model_name] = y_pred.astype(int)
                print(f"模型 {model_name} 预测完成")
            except Exception as e:
                warnings.warn(f"模型 {model_name} 预测失败: {str(e)}")
                predictions[model_name] = np.nan

        # 数据验证
        if predictions.isnull().sum().sum() > 0:
            warnings.warn("部分模型预测存在缺失值，请检查输入数据与模型兼容性")

        return predictions

    def save_model(self, path: str):
        """
        保存模型到指定路径

        参数:
        path: 保存路径
        """
        if not self.is_trained:
            raise RuntimeError("没有训练好的模型可以保存")

        model_data = {
            'models': self.models,
            'normalizer': self.normalizer,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'class_labels': self.class_labels,
            'cluster_num': self.cluster_num,
            'normalization_method': self.normalization_method,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"模型已保存到 {path}")

    def load_model(self, path: str):
        """
        从指定路径加载模型

        参数:
        path: 模型文件路径
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件 {path} 不存在")

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.models = model_data['models']
        self.normalizer = model_data['normalizer']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        self.class_labels = model_data['class_labels']
        self.cluster_num = model_data['cluster_num']
        self.normalization_method = model_data['normalization_method']
        self.X_train = model_data['X_train']
        self.X_test = model_data['X_test']
        self.y_train = model_data['y_train']
        self.y_test = model_data['y_test']
        self.is_trained = True

        print(f"模型已从 {path} 加载")

    def get_available_algorithms(self) -> List[str]:
        """
        返回当前支持的算法列表

        返回:
        算法名称列表
        """
        return list(self.algorithms.keys())

    def update_algorithm_params(self, algorithm_name: str, new_params: Dict):
        """
        动态更新特定算法的参数配置

        参数:
        algorithm_name: 算法名称
        new_params: 新的参数配置
        """
        if algorithm_name not in self.algorithms:
            raise ValueError(f"算法 {algorithm_name} 不存在")

        self.algorithms[algorithm_name].update(new_params)
        print(f"算法 {algorithm_name} 参数已更新")


    def _check_trained(self):
        """检查模型是否已训练"""
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用fit方法")

    def _save_evaluation_report(
            self,
            results: Dict[str, Dict],
            report_path: str
    ):
        """
        保存详细的评估报告到Excel文件 - 简洁功能版

        参数:
        results: 所有模型的评估结果
        report_path: 报告保存路径
        """
        try:
            with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
                # 1. 创建汇总报告工作表
                self._create_summary_sheet(writer, results)

                # 2. 为每个模型创建详细报告
                for model_name, metrics in results.items():
                    self._create_model_combined_sheet(writer, model_name, metrics)

            print(f"评估报告已保存到: {report_path}")

        except Exception as e:
            warnings.warn(f"保存评估报告失败: {str(e)}")
            raise

    def _create_summary_sheet(self, writer, results):
        """创建模型汇总报告工作表"""
        summary_data = []

        for model_name, metrics in results.items():
            summary_row = {
                '模型名称': model_name,
                '准确率': f"{metrics['accuracy']:.4f}",
                '加权精确率': f"{metrics['precision_weighted']:.4f}",
                '加权召回率': f"{metrics['recall_weighted']:.4f}",
                '加权F1分数': f"{metrics['f1_weighted']:.4f}",
                '宏平均F1分数': f"{metrics['f1_macro']:.4f}",
                'Cohen Kappa': f"{metrics['cohen_kappa']:.4f}",
                'Matthews相关系数': f"{metrics['matthews_corrcoef']:.4f}",
                '类别数量': metrics['n_classes']
            }
            summary_data.append(summary_row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='模型汇总', index=False)

    def _create_model_combined_sheet(self, writer, model_name, metrics):
        """为单个模型创建综合报告表格（包含混淆矩阵和所有指标）"""
        # 限制sheet名称长度
        sheet_name = f"{model_name[:25]}" if len(model_name) > 25 else model_name

        # 创建综合表格
        combined_df = self._build_combined_table(metrics)

        # 写入Excel
        combined_df.to_excel(writer, sheet_name=sheet_name, index=True)

        # 设置合适的列宽
        worksheet = writer.sheets[sheet_name]
        for col_idx, column in enumerate(combined_df.columns, 1):
            col_letter = chr(64 + col_idx)  # A, B, C, ...
            worksheet.column_dimensions[col_letter].width = 12

    def _build_combined_table(self, metrics):
        """构建包含混淆矩阵和所有指标的综合表格"""
        n_classes = metrics['n_classes']
        cm = metrics['confusion_matrix']
        precision_per_class = metrics['precision_per_class']
        recall_per_class = metrics['recall_per_class']
        f1_per_class = metrics['f1_per_class']

        # 创建行索引和列索引
        row_index = [f"真实{i}" for i in range(n_classes)] + ['真实-总的', 'F1', 'Precision', 'Recall']
        col_index = [f"预测{i}" for i in range(n_classes)] + ['总的']

        # 初始化DataFrame
        combined_df = pd.DataFrame(index=row_index, columns=col_index)

        # 1. 填充混淆矩阵部分
        for i in range(n_classes):
            for j in range(n_classes):
                combined_df.iloc[i, j] = cm[i, j]

        # 2. 填充行和（真实-总的）
        row_sums = cm.sum(axis=1)
        for i in range(n_classes):
            combined_df.iloc[i, n_classes] = row_sums[i]  # 总的列

        # 3. 填充列和（预测-总的）
        col_sums = cm.sum(axis=0)
        for j in range(n_classes):
            combined_df.iloc[n_classes, j] = col_sums[j]  # 真实-总的行

        # 4. 填充总样本数（右下角）
        total_samples = cm.sum()
        combined_df.iloc[n_classes, n_classes] = total_samples

        # 5. 填充各类别的F1分数
        for i in range(n_classes):
            combined_df.iloc[n_classes + 1, i] = f"{f1_per_class[i]:.4f}"  # F1行

        # 6. 填充各类别的精确率
        for i in range(n_classes):
            combined_df.iloc[n_classes + 2, i] = f"{precision_per_class[i]:.4f}"  # Precision行

        # 7. 填充各类别的召回率
        for i in range(n_classes):
            combined_df.iloc[n_classes + 3, i] = f"{recall_per_class[i]:.4f}"  # Recall行

        # 8. 填充整体指标（总的列）
        # F1宏平均
        combined_df.iloc[n_classes + 1, n_classes] = f"{metrics['f1_macro']:.4f}"
        # 精确率加权平均
        combined_df.iloc[n_classes + 2, n_classes] = f"{metrics['precision_weighted']:.4f}"
        # 召回率加权平均
        combined_df.iloc[n_classes + 3, n_classes] = f"{metrics['recall_weighted']:.4f}"

        # 9. 添加准确率（放在表格右下角下方）
        # 我们可以在表格下方添加一行来显示准确率
        accuracy_row = pd.DataFrame([[f"{metrics['accuracy']:.4f}"] + [''] * (n_classes)],
                                    index=['准确率'],
                                    columns=combined_df.columns)
        ck_row = pd.DataFrame([[f"{metrics['cohen_kappa']:.4f}"] + [''] * (n_classes)],
                                    index=['cohen kappa'],
                                    columns=combined_df.columns)
        mc_row = pd.DataFrame([[f"{metrics['matthews_corrcoef']:.4f}"] + [''] * (n_classes)],
                                    index=['matthews corrcoef'],
                                    columns=combined_df.columns)

        # 合并表格
        final_df = pd.concat([combined_df, accuracy_row, ck_row, mc_row])

        return final_df


# 测试代码
if __name__ == '__main__':
    # 创建测试数据
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    n_classes = 4

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'Feature_{i + 1}' for i in range(n_features)]
    )

    y_values = np.array([0] * 40 + [1] * 100 + [2] * 50 + [3] * 10)
    np.random.shuffle(y_values)
    y = pd.Series(y_values, name='label')

    Data = pd.concat([X, y], axis=1)
    print(Data.describe())

    print("测试数据集信息:")
    print(f"特征形状: {X.shape}")
    print(f"目标变量分布:\n{y.value_counts().sort_index()}")

    # 测试类功能
    print("\n" + "=" * 50)
    print("测试 class_supervised_classification")
    print("=" * 50)

    # 创建分类器实例
    clf = class_supervised_classification(cluster_num=n_classes)
    print(clf.get_available_algorithms())

    # 训练模型（不进行评价）
    print("开始训练模型...")
    train_info = clf.fit(Data, col_y='label', norm='min-max')
    print(f"训练完成，共训练 {len(train_info['models_trained'])} 个模型")

    # 进行模型评价
    print("\n开始模型评价...")
    evaluation_results = clf.fit_result(save_report=False)

    print("\n模型评价结果摘要:")
    for model_name, metrics in evaluation_results.items():
        print(f"{model_name}: 准确率={metrics['accuracy']:.4f}, F1加权={metrics['f1_weighted']:.4f}")

    # 预测新数据
    print("\n进行预测...")
    sample_data = Data.iloc[:10]
    predictions = clf.predict(sample_data, norm='z-score')
    print("预测结果:")
    print(predictions)

    # 模型保存和加载测试
    print("\n测试模型保存和加载...")
    clf.save_model('test_model.pkl')

    # 创建新实例并加载模型
    clf_loaded = class_supervised_classification(path_saved_models='test_model.pkl')

    # 使用加载的模型进行预测
    predictions_loaded = clf_loaded.predict(sample_data)
    print("加载模型的预测结果:")
    print(predictions_loaded)

    # 清理测试文件
    for file in ['test_model.pkl', 'test_evaluation_report.xlsx']:
        if os.path.exists(file):
            os.remove(file)
            print(f"已清理测试文件: {file}")

    print("\n所有测试完成!")