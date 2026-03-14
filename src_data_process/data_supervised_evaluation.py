import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, classification_report, cohen_kappa_score, matthews_corrcoef
)
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Union


def evaluate_supervised_clustering(
        df: pd.DataFrame,
        col_org: str,
        cols_compare: List[str],
        save_report: bool = False,
        report_path: str = "supervised_clustering_evaluation.xlsx",
) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
    """
    评估监督聚类（分类）算法的结果（无绘图版本）

    参数:
    df: 包含原始标签和预测结果的DataFrame
    col_org: 原始分类标签列名
    cols_compare: 需要评估的聚类结果列名列表
    save_report: 是否保存详细报告
    report_path: 报告保存路径

    返回:
    包含各项评估指标的字典
    """
    # 1. 输入验证
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("输入DataFrame不能为空")

    if col_org not in df.columns:
        raise ValueError(f"原始分类列 '{col_org}' 不存在")

    missing_cols = [col for col in cols_compare if col not in df.columns]
    if missing_cols:
        raise ValueError(f"以下聚类结果列不存在: {', '.join(missing_cols)}")

    # 2. 准备结果容器
    results = {}
    report_data = []

    # 3. 对每个聚类结果进行评估
    for col_cluster in cols_compare:
        # 3.1 提取数据
        y_true = df[col_org].values
        y_pred = df[col_cluster].values

        # 3.2 标签编码
        le = LabelEncoder()
        y_true_encoded = le.fit_transform(y_true)
        y_pred_encoded = le.transform(y_pred)
        classes = le.classes_
        n_classes = len(classes)

        # 3.3 计算评估指标
        metrics = calculate_classification_metrics(y_true_encoded, y_pred_encoded, n_classes)

        # 3.4 保存结果
        results[col_cluster] = metrics

        # 3.5 添加到报告数据
        report_row = {
            '聚类方法': col_cluster,
            '准确率': metrics['accuracy'],
            '加权精确率': metrics['precision_weighted'],
            '加权召回率': metrics['recall_weighted'],
            '加权F1分数': metrics['f1_weighted'],
            '宏平均F1分数': metrics['f1_macro'],
            'Cohen Kappa': metrics['cohen_kappa'],
            '类别数量': n_classes
        }
        report_data.append(report_row)

    # 4. 创建详细报告
    report_df = pd.DataFrame(report_data)

    # 5. 保存报告
    if save_report:
        # 使用pandas的ExcelWriter，使用with语句自动关闭
        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            # 写入汇总报告
            report_df.to_excel(writer, sheet_name='汇总报告', index=False)

            # 写入详细分类报告
            for col_cluster in cols_compare:
                # 提取数据
                y_true = df[col_org].values
                y_pred = df[col_cluster].values

                # 标签编码
                le = LabelEncoder()
                y_true_encoded = le.fit_transform(y_true)
                y_pred_encoded = le.transform(y_pred)
                classes = le.classes_

                # 计算混淆矩阵
                cm = confusion_matrix(y_true_encoded, y_pred_encoded)

                # 生成分类报告，添加zero_division参数避免警告
                report = classification_report(
                    y_true_encoded,
                    y_pred_encoded,
                    target_names=classes,
                    output_dict=True,
                    zero_division=0
                )

                # 转换为DataFrame
                report_df_detail = pd.DataFrame(report).transpose()

                # 创建混淆矩阵DataFrame
                cm_df = pd.DataFrame(
                    cm,
                    index=[f"真实: {cls}" for cls in classes],
                    columns=[f"预测: {cls}" for cls in classes]
                )

                # 创建指标行
                # 计算每个类别的指标
                precision_per_class = precision_score(y_true_encoded, y_pred_encoded, average=None, zero_division=0)
                recall_per_class = recall_score(y_true_encoded, y_pred_encoded, average=None, zero_division=0)
                f1_per_class = f1_score(y_true_encoded, y_pred_encoded, average=None, zero_division=0)

                # 计算整体指标
                accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
                precision_macro = precision_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
                recall_macro = recall_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
                f1_macro = f1_score(y_true_encoded, y_pred_encoded, average='macro', zero_division=0)
                precision_weighted = precision_score(y_true_encoded, y_pred_encoded, average='weighted',
                                                     zero_division=0)
                recall_weighted = recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
                f1_weighted = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)

                # 创建指标行
                index_names = [f"真实: {cls}" for cls in classes] + ['F1', 'Precision', 'Recall', 'ALL']

                # 创建指标DataFrame
                metrics_df = pd.DataFrame(index=index_names, columns=[f"预测: {cls}" for cls in classes] + ['指标值'])

                # 填充每个类别的指标
                for i, cls in enumerate(classes):
                    metrics_df.loc['F1', f"预测: {cls}"] = f1_per_class[i]
                    metrics_df.loc['Precision', f"预测: {cls}"] = precision_per_class[i]
                    metrics_df.loc['Recall', f"预测: {cls}"] = recall_per_class[i]

                # 填充整体指标
                metrics_df.loc['ALL', '指标值'] = accuracy
                metrics_df.loc['F1', '指标值'] = f1_macro
                metrics_df.loc['Precision', '指标值'] = precision_macro
                metrics_df.loc['Recall', '指标值'] = recall_macro

                # 创建新的sheet名称（确保唯一）
                # 限制sheet名称长度（Excel限制为31个字符）
                sheet_name = f"{col_cluster[:20]}_详细报告" if len(col_cluster) > 20 else f"{col_cluster}_详细报告"

                # 写入混淆矩阵
                cm_df.to_excel(writer, sheet_name=sheet_name, startrow=0)

                # 写入指标矩阵（在混淆矩阵下方）
                metrics_start_row = len(cm_df) + 2
                metrics_df.to_excel(writer, sheet_name=sheet_name, startrow=metrics_start_row)

                # 写入分类报告（在指标矩阵下方）
                report_start_row = metrics_start_row + len(metrics_df) + 2
                report_df_detail.to_excel(writer, sheet_name=sheet_name, startrow=report_start_row)

        print(f"报告已保存到: {report_path}")

    return results

    # # 5. 保存报告
    # if save_report:
    #     # 保存汇总报告
    #     report_df.to_excel(report_path, index=False)
    #
    #     # 创建Excel写入器（如果文件存在则覆盖）
    #     if os.path.exists(report_path):
    #         writer = pd.ExcelWriter(report_path, engine='openpyxl', mode='w')
    #     else:
    #         writer = pd.ExcelWriter(report_path, engine='openpyxl', mode='a')
    #
    #     # 写入汇总报告
    #     report_df.to_excel(writer, sheet_name='汇总报告', index=False)
    #
    #     # 写入详细分类报告
    #     for col_cluster in cols_compare:
    #         # 提取数据
    #         y_true = df[col_org].values
    #         y_pred = df[col_cluster].values
    #
    #         # 标签编码
    #         le = LabelEncoder()
    #         y_true_encoded = le.fit_transform(y_true)
    #         y_pred_encoded = le.transform(y_pred)
    #         classes = le.classes_
    #
    #         # 计算混淆矩阵
    #         cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    #
    #         # 生成分类报告
    #         report = classification_report(
    #             y_true_encoded,
    #             y_pred_encoded,
    #             target_names=classes,
    #             output_dict=True
    #         )
    #
    #         # 转换为DataFrame
    #         report_df = pd.DataFrame(report).transpose()
    #
    #         #AI看这里，在cm矩阵后面，请加上这个矩阵每一个类别对应的F1、precision、recall三个判别指数，以及整体的F1、precision、recall判别指数，然后index=[f"真实: {cls}" for cls in classes]+['F1', 'precision', 'recall', 'ALL']
    #         # 创建混淆矩阵DataFrame
    #         cm_df = pd.DataFrame(
    #             cm,
    #             index=[f"真实: {cls}" for cls in classes],
    #             columns=[f"预测: {cls}" for cls in classes]
    #         )
    #
    #         # 创建新的sheet名称（确保唯一）
    #         sheet_name = f"{col_cluster}_详细报告"
    #
    #         # 写入混淆矩阵
    #         cm_df.to_excel(writer, sheet_name=sheet_name, startrow=0)
    #
    #         # 写入分类报告（在混淆矩阵下方）
    #         start_row = len(cm_df) + 3  # 空2行
    #         report_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row)
    #
    #     # 保存并关闭Excel文件
    #     writer.save()
    #     writer.close()
    #
    # return results


def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_classes: int
) -> Dict[str, Union[float, np.ndarray]]:
    """
    计算分类评估指标

    参数:
    y_true: 真实标签
    y_pred: 预测标签
    n_classes: 类别数量

    返回:
    包含各项指标的字典
    """
    # 计算基本指标
    accuracy = accuracy_score(y_true, y_pred)
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')

    # 计算额外指标
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    matthews_corr = matthews_corrcoef(y_true, y_pred)

    # 计算每个类别的指标
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)

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
        'f1_per_class': f1_per_class
    }


def test_supervised_clustering_evaluation():
    """
    测试监督聚类结果评估接口
    """
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    n_classes = 4

    # 真实标签
    true_labels = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)

    # 创建完美匹配的预测
    perfect_labels = true_labels.copy()

    # 创建部分匹配的预测（80%准确率）
    partial_labels = true_labels.copy()
    flip_indices = np.random.choice(n_samples, size=int(n_samples * 0.3), replace=False)
    for idx in flip_indices:
        partial_labels[idx] = np.random.choice(['A', 'B', 'C', 'D'])

    # 创建随机预测
    random_labels = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)

    # 创建DataFrame
    data = {
        'ORG': true_labels,
        'Perfect': perfect_labels,
        'Partial': partial_labels,
        'Random': random_labels,
        'RF': np.random.choice(['A', 'B', 'C', 'D'], size=n_samples),
        'KNN': np.random.choice(['A', 'B', 'C', 'D'], size=n_samples),
        'LDA': np.random.choice(['A', 'B', 'C', 'D'], size=n_samples),
        'XGBOOST': np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)
    }
    df = pd.DataFrame(data)

    # 设置评估参数
    col_org = 'ORG'
    cols_compare = ['Perfect', 'Partial', 'Random', 'RF', 'KNN', 'LDA', 'XGBOOST']

    # 调用评估接口
    results = evaluate_supervised_clustering(
        df=df,
        col_org=col_org,
        cols_compare=cols_compare,
        save_report=True,
        report_path='supervised_clustering_evaluation.xlsx',
    )

    # 打印结果
    print("\n监督聚类评估结果:")
    for method, metrics in results.items():
        print(f"\n{method}:")
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  加权F1分数: {metrics['f1_weighted']:.4f}")
        print(f"  宏平均F1分数: {metrics['f1_macro']:.4f}")
        print(f"  Cohen Kappa: {metrics['cohen_kappa']:.4f}")

    return results


# 运行测试
if __name__ == '__main__':
    test_results = test_supervised_clustering_evaluation()
    print("\n测试完成!")