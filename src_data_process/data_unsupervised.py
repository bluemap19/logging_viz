import math
from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import contingency_matrix
from scipy.spatial.distance import pdist, cdist
from sklearn.datasets import make_classification
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
# 在 Windows 系统上使用 MKL（Intel Math Kernel Library）时，KMeans 可能存在内存泄漏问题，特别是当数据划分的块数少于可用线程数时。
# os.environ["OMP_NUM_THREADS"] = "4"  # 4 是一个保守值，可根据 CPU 核心数调整
# 设置线程数，通常设置为CPU核心数的一半或更少
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

class ClusteringPipeline:
    def __init__(self,
                 algorithms: dict = None,
                 scale_data: bool = True,
                 cluster_num: int = 10,
                 saved_model_path: str = None):
        """
        增强的无监督聚类集成接口

        参数：
        algorithms : 包含算法名称和实例的字典，默认包含5种算法
        scale_data : 是否自动标准化数据（推荐开启）
        cluster_num : 默认聚类簇数
        saved_model_path : 预训练模型的加载路径
        """
        self.scaler = StandardScaler() if scale_data else None
        self.cluster_num = cluster_num
        self.algorithm_config = algorithms or self._default_algorithm_config()
        self.algorithms = {}  # 实际的算法实例将在fit中初始化

        # 用于存储训练后的模型
        self.models_ = {}
        self.scaler_ = None

        # 加载预训练模型
        if saved_model_path:
            self.load_model(saved_model_path)
        else:
            # 初始化算法配置
            self._init_algorithm_config(None)

    def _default_algorithm_config(self):
        """ 默认算法配置（只保存配置，不初始化实例）"""
        return {
            'KMeans': {'type': 'kmeans'},
            'DBSCAN': {'type': 'dbscan', 'eps': None, 'min_samples': None},
            'Hierarchical': {'type': 'hierarchical'},
            'Spectral': {'type': 'spectral'},
            'GMM': {'type': 'gmm'},
            'SOM': {'type': 'som', 'grid_rows': None, 'grid_cols': None,
                    'sigma': 0.5, 'learning_rate': 0.5, 'num_iteration': 1000}
        }

    def _init_algorithm_config(self, X):
        """ 根据数据初始化算法实例 """
        # DBSCAN需要特殊处理
        if 'DBSCAN' in self.algorithm_config:
            config = self.algorithm_config['DBSCAN']

            # 如果未提供参数且数据可用，则自动计算合理的默认值
            if X is not None:
                # 1. 基于数据特征的自动配置策略
                n_samples, n_features = X.shape

                # 默认min_samples设置更保守（平衡噪声点和簇大小）
                min_samples_default = max(10, int(0.01 * n_samples))  # 至少20个点，最多5%的样本数

                # 默认eps设置更加智能（考虑特征尺度）
                # 计算特征标准差的中位数作为参考
                if hasattr(self, 'scaler_') and self.scaler_ is not None:
                    # 使用标准化后的数据估计特征尺度
                    scaler_mean = self.scaler_.mean_ if hasattr(self.scaler_, 'mean_') else 0
                    scaler_scale = self.scaler_.scale_ if hasattr(self.scaler_, 'scale_') else 1
                    feature_stds = np.std(X, axis=0) / scaler_scale
                    median_std = np.median(feature_stds)

                    # eps设置：基于数据分布密度
                    eps_default = 0.3 * median_std * n_features ** 0.5  # 维度敏感性调整
                else:
                    # 未标准化时，使用经验值
                    eps_default = 0.5

                # 设置默认值（仅当用户未提供时）
                if config.get('eps') is None:
                    config['eps'] = eps_default
                    print(f"DBSCAN: 自动设置 eps = {eps_default:.4f} (基于数据特征尺度)")

                if config.get('min_samples') is None:
                    config['min_samples'] = min_samples_default
                    print(f"DBSCAN: 自动设置 min_samples = {min_samples_default} (基于样本数量)")

            # 创建DBSCAN实例
            self.algorithms['DBSCAN'] = DBSCAN(
                eps=config.get('eps', 0.5),
                min_samples=config.get('min_samples', 10)
            )

        if 'SOM' in self.algorithm_config:
            config = self.algorithm_config['SOM']

            # 自动确定网格大小（如果未指定）
            grid_rows = config.get('grid_rows')
            grid_cols = config.get('grid_cols')

            # 基于聚类数和数据维度自动确定网格大小
            if grid_rows is None or grid_cols is None:
                # SOM网格尺寸的经验公式：基于聚类数的平方根
                grid_size = max(3, int(math.sqrt(self.cluster_num)))
                grid_rows = grid_size
                grid_cols = math.ceil(self.cluster_num / grid_rows)
                config['grid_rows'] = grid_rows
                config['grid_cols'] = grid_cols
                print(f"SOM: 自动设置网格大小: {grid_rows}x{grid_cols}")

            # 创建SOM实例
            self.algorithms['SOM'] = MiniSom(
                grid_rows,
                grid_cols,
                X.shape[1] if X is not None else 1,  # 特征维度
                sigma=config.get('sigma', 0.5),
                learning_rate=config.get('learning_rate', 0.5),
                random_seed=42
            )
        # ===========================================


        # 初始化其他算法实例
        for name, config in self.algorithm_config.items():
            if name == 'DBSCAN':  # 已经在上面处理
                continue

            if config['type'] == 'kmeans':
                self.algorithms[name] = KMeans(n_clusters=self.cluster_num, random_state=42, n_init=10)
            elif config['type'] == 'hierarchical':
                self.algorithms[name] = AgglomerativeClustering(n_clusters=self.cluster_num)
            elif config['type'] == 'spectral':
                self.algorithms[name] = SpectralClustering(n_clusters=self.cluster_num, random_state=42)
            elif config['type'] == 'gmm':
                self.algorithms[name] = GaussianMixture(n_components=self.cluster_num, random_state=42)


    def fit(self, X: pd.DataFrame, save_path: str = None):
        """
        执行所有聚类算法并进行模型训练

        参数：
        X : N*M的DataFrame，M为特征维度
        save_path : 模型保存路径（可选）

        返回：
        self : 返回实例本身
        """
        self.X_ = X.copy()
        self.results_ = pd.DataFrame(index=X.index)
        self.evaluation_ = {}
        self.models_ = {}

        # 数据标准化
        if self.scaler:
            self.X_scaled_ = self.scaler.fit_transform(X)
            self.scaler_ = self.scaler
        else:
            self.X_scaled_ = X.values

        # 初始化算法实例（使用数据信息）
        self._init_algorithm_config(X)

        # 执行所有聚类算法
        for name, algo in self.algorithms.items():
            try:
                # 训练模型
                if name == 'GMM':
                    algo.fit(self.X_scaled_)
                    labels = algo.predict(self.X_scaled_)

                # ============== 添加SOM处理 ==============
                elif name == 'SOM':
                    config = self.algorithm_config['SOM']

                    # 训练SOM
                    algo.train(self.X_scaled_, config.get('num_iteration', 1000), verbose=False)

                    # 获取获胜神经元位置
                    winners = np.array([algo.winner(x) for x in self.X_scaled_])

                    # 将位置转换为单一簇标签
                    win_map = winners[:, 0] * algo._weights.shape[1] + winners[:, 1]
                    labels, unique_labels = pd.factorize(win_map)

                    # 保存模型和结果
                    self.models_[name] = algo
                    self.results_[f'{name}'] = labels

                    # 评估聚类质量
                    if len(unique_labels) > 1:
                        self.evaluation_[name] = silhouette_score(self.X_scaled_, labels)
                    else:
                        self.evaluation_[name] = float('nan')

                    print(f"SOM: 发现 {len(unique_labels)} 个簇")
                # ===========================================

                else:
                    labels = algo.fit_predict(self.X_scaled_)

                # 保存训练后的模型
                self.models_[name] = algo

                # 保存聚类结果
                self.results_[f'{name}'] = labels

                # 评估聚类质量（Silhouette Score）
                if (len(np.unique(labels)) > 1) and (name != 'DBSCAN'):  # DBSCAN可能有很多噪声点
                    # 对于DBSCAN，评估时忽略噪声点
                    if name == 'DBSCAN':
                        valid_mask = labels != -1
                        if np.sum(valid_mask) > 1:  # 需要至少2个有效点
                            self.evaluation_[name] = silhouette_score(self.X_scaled_[valid_mask], labels[valid_mask])
                        else:
                            self.evaluation_[name] = float('nan')
                    else:
                        self.evaluation_[name] = silhouette_score(self.X_scaled_, labels)
                else:
                    self.evaluation_[name] = float('nan')

            except Exception as e:
                print(f"[WARNING] {name} failed: {str(e)}")
                self.results_[f'{name}'] = np.nan
                self.evaluation_[name] = float('nan')

        # 打印DBSCAN聚类信息
        if 'DBSCAN' in self.results_:
            dbscan_labels = self.results_['DBSCAN']
            if dbscan_labels.isna().any():
                print("[WARNING] DBSCAN returned NaN values")
            else:
                n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                n_noise = sum(dbscan_labels == -1)
                print(
                    f"DBSCAN: 发现 {n_clusters} 个簇，{n_noise} 个噪声点 (eps={self.algorithm_config['DBSCAN']['eps']}, min_samples={self.algorithm_config['DBSCAN']['min_samples']})")

        # 保存最佳模型
        if save_path:
            self.save_model(save_path)

        # 打印评估结果
        print("\n聚类算法评估得分（轮廓系数）：")
        for algo, score in self.evaluation_.items():
            if not np.isnan(score):
                print(f"{algo}: {score:.4f}")

        # 找到最佳算法（排除无效分数）
        valid_scores = {k: v for k, v in self.evaluation_.items() if not np.isnan(v)}
        if valid_scores:
            best_algo = max(valid_scores, key=valid_scores.get)
            print(f"\n最佳聚类算法：{best_algo}（得分：{self.evaluation_[best_algo]:.4f}）")
        else:
            print("\n警告：所有算法评估失败")

        return self

    def predict(self, X: pd.DataFrame, algorithm: list or str = None):
        """
        使用训练好的模型进行预测，支持单个算法或算法列表

        参数：
        X : 需要预测的数据，DataFrame格式
        algorithm :
            - None: 使用评估得分最高的算法（最佳算法）
            - str: 单个算法名称
            - list: 算法名称列表，如 ['GMM', 'KMeans']

        返回：
        DataFrame : 包含预测结果的DataFrame（每列对应一个算法）
        """
        if not self.models_:
            raise RuntimeError("模型尚未训练，请先调用fit方法")

        # 确定要使用的算法列表
        if algorithm is None:
            # 默认使用最佳算法
            best_algo = max(self.evaluation_, key=self.evaluation_.get)
            algorithm_list = [best_algo]
            print(f"使用最佳算法进行预测: {best_algo}")
        elif isinstance(algorithm, str):
            # 单个算法
            algorithm_list = [algorithm]
        elif isinstance(algorithm, list):
            # 算法列表
            algorithm_list = algorithm
        else:
            raise TypeError("algorithm参数必须是字符串、字符串列表或None")

        # 验证算法名称是否有效
        for algo in algorithm_list:
            if algo not in self.models_:
                raise ValueError(f"未找到算法 '{algo}'，可用的算法有: {list(self.models_.keys())}")

        # 数据预处理（使用保存的scaler）
        if self.scaler_:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X.values

        # 准备结果DataFrame
        results = pd.DataFrame(index=X.index)

        # 对每个算法进行预测
        for algo in algorithm_list:
            model = self.models_[algo]

            try:
                # 根据算法类型进行预测
                if algo == 'GMM':
                    labels = model.predict(X_scaled)
                # ============== 添加SOM预测处理 ==============
                elif algo == 'SOM':
                    # 获取每个样本的获胜神经元位置
                    winners = np.array([model.winner(x) for x in X_scaled])

                    # 将位置转换为簇标签
                    win_map = winners[:, 0] * model._weights.shape[1] + winners[:, 1]
                    labels = win_map.astype(int)

                    results[f'{algo}'] = labels
                # ===========================================

                elif algo in ['DBSCAN', 'Spectral', 'Hierarchical']:
                    # 这些算法不支持直接预测，需要训练集作为参考
                    labels = model.fit_predict(X_scaled)  # 直接在新数据上训练预测
                    print(f"[INFO] 算法 {algo} 需要使用直接训练模式预测，非原始训练模型")
                else:
                    labels = model.predict(X_scaled)

                results[f'{algo}'] = labels

            except Exception as e:
                print(f"[ERROR] {algo} 预测失败: {str(e)}")
                results[f'{algo}'] = np.nan

        return results


    def get_results(self, X: pd.DataFrame = None) -> pd.DataFrame:
        """
        获取聚类结果
        参数：
        X : 可选的新数据，如果未提供则返回训练数据的聚类结果

        返回：
        DataFrame : N*G的聚类结果DataFrame
        """
        if X is not None:
            # 对新数据进行预测
            results = pd.DataFrame(index=X.index)

            for algo, model in self.models_.items():
                try:
                    # 预处理数据
                    if self.scaler_:
                        X_scaled = self.scaler_.transform(X)
                    else:
                        X_scaled = X.values

                    # 预测
                    if algo == 'GMM':
                        labels = model.predict(X_scaled)
                    elif algo == 'DBSCAN' or algo == 'Spectral' or algo == 'Hierarchical':
                        labels = model.fit_predict(X_scaled)  # 直接在新数据上训练预测
                    else:
                        labels = model.predict(X_scaled)

                    results[f'{algo}'] = labels
                except Exception as e:
                    print(f"[WARNING] {algo} 预测失败: {str(e)}")
                    results[f'{algo}'] = np.nan

            return results
        else:
            # 返回训练数据的聚类结果
            if not hasattr(self, 'results_'):
                raise RuntimeError("模型尚未训练，请先调用fit方法")

            return self.results_


    def save_model(self, save_path: str):
        """
        保存整个聚类管道（包括scaler和模型）
        参数：
        save_path : 模型保存路径
        """
        if not save_path.endswith('.joblib'):
            save_path += '.joblib'

        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 保存整个对象
        save_data = {
            'models': self.models_,
            'scaler': self.scaler_,
            'evaluation': self.evaluation_,
            'cluster_num': self.cluster_num
        }

        joblib.dump(save_data, save_path)
        print(f"模型成功保存到: {save_path}")

    def load_model(self, load_path: str):
        """
        加载整个聚类管道（包括scaler和模型）

        参数：
        load_path : 模型加载路径
        """
        if not load_path.endswith('.joblib'):
            load_path += '.joblib'

        try:
            # 加载数据
            save_data = joblib.load(load_path)

            # 恢复对象状态
            self.models_ = save_data['models']
            self.scaler_ = save_data['scaler']
            self.evaluation_ = save_data['evaluation']
            self.cluster_num = save_data.get('cluster_num', 10)

            # 恢复Scaler状态
            if self.scaler_:
                self.scaler = StandardScaler()
                self.scaler.mean_ = self.scaler_.mean_
                self.scaler.scale_ = self.scaler_.scale_
            else:
                self.scaler = None

            print(f"模型成功从 {load_path} 加载")

        except FileNotFoundError:
            raise FileNotFoundError(f"未找到模型文件: {load_path}")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {str(e)}")

    def get_evaluation_results(self):
        """ 获取所有算法的评估指标 """
        if not hasattr(self, 'evaluation_'):
            raise RuntimeError("模型尚未训练，请先调用fit方法")

        return self.evaluation_




# 生成示例数据（与用户代码兼容）
def generate_data(n_samples=1000, n_features=5, n_clusters=5):
    X, _ = make_blobs(n_samples=n_samples,
                      n_features=n_features,
                      centers=n_clusters,
                      cluster_std=1.5,
                      random_state=42)
    return pd.DataFrame(X, columns=[f"Feature_{i + 1}" for i in range(n_features)])


# # 无监督聚类接口使用，流程演示
# if __name__ == "__main__":
#     # 1. 生成数据
#     df = generate_data()
#
#     # 2. 初始化接口
#     pipeline = ClusteringPipeline()
#
#     # 3. 执行聚类
#     pipeline.fit(df)
#
#     # 4. 获取结果
#     results = pipeline.get_results()
#
#     # 5. 结果验证
#     print("聚类结果维度:", results.shape)
#     print("\n各算法聚类分布:")
#     print(results.apply(pd.Series.value_counts))


# 无监督结果评价参数DBI计算，只有DBI参数需要手动计算，其他的SI、DVI、CH参数，均存在设计好的接口
def calculate_dunn_index(X, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan

    # 计算簇内最大直径
    intra_dists = [np.max(pdist(X[labels == i])) if np.sum(labels == i) > 1 else 0
                   for i in unique_labels]

    # 计算簇间最小距离
    inter_dists = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            mask_i = (labels == unique_labels[i])
            mask_j = (labels == unique_labels[j])
            dist = np.min(cdist(X[mask_i], X[mask_j]))
            inter_dists.append(dist)

    return np.min(inter_dists) / np.max(intra_dists) if inter_dists else np.nan



def evaluate_clustering(X, cluster_df):
    """
    :param X:输入数据的输入属性，X1，X2.....Xn
    :param cluster_df: 分类列，Y1，Y2.....Yn
    :return: Y1，Y2.....Yn对应的无监督聚类结果评价标准Silhouette、CH、DBI、DVI
    """
    results = pd.DataFrame(index=cluster_df.columns,
                           columns=['Silhouette', 'CH', 'DBI', 'DVI', 'UIndex'])

    for algo in cluster_df.columns:
        labels = cluster_df[algo].values

        try:
            # 过滤噪声标签（适用于DBSCAN）
            valid_mask = (labels != -1) if 'DBSCAN' in algo else slice(None)
            X_valid = X[valid_mask]
            labels_valid = labels[valid_mask]

            if len(np.unique(labels_valid)) < 2:
                raise ValueError("有效簇数不足")

            # 计算指标
            results.loc[algo, 'Silhouette'] = silhouette_score(X_valid, labels_valid)
            results.loc[algo, 'CH'] = calinski_harabasz_score(X_valid, labels_valid)
            results.loc[algo, 'DBI'] = davies_bouldin_score(X_valid, labels_valid)
            results.loc[algo, 'DVI'] = calculate_dunn_index(X_valid, labels_valid)
            # 避免除以零
            if results.loc[algo, 'DBI'] <= 0:
                results.loc[algo, 'UIndex'] = float('nan')
            else:
                # 修正公式：使用科学运算（** 表示幂运算，* 表示乘法）
                # results.loc[algo, 'UIndex'] = ((100**results.loc[algo, 'DVI'])*results.loc[algo, 'Silhouette'])/(results.loc[algo, 'DBI'])
                results.loc[algo, 'UIndex'] = 1/((0.2/results.loc[algo, 'Silhouette'])+(results.loc[algo, 'DBI']/2)+(0.02/results.loc[algo, 'DVI']))

        except Exception as e:
            print(f"{algo}评估失败：{str(e)}")
            results.loc[algo] = np.nan

    return results.sort_values('Silhouette', ascending=False)



# # 评估函数 evaluate_clustering 使用案例，计算'Silhouette', 'CH', 'DBI', 'DVI'四个参数
# if __name__ == "__main__":
#     # 加载数据（假设df包含原始特征+聚类结果）
#     # 特征列示例：df_features = df[['Feature_1', ..., 'Feature_M']]
#     # 聚类结果列示例：cluster_columns = ['Cluster_KMeans', ..., 'Cluster_GMM']
#
#     # 生成模拟数据
#     from sklearn.datasets import make_blobs
#
#     X, _ = make_blobs(n_samples=1000, n_features=5, centers=5, random_state=42)
#     X = StandardScaler().fit_transform(X)
#     # 假设cluster_df是包含各算法结果的DataFrame
#     cluster_df = pd.DataFrame({
#         'Cluster_KMeans': KMeans(n_clusters=5, random_state=42).fit_predict(X),
#         'Cluster_DBSCAN': DBSCAN(eps=0.5, min_samples=5).fit_predict(X),
#         'Cluster_Hierarchical': AgglomerativeClustering(n_clusters=5).fit_predict(X),
#         'Cluster_Spectral': SpectralClustering(n_clusters=5, random_state=42).fit_predict(X),
#         'Cluster_GMM': GaussianMixture(n_components=5, random_state=42).fit_predict(X)
#     })
#
#     print(cluster_df.shape, '\n', cluster_df.head(10))
#
#     # 执行评估
#     evaluation_results = evaluate_clustering(X, cluster_df)
#
#     # 结果展示
#     print("聚类算法性能对比：")
#     print(evaluation_results.round(3))



def cluster_result_mapping(true_labels, pred_labels):
    cm = contingency_matrix(pred_labels, true_labels)

    # replace_dict = {}
    # acc_dict = {}
    # acc_num_dict = {}
    # 这个是最基础的算法，看聚类结果，因为 聚类结果 多于 原始分类结果 ，把每一个 聚类结果袋 分别看一下能够放入 原始分类 的哪一个分类袋
    # 但这个方法应对不了 非均衡数据 只能应对 均衡数据，非均衡数据会全部分给那个数据量最大的分类
    # 并构造replace的词典，进行聚类结果相应的替换
    # for i in range(cm.shape[0]):
    #     max_num = np.max(cm[i, :])
    #     index_max = np.argmax(cm[i, :])
    #     acc_temp = max_num/np.sum(cm[i, :])
    #
    #     replace_dict[i] = index_max
    #     acc_dict[i] = acc_temp
    #     acc_num_dict[i] = max_num


    # 计算列和和行和
    sum_col = np.sum(cm, axis=0)
    sum_row = np.sum(cm, axis=1)
    # 避免除以零
    sum_col = np.where(sum_col == 0, 1, sum_col)
    sum_row = np.where(sum_row == 0, 1, sum_row)

    # 2. 按行归一化（每行和为1）- 反映每个预测类别的真实分布
    cm_row_normalized = cm / sum_row.reshape(-1, 1)
    # 1. 按列归一化（每列和为1）- 反映每个真实类别的预测分布
    cm_col_normalized = cm / sum_col.reshape(1, -1)

    replace_dict = {}
    acc_dict = {}
    acc_num_dict = {}
    # 先从 聚类结果出发 看一下聚类结果对应的 replace 字典，这个一般可能存在不太完善的地方
    for i in range(cm_row_normalized.shape[0]):
        acc_temp = np.max(cm_row_normalized[i, :])
        index_max = np.argmax(cm_row_normalized[i, :])
        max_num = cm[i, index_max]

        replace_dict[i] = index_max
        acc_dict[i] = acc_temp
        acc_num_dict[i] = max_num
    # 再从真实结果出发，确保每一个真实结果对应的类都存在至少一个预测的类与之相对应，当然，后面的类的保底预测类bagging可能会覆盖前面的类的保底bagging，这个要想办法进行后续的改进
    for i in range(cm_col_normalized.shape[1]):
        acc_temp = np.max(cm_col_normalized[:, i])
        index_max = np.argmax(cm_col_normalized[:, i])
        max_num = cm[index_max, i]

        replace_dict[index_max] = i
        acc_dict[index_max] = acc_temp
        acc_num_dict[index_max] = max_num
    # for i in range(cm_row_normalized.shape[0]):
    acc_dict[-1] = np.sum(list(acc_num_dict.values()))/np.sum(cm)

    return replace_dict, acc_dict, acc_num_dict


def evaluate_clustering_withlabel(true_labels, pred_labels):
    """使用标签数据，综合评估聚类质量"""
    # 处理噪声标签（如DBSCAN的-1标签），剔除无效数据
    clean_labels = pred_labels.copy()
    clean_labels[pred_labels == -1] = max(pred_labels) + 1
    replace_dict, acc_dict, acc_num_dict = cluster_result_mapping(true_labels, pred_labels)
    vector_func = np.vectorize(lambda x: replace_dict.get(x, x))
    pred_labels_replaced = vector_func(pred_labels).astype(np.int32)

    return pred_labels_replaced, acc_dict, acc_num_dict, replace_dict


# 根据标签信息评估无监督模型聚类效果
def evaluate_clustering_performance_with_label(df, true_col='Type_native'):
    """
    :param df: 为无监督分类结果加上原始分类标准结果，只能是分类结果，不能存在其他数据。
    :param true_col: 哪一列为真实分类标签的数据列
    :return:
    """
    # 编码真实标签,真实分类标签
    le = LabelEncoder()         # 一个用于将非数字型标签（如文字）转换为数字型标签的工具
    encoded_true = le.fit_transform(df[true_col])

    df_type_result_new = pd.DataFrame()
    acc_result = {}
    acc_num_result = {}
    replace_dict_dict = {}
    # 遍历每个聚类算法
    for algo in df.columns:
        # 跳过原始标签列
        if algo.__contains__(true_col):
            continue

        # 执行评估
        pred_labels_replaced, acc_dict, acc_num_dict, replace_dict = evaluate_clustering_withlabel(encoded_true, df[algo].values)
        df_type_result_new[algo] = pred_labels_replaced
        acc_result[algo] = acc_dict
        acc_num_result[algo] = acc_num_dict
        replace_dict_dict[algo] = replace_dict

    # 假设df1和df2行数相同
    df_type_result_new.index = df.index  # 直接赋值索引对象[6,7](@ref)

    return (df_type_result_new, pd.DataFrame(acc_result, columns=list(acc_result.keys())),
            pd.DataFrame(acc_num_result, columns=list(acc_num_result.keys())), replace_dict_dict)



def get_random_data_with_type(attributes = ['STAT_ENT', 'STAT_DIS', 'STAT_CON', 'STAT_XY_HOM', 'STAT_HOM', 'STAT_XY_CON', 'DYNA_DIS', 'STAT_ENG'], class_col = 'Type'):
    # 创建一个具有多个类别的分类数据集
    X, y = make_classification(
        n_samples=1000,  # 样本数量
        n_features=len(attributes),  # 特征数量（等于属性数量）
        n_informative=4,  # 信息特征数量
        n_redundant=2,  # 冗余特征数量
        n_repeated=1,  # 重复特征数量
        n_classes=4,  # 类别数量（4个类别）
        n_clusters_per_class=1,  # 每个类别的簇数
        flip_y=0.1,  # 噪声比例（标签随机翻转）
        class_sep=1.0,  # 类别之间的间隔大小
        random_state=42
    )

    # 将特征矩阵转换为DataFrame
    df_attributes = pd.DataFrame(X, columns=attributes)

    # 缩放特征到不同的范围，使每个属性有不同的统计特性
    df_attributes['STAT_ENT'] = df_attributes['STAT_ENT'] * 2 + 5
    df_attributes['STAT_DIS'] = np.log1p(np.abs(df_attributes['STAT_DIS'] * 0.5)) * 10
    df_attributes['STAT_CON'] = np.exp(df_attributes['STAT_CON'] * 0.5 + 1) + 0.5
    df_attributes['STAT_XY_HOM'] = np.round(df_attributes['STAT_XY_HOM'] * 50) / 10
    df_attributes['STAT_HOM'] = np.abs(df_attributes['STAT_HOM'] * 3.0 + 1.0)
    df_attributes['STAT_XY_CON'] = df_attributes['STAT_XY_CON'] * 10 + 15
    df_attributes['DYNA_DIS'] = np.sin(df_attributes['DYNA_DIS'] * 0.3) * 5 + 8
    df_attributes['STAT_ENG'] = df_attributes['STAT_ENG'] * 2.5 + 12.0

    # 添加分类列'Type'
    Type_T = np.random.choice([0, 1, 2, 3, 4], size=df_attributes.shape[0], p=[0.05, 0.1, 0.15, 0.3, 0.4])
    df_attributes[class_col] = Type_T
    return df_attributes

if __name__ == '__main__':
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    # 定义列名
    attributes = ['STAT_ENT', 'STAT_DIS', 'STAT_CON', 'STAT_XY_HOM', 'STAT_HOM', 'STAT_XY_CON', 'DYNA_DIS', 'STAT_ENG']
    class_col = 'Type'

    # 1.初始化数据，合并属性列和分类列
    df = get_random_data_with_type(attributes, class_col)

    # 显示数据摘要
    print("数据摘要:")
    print(df.head())

    # 2. 初始化接口
    Unsupervised_Pipeline = ClusteringPipeline(cluster_num=8, scale_data=True)
    # 3. 模型训练
    Unsupervised_Pipeline.fit(df[attributes])

    # # 获取结果
    # results = Unsupervised_Pipeline.get_results()
    # print('无监督聚类结果：')
    # print(results.describe())

    pred_result = Unsupervised_Pipeline.predict(df[attributes], algorithm=['KMeans', 'DBSCAN', 'Hierarchical', 'Spectral', 'GMM', 'SOM'])
    print(pred_result.describe())

    # 无监督聚类的评价标准Silhouette、CH、DBI、DVI计算
    unsupervised_evluate_result = evaluate_clustering(df[attributes], pred_result)
    print(unsupervised_evluate_result)

    df_result_all = pd.concat([df[class_col], pred_result], axis=1)
    df_type_result_new, acc_df, acc_num_df, replace_dict_dict = evaluate_clustering_performance_with_label(df_result_all, true_col=class_col)
    print(df_type_result_new.describe())

    print(acc_df)

    print(acc_num_df)

    print(replace_dict_dict)