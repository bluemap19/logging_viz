import logging
import math
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from scipy import ndimage
from src_logging.curve_preprocess import get_resolution_by_depth
from src_plot.well_logging_viz.cache_logging import EnhancedWellLogCache, CacheConfig
from src_well_data.data_logging_well import DATA_WELL

logger = logging.getLogger(__name__)

# 负责测井相关数据的管理（都需要绘制哪些数据，常规、成像、核磁、分类）、检查、访问、压缩、缓存、以及返回给可视化模块单元
class LoggingDataManager:
    # ========== 性能配置 ==========
    PERFORMANCE_CONFIG = {
        'cache_enabled': True,
        'max_cache_size': 500,
        'compression_level': 1,
        'fmi_cache_size': 50,
        'nmr_cache_size': 100,
    }

    def __init__(self, logging_data:pd.DataFrame=None,  # 常规测井数据
                    fmi_data:dict=None,                 # FMI测井数据
                    nmr_data:dict=None                  # NMR测井数据
        ):
        self.depth_min: float = np.inf      # 数据最小深度
        self.depth_max: float = -np.inf     # 数据最大深度

        # 初始化缓存系统
        self.cache_system = EnhancedWellLogCache(
            CacheConfig(
                enabled=self.PERFORMANCE_CONFIG['cache_enabled'],
                max_size=self.PERFORMANCE_CONFIG['max_cache_size'],
                fmi_max_size=self.PERFORMANCE_CONFIG['fmi_cache_size'],
                compression_level=self.PERFORMANCE_CONFIG['compression_level'],
                nmr_max_size=self.PERFORMANCE_CONFIG['nmr_cache_size']
            )
        )
        self.resolution = 0.1                           # 分辨率初始化

        self._validate_logging_data(logging_data)
        self.logging_data = logging_data                # 测井数据输入配置
        self.depth_col = self.logging_data.columns[0]   # 深度列配置
        self.config_logging = {}                        # 测井数据绘图配置
        self.config_type = {}                           # 分类数据绘图配置

        self._validate_fmi_data(fmi_data)               # 验证电成像数据
        self.fmi_data = fmi_data                        # 电成像数据初始化
        self.config_fmi = {}                            # 电成像数据绘图配置

        self._validate_nmr_data(nmr_data)               # 验证NMR谱类数据
        self.nmr_data = nmr_data                        # nmr数据初始化
        self.config_nmr = {}                            # nmr核磁数据绘制设置

        self._get_depth_limits()                        # 计算深度上下限


    def _check_curves_config(self, config_logging_data:Dict[str, Any]={}) -> None:
        """
        验证配置列的有效性，主要用在画图函数中，看看那些要画的列是否存在
        验证所有指定列是否存在于数据中
        """
        # 0. 验证数据是否存在
        if self.logging_data is None or self.logging_data.empty:
            logger.info("无 logging 数据， 跳过验证")
            return None

        # 1. 检查数据类型是否正确
        if not isinstance(self.logging_data, pd.DataFrame):
            raise ValueError("logging_data 参数必须是pandas DataFrame类型")

        # 修复这里：检查 config_logging_data 的类型
        if (config_logging_data is not None) and (not isinstance(config_logging_data, dict)):
            raise ValueError("config_logging_data 参数必须是 Dict 类型")

        if (config_logging_data is None) or len(config_logging_data) == 0:
            config_logging_data['depth_col'] = self.logging_data.columns.to_list()[0]
            config_logging_data['curve_cols'] = self.logging_data.columns.to_list()[1:-1]
            config_logging_data['type_cols'] = [self.logging_data.columns.to_list()[-1]]
            depth_col = config_logging_data['depth_col']
            curve_cols = config_logging_data['curve_cols']
        else:
            # 2. 验证深度列参数
            if 'depth_col' not in config_logging_data.keys():
                raise ValueError("config_logging_data必须存在非空字符串depth_col，限制深度列")
            else:
                config_logging_data['depth_col'] = self.logging_data.columns[0]

            depth_col = config_logging_data['depth_col']

            # 3. 验证曲线列参数
            if 'curve_cols' in config_logging_data.keys():
                curve_cols = config_logging_data['curve_cols']
                if not curve_cols or not isinstance(curve_cols, list):
                    raise ValueError("curve_cols必须是非空列表，限制绘制哪些列")
            else:
                config_logging_data['curve_cols'] = self.logging_data.columns.to_list()[1:-1]
                curve_cols = config_logging_data['curve_cols']

            # 4. 处理可选的分类列参数
            if 'type_cols' in config_logging_data.keys():
                type_cols = config_logging_data['type_cols']
                if not isinstance(type_cols, list):
                    raise ValueError("type_cols必须是列表或None")

        # 修改：展平曲线列结构以检查所有必要列
        required_cols = [depth_col]
        flat_curve_cols = []
        for item in curve_cols:
            if isinstance(item, list):
                # 嵌套列表：多曲线道
                flat_curve_cols.extend(item)
                required_cols.extend(item)
            elif isinstance(item, str):
                # 单曲线
                flat_curve_cols.append(item)
                required_cols.append(item)
            elif item is None:
                pass

        # 检查列名存在性 检查所有指定列是否存在于数据中
        missing_cols = set(required_cols) - set(self.logging_data.columns)
        if missing_cols:
            raise ValueError(f"数据中缺少以下必要列: {missing_cols}")

        # 6. 检查数据基本完整性
        if self.logging_data.empty:
            raise ValueError("输入数据不能为空DataFrame")

        if self.logging_data[depth_col].isna().all():
            raise ValueError("深度列数据全部为空")

        logger.info("输入参数验证通过，曲线结构: %s", curve_cols)

    def _validate_logging_data(self, logging_data:pd.DataFrame=None) -> None:
        """
        验证输入参数的有效性 - 确保数据格式正确且必要列存在

        参数验证流程：
        1. 检查数据类型是否正确
        2. 检查必要参数是否为空
        3. 验证所有指定列是否存在于数据中
        4. 检查数据基本完整性（非空、深度列有效）
        """
        # 0. 验证数据是否存在
        if logging_data is None or logging_data.empty:
            logger.info("无 logging 数据")
            return None

        # 1. 检查数据类型是否正确
        if not isinstance(logging_data, pd.DataFrame):
            raise ValueError("logging_data 参数必须是pandas DataFrame类型")

        # 3. 检查是否严格递增
        if not logging_data.columns[0].upper().__contains__("DEPTH"):
            logger.warning('logging_data 第一列不为深度列，请注意，列名为{}'.format(logging_data.columns[0]))

        # 4. 检查深度间隔是否合理（可选，但推荐）
        depth_diff = logging_data[logging_data.columns[0]].diff().dropna()
        if len(depth_diff) > 0:
            avg_interval = depth_diff.mean()
            std_interval = depth_diff.std()

            # 记录深度间隔统计信息
            logger.info(f"深度间隔统计: 平均={avg_interval:.4f}, 标准差={std_interval:.4f}")

            # 检查是否存在异常间隔（可选）
            unusual_intervals = depth_diff[(depth_diff > avg_interval + (3 * std_interval+0.0001))]
            # 检查是否存在错误间隔（可选）
            mistake_intervals = depth_diff[(depth_diff <= 0)]
            if len(unusual_intervals) > 0 or len(mistake_intervals) > 0:
                logger.warning(f"发现 {len(unusual_intervals)} 个异常深度间隔, {len(mistake_intervals)} 个错误深度间隔")

        self.resolution = get_resolution_by_depth(logging_data[logging_data.columns[0]].dropna().values)

    def _validate_fmi_data(self, fmi_dict: Optional[Dict[str, Any]]) -> None:
        """验证FMI图像数据的结构和完整性 - 增强验证"""
        if fmi_dict is None:
            logger.info("无FMI数据")
            return

        # 检查必要键是否存在
        required_keys = ['depth', 'image_data']
        for key in required_keys:
            if key not in fmi_dict:
                raise ValueError(f"FMI字典缺少必要键: {key}")

        # 验证深度数据
        depth_data = fmi_dict['depth']
        if depth_data is None:
            raise ValueError("FMI深度数据不能为None")

        # 确保深度数据是一维数组
        if depth_data.ndim != 1:
            logger.warning(f"FMI深度数据必须是一维数组，已经自动转换了")
            fmi_dict["depth"] = depth_data.ravel()

        # 验证图像数据
        if not isinstance(fmi_dict['image_data'], list) or len(fmi_dict['image_data']) == 0:
            raise ValueError("FMI图像数据必须是非空列表")

        # 验证每个图像数据的格式和维度匹配
        for i, image_data in enumerate(fmi_dict['image_data']):
            if not isinstance(image_data, np.ndarray):
                raise ValueError(f"FMI图像数据[{i}]必须是numpy数组")

            # 检查图像维度
            if image_data.ndim not in [2, 3]:
                raise ValueError(f"FMI图像数据[{i}]必须是2D或3D数组，实际维度: {image_data.ndim}")

            # 检查深度维度匹配
            if image_data.shape[0] != len(depth_data):
                logger.warning(f"FMI图像数据[{i}]深度维度不匹配: 图像{image_data.shape[0]} != 深度{len(depth_data)}")

        # 自动生成标题（如果未提供）
        if 'title' not in fmi_dict or fmi_dict['title'] is None:
            fmi_dict['title'] = [f'FMI_{i + 1}' for i in range(len(fmi_dict['image_data']))]

        logger.info("FMI数据验证通过，包含%d个图像", len(fmi_dict['image_data']))

    def _validate_nmr_data(self, nmr_dict: Optional[Dict[str, Any]]=None) -> None:
        """验证新的NMR数据格式"""
        if nmr_dict is None:
            logger.info("无NMR数据")
            return

        if not isinstance(nmr_dict, dict):
            raise ValueError("NMR_dict必须是字典类型")

        # 检查必要键
        required_keys = ['depth', 'nmr_data']
        for key in required_keys:
            if key not in nmr_dict:
                raise ValueError(f"NMR字典缺少必要键: {key}")

        # 验证深度数据
        depth_data = nmr_dict['depth']
        if depth_data is None:
            raise ValueError("NMR深度数据不能为None")
        
        if depth_data.ndim != 1:
            logger.warning("NMR深度数据必须是一维数组，已自动转换")
            nmr_dict["depth"] = depth_data.ravel()

        # 验证nmr_data
        if not isinstance(nmr_dict['nmr_data'], list) or len(nmr_dict['nmr_data']) == 0:
            raise ValueError("NMR数据必须是非空列表")

        # 验证每个NMR数据道
        for i, nmr_array in enumerate(nmr_dict['nmr_data']):
            if not isinstance(nmr_array, np.ndarray):
                raise ValueError(f"NMR数据[{i}]必须是numpy数组")
            if nmr_array.ndim != 2:
                raise ValueError(f"NMR数据[{i}]必须是2D数组")
            if nmr_array.shape[0] != len(depth_data):
                logger.warning(f"NMR数据[{i}]深度维度不匹配: 数据{nmr_array.shape[0]} != 深度{len(depth_data)}")

        # 自动生成标题（如果未提供）
        if 'title' not in nmr_dict or nmr_dict['title'] is None:
            nmr_dict['title'] = [f'NMR_{i + 1}' for i in range(len(nmr_dict['nmr_data']))]

        logger.info("NMR数据验证通过，包含%d个数据道", len(nmr_dict['nmr_data']))

    def _get_depth_limits(self) -> [float, float]:
        """
        设置深度显示范围，根据配置过滤数据

        处理逻辑：
        1. 如果没有配置限制，使用完整数据范围
        2. 如果有配置，确保配置有效且与数据范围有重叠
        3. 根据最终范围过滤数据
        """
        # 初始化深度范围
        full_depth_min = float('inf')
        full_depth_max = float('-inf')

        # 检查各类数据并更新深度范围
        if self.logging_data is not None and not self.logging_data.empty:
            full_depth_min = min(full_depth_min, self.logging_data[self.logging_data.columns[0]].min())
            full_depth_max = max(full_depth_max, self.logging_data[self.logging_data.columns[0]].max())

        if self.fmi_data is not None and 'depth' in self.fmi_data:
            full_depth_min = min(full_depth_min, self.fmi_data['depth'].min())
            full_depth_max = max(full_depth_max, self.fmi_data['depth'].max())

        if self.nmr_data is not None:
            full_depth_min = min(full_depth_min, self.nmr_data['depth'].min())
            full_depth_max = max(full_depth_max, self.nmr_data['depth'].max())

        # 如果没有数据，使用默认范围
        if full_depth_min == float('inf'):
            full_depth_min = 0.0
            full_depth_max = 100.0
            logger.warning("未找到有效数据，使用默认深度范围: 0-100m")

        # 无配置时使用完整数据范围
        self.depth_min = full_depth_min
        self.depth_max = full_depth_max

        return self.depth_min, self.depth_max

    def _generate_cache_key(self, depth_range: Tuple[float, float], data_type: str = 'data') -> str:
        """生成精确的缓存键"""
        # 使用更高精度避免冲突
        if data_type == 'fmi':
            # FMI 缓存键生成
            return f"fmi_{min(depth_range):.4f}_{max(depth_range):.4f}"
        elif data_type == 'nmr':
            # NMR 缓存键生成
            return f"nmr_{min(depth_range):.4f}_{max(depth_range):.4f}"
        else:
            # 常规测井 缓存键生成
            return f"data_{min(depth_range):.4f}_{max(depth_range):.4f}"

    def get_visible_logging_data(self, top_depth: float, bottom_depth: float) -> Optional[pd.DataFrame]:
        """获取可见范围内的常规测井数据（带缓存）"""
        if self.logging_data is None or self.logging_data.empty:
            return None

        depth_range = (top_depth, bottom_depth)

        # 尝试从缓存获取
        cached_data = self.cache_system.get_logging_data(depth_range)
        if cached_data is not None:
            logger.debug(f"常规测井缓存命中: {depth_range}")
            return cached_data

        # 缓存未命中，从原始数据查询
        logger.debug(f"常规测井缓存未命中，查询数据: {depth_range}")
        depth_col = self.logging_data.columns[0]
        mask = (self.logging_data[depth_col] >= top_depth) & (self.logging_data[depth_col] <= bottom_depth)
        visible_data = self.logging_data[mask].copy()

        if not visible_data.empty:
            # 设置缓存
            self.cache_system.set_logging_data(depth_range, visible_data)
            logger.debug(f"常规测井缓存设置: {depth_range}, 数据点{len(visible_data)}")

        return visible_data

    def _compress_fmi_vertically(self, image_data: np.ndarray, target_points: int) -> np.ndarray:
        """
        对FMI图像数据进行垂直方向压缩
        参数:
        - image_data: 原始图像数据，形状为 (depth_points, width) 或 (depth_points, height, width)
        - target_points: 目标深度点数
        返回:
        - 压缩后的图像数据
        """
        if image_data.ndim not in [2, 3]:
            raise ValueError(f"不支持的图像维度: {image_data.ndim}，期望2D或3D数组")

        original_points = image_data.shape[0]

        if original_points <= target_points:
            # 无需压缩
            return image_data

        # 计算压缩比例
        zoom_factors = [target_points / original_points]

        if image_data.ndim == 2:
            # 2D图像: (depth, width)
            zoom_factors.append(1.0)  # 宽度方向不压缩
        else:
            # 3D图像: (depth, height, width)
            zoom_factors.extend([1.0, 1.0])  # 高度和宽度方向不压缩

        # 使用scipy的zoom函数进行高质量缩放
        compressed = ndimage.zoom(image_data, zoom_factors, order=1)  # 双线性插值
        return compressed

    def _compress_depths(self, depths: np.ndarray, target_points: int) -> np.ndarray:
        """
        压缩深度数组，保持深度范围的线性关系
        """
        if len(depths) <= target_points:
            return depths

        # 创建等间距的深度点
        return np.linspace(depths[0], depths[-1], target_points)

    def get_visible_fmi_data(self, top_depth: float, bottom_depth: float, max_vertical_points: int=400) -> Optional[Dict[str, Any]]:
        """获取可见范围内的FMI数据（带缓存）"""
        if self.fmi_data is None:
            return None

        depth_range = (top_depth, bottom_depth)

        # 尝试从缓存获取
        cached_fmi = self.cache_system.get_fmi_data(depth_range)
        if cached_fmi is not None:
            logger.debug(f"FMI缓存命中: {depth_range}")
            depth_temp = self._get_fmi_depth_slice(top_depth, bottom_depth)
            max_point = cached_fmi[0].shape[0]
            return {
                'depth': self._compress_depths(depth_temp, max_point),
                'image_data': cached_fmi,
                'title': self.fmi_data.get('title', [f'FMI_{i + 1}' for i in range(len(cached_fmi))])
            }

        # 缓存未命中，处理FMI数据
        logger.debug(f"FMI缓存未命中，处理数据: {depth_range}")
        depth_mask = (self.fmi_data['depth'] >= top_depth) & (self.fmi_data['depth'] <= bottom_depth)
        visible_depths = self.fmi_data['depth'][depth_mask]

        if len(visible_depths) == 0:
            return None

        # 提取对应的图像数据切片
        visible_images = []
        max_point = 0
        for image_data in self.fmi_data['image_data']:
            if image_data.shape[0] == len(self.fmi_data['depth']):
                visible_image = image_data[depth_mask]
                compressed_image = self._compress_fmi_vertically(visible_image, max_vertical_points)
                max_point = compressed_image.shape[0]
                visible_images.append(compressed_image)
            else:
                logger.warning("FMI图像深度维度不匹配，跳过该图像")

        if visible_images:
            # 设置缓存
            self.cache_system.set_fmi_data(depth_range, visible_images)
            logger.debug(f"FMI缓存设置: {depth_range}, 图像数{len(visible_images)}")

            return {
                'depth': self._compress_depths(visible_depths, max_point),
                'image_data': visible_images,
                'title': self.fmi_data['title']
            }

        return None

    def get_visible_nmr_data(self, top_depth: float, bottom_depth: float, spectral_density: float=5) -> Optional[Dict[str, Any]]:
        """获取可见范围内的NMR数据（带缓存）"""
        if self.nmr_data is None:
            return None

        depth_range = (top_depth, bottom_depth)

        # 尝试从缓存获取
        cached_nmr = self.cache_system.get_nmr_data(depth_range)
        if cached_nmr is not None:
            logger.debug(f"NMR缓存命中: {depth_range}")
            depth_temp = self._get_nmr_depth_slice(top_depth, bottom_depth)
            max_point = cached_nmr[0].shape[0]
            return {
                'depth': self._compress_depths(depth_temp, max_point),
                'nmr_data': cached_nmr,
                'title': self.nmr_data.get('title', [f'NMR_{i + 1}' for i in range(len(cached_nmr))])
            }

        # 缓存未命中，处理NMR数据
        logger.debug(f"NMR缓存未命中，处理数据: {depth_range}")
        depth_mask = (self.nmr_data['depth'] >= top_depth) & (self.nmr_data['depth'] <= bottom_depth)
        visible_depths = self.nmr_data['depth'][depth_mask]

        if len(visible_depths) == 0:
            return None

        # 提取对应的NMR数据切片
        visible_nmr_arrays = []
        spectral_num = math.ceil((max(visible_depths) - min(visible_depths))*spectral_density)
        max_point = 0
        for nmr_array in self.nmr_data['nmr_data']:
            if nmr_array.shape[0] == len(self.nmr_data['depth']):
                visible_nmr = nmr_array[depth_mask]
                compressed_image = self._compress_fmi_vertically(visible_nmr, spectral_num)
                max_point = compressed_image.shape[0]
                visible_nmr_arrays.append(compressed_image)
            else:
                logger.warning("NMR数据深度维度不匹配，跳过该数据道")

        if visible_nmr_arrays:
            # 设置缓存
            self.cache_system.set_nmr_data(depth_range, visible_nmr_arrays)
            logger.debug(f"NMR缓存设置: {depth_range}, 数据道数{len(visible_nmr_arrays)}")

            return {
                'depth': self._compress_depths(visible_depths, max_point),
                'nmr_data': visible_nmr_arrays,
                'title': self.nmr_data['title']
            }

        return None

    def _get_fmi_depth_slice(self, top_depth: float, bottom_depth: float) -> np.ndarray:
        """获取FMI深度切片"""
        if self.fmi_data is None:
            return np.array([])
        depth_mask = (self.fmi_data['depth'] >= top_depth) & (self.fmi_data['depth'] <= bottom_depth)
        return self.fmi_data['depth'][depth_mask]

    def _get_nmr_depth_slice(self, top_depth: float, bottom_depth: float) -> np.ndarray:
        """获取NMR深度切片"""
        if self.nmr_data is None:
            return np.array([])
        depth_mask = (self.nmr_data['depth'] >= top_depth) & (self.nmr_data['depth'] <= bottom_depth)
        return self.nmr_data['depth'][depth_mask]

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.cache_system.get_cache_stats()

    def clear_cache(self):
        """清空缓存"""
        self.cache_system.clear_cache()
        logger.info("缓存已清空")

    def close(self):
        """关闭数据管理器"""
        self.clear_cache()
        logger.info("数据管理器已关闭")


    def plot_config_check(self,
                          config_logging: Dict[str, Any] = None,
                          config_fmi: Dict[str, Any] = None,
                          config_nmr: Dict[str, Any] = None,
                          config_type: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
        """
        绘图配置智能检查与自动补全系统

        功能概述：
        1. 配置验证：检查必填字段和格式正确性
        2. 智能补全：缺失配置自动填充合理默认值
        3. 数据适配：根据实际数据调整配置参数
        4. 一致性检查：确保配置与数据匹配

        返回：
        Dict: 包含四个完整配置字典的集合
            {
                'logging': 完整的常规测井配置,
                'fmi': 完整的FMI配置,
                'nmr': 完整的NMR配置,
                'type': 完整的分类配置
            }
        """

        # 初始化默认配置模板
        default_configs = self._get_default_plot_configs()

        # 深度合并用户配置和默认配置
        config_logging = self._deep_merge_configs(default_configs['logging'], config_logging or {})
        config_fmi = self._deep_merge_configs(default_configs['fmi'], config_fmi or {})
        config_nmr = self._deep_merge_configs(default_configs['nmr'], config_nmr or {})
        config_type = self._deep_merge_configs(default_configs['type'], config_type or {})

        # 执行配置验证和适配
        validated_configs = {
            'logging': self._validate_and_adapt_logging_config(config_logging),
            'fmi': self._validate_and_adapt_fmi_config(config_fmi),
            'nmr': self._validate_and_adapt_nmr_config(config_nmr),
            'type': self._validate_and_adapt_type_config(config_type)
        }

        self.config_logging = config_logging
        self.config_fmi = config_fmi
        self.config_nmr = config_nmr
        self.config_type = config_type

        # 记录配置检查结果
        self._log_config_check_results(validated_configs)

        return validated_configs

    def _get_default_plot_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的默认绘图配置模板"""
        return {
            # 测井数据绘图相关配置
            'logging': {
                'depth_col': '',        # 深度列列名
                'curves_plot': [],      # 待绘制的曲线列
                'colors_plot': [],      # 相对应的颜色配置
                'line_style_plot': [],  # 线型配置
                'line_width': [],       # 线宽配置
                'axis_config': []       # 坐标轴是否log配置
            },
            # 电成像绘图设置
            'fmi': {
                'color_map': 'rainbow', # fmi成像颜色色条配置
                'auto_scale':False,     # 是否自动进行图像的窗口缩放
                'scale_range': [0, 256],# 像素范围的上下限裁剪缩放后的数据范围
                'title_fmi': [],        # fmi电成像的title
                'pix_density_y':8000,   # y方向最多绘制多少像素点
                'pix_density_x':400,    # x方向最多绘制多少像素点
            },
            # 核磁绘图相关设置
            'nmr': {
                'plot_density': 30,                 # 核磁绘制的密度plot_density个/窗口
                'plot_amplitude_scaling': 0.5,      # 核磁谱的高度缩放设置
                'x_logarithmic_scale': True,       # 核磁谱的x方向是否进行log化
                'color_fill': 'green',              # 颜色填充配置
                'spectrum_config': {
                    'line_style': '-',              # 核磁谱折线的线型设置
                    'line_width': 0.8,              # 默认线宽设置
                    'fill_alpha': 0.5,              # 填充物透明度设置
                    'baseline_visible': True,       # 基线是否可见
                },
                'axis_config': {
                    'x_axis_label': 'T2 Time (ms)', # x标签配置
                    'show_grid': True,              # 是否绘制格子线
                    'log_ticks': [0.1, 1, 10, 100, 1000]    # x方向的指示标签配置
                }
            },
            # 分类结果相关设置
            'type': {
                'types_cols': [],               # 分类列名设置
                'colors_type': {},              # 分类的颜色配置
                'width_type': {},               # 分类的水平方向的宽度设置
                'legend_dict': {},              # 分类的指示legend设置
                'show_legend': True,            # 是否展示legend
                'display_config': {             # legend绘制设置
                    'legend_facecolor': 'white',
                    'legend_columns': 4,        # legend每行有几个分类指示
                    'border_visible': False,    # 指示的边界线是否绘制
                    'border_width': 0.5,        # 边界线的线宽配置
                    'use_patterns': False,
                    'legend_fontsize': 10,
                    'pattern_alpha': 0.3,       # legend的指示填充物的透明度
                    'hatch_patterns': ['/', '\\', '|', '-', '+', 'x', 'o', 'O']
                }
            }
        }

    def _deep_merge_configs(self, default_config: Dict[str, Any], user_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        深度合并配置字典 - 递归处理嵌套字典

        合并策略：
        - 用户配置优先，缺失项使用默认值
        - 支持无限层级嵌套配置
        - 保留字典中所有原始键值对
        """
        result = default_config.copy()

        for key, value in user_config.items():
            if key in result:
                if isinstance(value, dict) and isinstance(result[key], dict):
                    # 递归合并嵌套字典
                    result[key] = self._deep_merge_configs(result[key], value)
                else:
                    # 直接覆盖标量值或列表
                    result[key] = value
            else:
                # 添加用户自定义的新配置项
                result[key] = value

        return result

    def _validate_and_adapt_logging_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证和适配常规测井绘图配置"""
        # 0. 前置安全检查
        if self.logging_data is None or self.logging_data.empty:
            logger.warning("无测井数据可用，使用最小化配置")
            return self._get_default_plot_configs()['logging']

        # 1. 深度列配置检查和适配
        if not config.get('depth_col') or not isinstance(config['depth_col'], str) or len(config['depth_col'].strip()) == 0:
            if self.logging_data.columns[0].lower().__contains__('dep'):
                # 自动检测深度列（通常为第一列）
                config['depth_col'] = self.logging_data.columns[0]
                logger.info(f"自动设置深度列: {config['depth_col']}")
        else:
            if config['depth_col'] not in self.logging_data.columns:
                raise ValueError(f"{config['depth_col']} col not in logging_data")

        # 2. 曲线绘制配置智能处理
        curves_plot = config.get('curves_plot', [])
        # if not curves_plot or len(curves_plot) == 0:
        if isinstance(curves_plot, str) and curves_plot.lower().__contains__('auto'):
            # 自动生成曲线配置：使用数据中的所有数值列（除深度列外）
            numeric_cols = self.logging_data.select_dtypes(include=[np.number]).columns.tolist()
            # 排除深度列
            if config['depth_col'] in numeric_cols:
                numeric_cols.remove(config['depth_col'])
            config['curves_plot'] = numeric_cols
            logger.info(f"自动检测到 {len(numeric_cols)} 个数值曲线列:{config['curves_plot']}")
        elif len(curves_plot) > 0:
            logger.info(f"配置可用，使用默认曲线配置：{curves_plot}")
        else:
            logger.warning(f"测井数据绘图配置为空，需要注意一下，不进行绘制了")

        # 3. 验证曲线列是否存在
        valid_curves = []
        if self.logging_data is not None:
            available_columns = set(self.logging_data.columns)
            for curve_item in config['curves_plot']:
                if isinstance(curve_item, list):
                    # 多曲线道：检查所有子曲线
                    valid_sub_curves = [c for c in curve_item if c in available_columns]
                    if valid_sub_curves:
                        valid_curves.append(valid_sub_curves)
                    else:
                        logger.warning(f"多曲线道中无有效列: {curve_item}")
                elif isinstance(curve_item, str):
                    # 单曲线：检查是否存在
                    if curve_item in available_columns:
                        valid_curves.append(curve_item)
                    else:
                        logger.warning(f"曲线列不存在: {curve_item}")
                else:
                    logger.warning(f"无效的曲线配置项: {curve_item}")

        config['curves_plot'] = valid_curves

        # 4. 颜色配置智能分配
        if not config['colors_plot'] or len(config['colors_plot']) <= len(config['curves_plot']):
            config['colors_plot'] = self._generate_curve_colors(config['curves_plot'], config['colors_plot'])

        # 5. 线型配置验证
        if not config['line_style_plot'] or len(config['line_style_plot']) <= len(config['curves_plot']):
            config['line_style_plot'] = self._generate_line_styles(config['curves_plot'], config['line_style_plot'])
        if not config['line_width'] or len(config['line_width']) <= len(config['curves_plot']):
            config['line_width'] = self._generate_line_width(config['curves_plot'], config['line_width'])

        return config

    def _validate_and_adapt_fmi_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证和适配FMI成像配置"""

        # 1. 颜色映射验证
        valid_cmaps = ['rainbow', 'hot', 'cool', 'viridis', 'plasma', 'inferno', 'magma', 'cividis']
        if config.get('color_map') not in valid_cmaps:
            config['color_map'] = 'rainbow'
            logger.warning(f"颜色映射无效，使用默认值: rainbow")

        # 2. 缩放范围验证和适配
        scale_range = config.get('scale_range', [0, 256])
        if not isinstance(scale_range, list) or len(scale_range) != 2:
            config['scale_range'] = [0, 256]
            logger.warning("缩放范围格式错误，使用默认值 [0, 256]")
        else:
            # 确保最小值小于最大值
            config['scale_range'] = [min(scale_range), max(scale_range)]

        # 3. 标题配置适配
        if not config.get('title_fmi'):
            if self.fmi_data and 'image_data' in self.fmi_data:
                num_images = len(self.fmi_data['image_data'])
                config['title_fmi'] = [f'FMI_{i + 1}' for i in range(num_images)]
                logger.info(f"自动生成 {num_images} 个FMI面板标题")
            else:
                config['title_fmi'] = ['FMI_Image']
                logger.warning("无FMI数据，使用默认标题")

        return config

    def _validate_and_adapt_nmr_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证和适配NMR谱图配置"""

        # 1. 绘图密度验证
        plot_density = config.get('plot_density', 20)
        if not isinstance(plot_density, (int, float)) or plot_density <= 0:
            config['plot_density'] = 20
            logger.warning("绘图密度无效，使用默认值: 20点/米")

        # 2. 振幅缩放因子验证
        plot_amplitude_scaling = config.get('plot_amplitude_scaling', 1.5)
        if not isinstance(plot_amplitude_scaling, (int, float)) or plot_amplitude_scaling <= 0:
            config['plot_amplitude_scaling'] = 1.5
            logger.warning("振幅缩放因子无效，使用默认值: 1.5")

        # 3. 对数刻度配置
        if not isinstance(config.get('x_logarithmic_scale'), bool):
            config['x_logarithmic_scale'] = False

        # 4. 颜色填充配置
        if not config.get('color_fill') or not isinstance(config['color_fill'], str):
            config['color_fill'] = 'green'
            logger.warning("填充颜色无效，使用默认值: green")

        # 5. 幅度高度配置配置
        if not config.get('scaling_config') or not isinstance(config['color_fill'], int):
            config['scaling_config'] = 1.2

        return config

    def _validate_and_adapt_type_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证和适配岩性分类配置"""

        # 1. 分类列验证和适配
        types_cols = config.get('types_cols', [])
        if not types_cols:
            return config
        elif types_cols is None or (isinstance(types_cols, str) and types_cols.lower().__contains__('auto')):
            # 自动检测分类列：查找包含分类信息的列
            if self.logging_data is not None:
                # 寻找整数类型或分类类型的列
                potential_type_cols = []
                for col in self.logging_data.columns:
                    if col == config.get('depth_col', 'DEPTH'):
                        continue  # 跳过深度列

                    # 检查是否为分类数据：整数类型或有限唯一值
                    unique_vals = self.logging_data[col].dropna().unique()
                    if len(unique_vals) <= 6 or self.logging_data[col].dtype in ['int', 'int32', 'int64']:
                        potential_type_cols.append(col)

                config['types_cols'] = potential_type_cols[:2]  # 最多取前2个
                logger.info(f"自动检测到 {len(potential_type_cols)} 个潜在分类列，使用前 {len(config['types_cols'])} 个:{potential_type_cols}")
            else:
                config['types_cols'] = ['LITHOLOGY', 'FACIES']  # 默认分类列
                logger.warning("无数据可用，使用默认分类列配置")

        # 2. 颜色配置生成
        colors_type = config.get('colors_type', {})
        if not colors_type:
            config['colors_type'] = self._generate_type_colors(config['types_cols'])

        # 3. 宽度配置生成
        width_type = config.get('width_type', {})
        if not width_type:
            config['width_type'] = self._generate_type_widths(config['types_cols'])

        # 4. 图例配置生成
        legend_dict = config.get('legend_dict', {})
        if not legend_dict:
            config['legend_dict'] = self._generate_legend_dict(config['types_cols'])

        # 5. 验证分类列是否存在
        if self.logging_data is not None:
            available_columns = set(self.logging_data.columns)
            valid_types = [t for t in config['types_cols'] if t in available_columns]
            config['types_cols'] = valid_types
            if len(valid_types) < len(config['types_cols']):
                logger.warning(f"移除了 {len(config['types_cols']) - len(valid_types)} 个不存在的分类列")

        return config

    def _generate_curve_colors(self, curves_plot: List[Any]=[], colors_plot: List[Any]=[]) -> List[Any]:
        """为曲线配置智能生成颜色方案"""
        # 默认颜色序列
        default_colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF', '#FFA500', '#800080', '#008000', '#000080', '#808000']

        # 多曲线道颜色序列
        multi_curve_colors = ['#FF0000', '#000000', '#0000FF', '#800080', '#FFA500', '#FFFF00']

        # 如果曲线数量超过默认颜色，自动扩展颜色序列
        for i in range(len(curves_plot)):
            if len(colors_plot) < len(curves_plot):
                item = curves_plot[i]
                if isinstance(item, list):
                    colors_plot.append(multi_curve_colors[:len(item)])
                else:
                    colors_plot.append(default_colors[i%len(default_colors)])

        return colors_plot

    def _generate_line_styles(self, curves_plot: List[Any]=[], line_style_plot: List[Any]=[]) -> List[Any]:
        """验证线型配置的完整性"""
        # 默认线型配置序列
        default_line_styles = ['-', '--', '-.', ':']

        # 多曲线道线型配置序列
        multi_line_styles = ['-', '-.', '--', ':']

        # 如果曲线数量超过默认线型配置，自动扩展线型配置序列
        for i in range(len(curves_plot)):
            if len(line_style_plot) < len(curves_plot):
                item = curves_plot[i]
                if isinstance(item, list):
                    line_style_plot.append(multi_line_styles[:len(item)])
                else:
                    line_style_plot.append(default_line_styles[0])

        return line_style_plot

    def _generate_line_width(self, curves_plot: List[Any]=[], line_width: List[Any]=[]) -> List[Any]:
        """验证线型配置的完整性"""
        # 默认线型配置序列
        default_line_width = [1.0, 1.4, 1.8]

        # 多曲线道线型配置序列
        multi_line_width = [0.8, 1.0, 1.4]

        # 如果曲线数量超过默认线型配置，自动扩展线型配置序列
        for i in range(len(curves_plot)):
            if len(line_width) < len(curves_plot):
                item = curves_plot[i]
                if isinstance(item, list):
                    line_width.append(multi_line_width[:len(item)])
                else:
                    line_width.append(default_line_width[0])

        return line_width

    def _generate_type_colors(self, types_cols: List[str]) -> Dict[int, str]:
        """为分类数据生成颜色映射"""
        # 标准岩性颜色方案
        lithology_colors = {
            0: '#FF6B6B',  # 砂岩 - 红色
            1: '#4ECDC4',  # 泥岩 - 青色
            2: '#45B7D1',  # 页岩 - 蓝色
            3: '#96CEB4',  # 石灰岩 - 绿色
            4: '#FECA57',  # 白云岩 - 黄色
            5: '#FF9FF3',  # 煤层 - 粉色
            6: '#54A0FF',  # 盐岩 - 亮蓝
            7: '#5F27CD',  # 石膏 - 紫色
            8: '#00D2D3',  # 火成岩 - 青蓝
            9: '#FF9F43'  # 变质岩 - 橙色
        }

        # 如果数据可用，根据实际分类值调整颜色
        if self.logging_data is not None and types_cols:
            actual_types = set()
            for col in types_cols:
                if col in self.logging_data.columns:
                    unique_vals = self.logging_data[col].dropna().unique()
                    for val in unique_vals:
                        if not pd.isna(val) and val >= 0:
                            actual_types.add(int(val))

            # 为实际存在的类型分配颜色
            sorted_types = sorted(actual_types)
            color_map = {}
            for i, type_val in enumerate(sorted_types):
                color_map[type_val] = lithology_colors.get(i % len(lithology_colors), '#CCCCCC')

            return color_map

        return lithology_colors

    def _generate_type_widths(self, types_cols: List[str]) -> Dict[int, float]:
        """为分类数据生成显示宽度配置"""
        # 默认宽度配置：类型值越大，显示宽度越大
        if self.logging_data is not None and types_cols:
            actual_types = set()
            for col in types_cols:
                if col in self.logging_data.columns:
                    unique_vals = self.logging_data[col].dropna().unique()
                    for val in unique_vals:
                        if not pd.isna(val) and val >= 0:
                            actual_types.add(int(val))

            if actual_types:
                sorted_types = sorted(actual_types)
                width_map = {}
                n_types = len(sorted_types)
                for i, type_val in enumerate(sorted_types):
                    # 均匀分配宽度：第一种类型宽度=1/N，第二种=2/N，以此类推
                    width_map[type_val] = (i + 1) / n_types
                return width_map

        # 默认宽度配置
        return {
            0: 0.1,  # 砂岩 - 窄
            1: 0.2,  # 泥岩 - 中等
            2: 0.3,  # 页岩 - 宽
            3: 0.4,  # 石灰岩 - 更宽
            4: 0.5  # 其他 - 最宽
        }

    def _generate_legend_dict(self, types_cols: List[str]) -> Dict[int, str]:
        """为分类数据生成图例字典"""
        # 标准岩性图例
        standard_legend = {
            0: 'Sandstone',  # 砂岩
            1: 'Mudstone',  # 泥岩
            2: 'Shale',  # 页岩
            3: 'Limestone',  # 石灰岩
            4: 'Dolomite',  # 白云岩
            5: 'Coal',  # 煤层
            6: 'Salt',  # 盐岩
            7: 'Gypsum',  # 石膏
            8: 'Igneous',  # 火成岩
            9: 'Metamorphic'  # 变质岩
        }

        # 如果数据可用，根据实际分类值调整图例
        if self.logging_data is not None and types_cols:
            actual_types = set()
            for col in types_cols:
                if col in self.logging_data.columns:
                    unique_vals = self.logging_data[col].dropna().unique()
                    for val in unique_vals:
                        if not pd.isna(val) and val >= 0:
                            actual_types.add(int(val))

            # 为实际存在的类型生成图例
            legend_map = {}
            for type_val in sorted(actual_types):
                legend_map[type_val] = standard_legend.get(type_val, f'Type_{type_val}')

            return legend_map

        return standard_legend


    def _log_config_check_results(self, configs: Dict[str, Dict[str, Any]]):
        """记录配置检查的详细结果"""
        logger.info("=" * 60)
        logger.info("绘图配置检查完成")
        logger.info("=" * 60)

        # 常规测井配置统计
        logging_config = configs['logging']
        curves_count = len(logging_config['curves_plot'])
        logger.info(f"📊 常规测井配置: {curves_count} 个曲线道")

        # FMI配置统计
        fmi_config = configs['fmi']
        fmi_titles = len(fmi_config['title_fmi'])
        logger.info(f"🖼️ FMI成像配置: {fmi_titles} 个图像面板")

        # NMR配置统计
        nmr_config = configs['nmr']
        logger.info(f"🔬 NMR谱图配置: 密度={nmr_config['plot_density']}点/米, 振幅缩放={nmr_config['plot_amplitude_scaling']}x")

        # 分类配置统计
        type_config = configs['type']
        types_count = len(type_config['types_cols'])
        legend_count = len(type_config['legend_dict'])
        logger.info(f"🎨 岩性分类配置: {types_count} 个分类列, {legend_count} 个图例项")

        logger.info("所有配置验证通过，准备进行可视化")

    def get_logging_resolution(self):
        return self.resolution

    def cal_plot_num(self):
        """计算子图总数：FMI面板 + 曲线面板 + 分类面板 + NMR谱面板"""
        if self.config_fmi:
            self.n_curve_panels = len(self.config_logging['curves_plot']) if self.config_logging['curves_plot'] and (self.logging_data is not None) and (not self.logging_data.empty) else 0
        else:
            self.n_curve_panels = 0

        if self.config_type and (self.logging_data is not None) and ('types_cols' in self.config_type.keys()):
            self.n_type_panels = len(self.config_type['types_cols']) if self.config_type['types_cols'] and (self.logging_data is not None) and (not self.logging_data.empty) else 0
        else:
            self.n_type_panels = 0

        if self.config_fmi and (self.fmi_data is not None) and ('image_data' in self.fmi_data.keys()):
            self.n_fmi_panels = len(self.fmi_data['image_data']) if 'image_data' in self.fmi_data and self.fmi_data['image_data'] else 0
        else:
            self.n_fmi_panels = 0

        if self.config_nmr and (self.nmr_data is not None) and ('nmr_data' in self.nmr_data.keys()):
            self.n_nmr_panels = len(self.nmr_data['nmr_data']) if 'nmr_data' in self.nmr_data and self.nmr_data['nmr_data'] else 0
        else:
            self.n_nmr_panels = 0

        return self.n_curve_panels, self.n_type_panels, self.n_fmi_panels, self.n_nmr_panels

    def _calculate_curve_display_limits(self, curve_col: List[str] or str):
        """
        计算曲线的显示范围（X轴范围）

        范围计算策略：
        1. 过滤异常值（-999到999之外的值）
        2. 处理常数数据：添加相对边距
        3. 处理非常数数据：添加5%边距
        4. 确保非负数据的显示范围从0或正值开始
        """
        curve_data = self.logging_data[curve_col]

        # 过滤异常值：排除极端值和空值
        valid_mask = (curve_data > -99999) & (curve_data < 999999) & ~curve_data.isna()
        valid_data = curve_data[valid_mask]

        if valid_data.empty:
            # 无有效数据时使用原始数据范围
            min_val, max_val = curve_data.min(), curve_data.max()
        else:
            # 使用有效数据范围
            min_val, max_val = valid_data.min(), valid_data.max()

        # 处理常数数据（变化范围极小）
        if abs(max_val - min_val) < 1e-10:
            # 添加相对边距：非零值用10%边距，零值用固定边距
            margin = abs(min_val) * 0.1 if min_val != 0 else 1.0
            min_val -= margin
            max_val += margin
        else:
            # 添加5%边距
            data_range = max_val - min_val
            margin = data_range * 0.05
            # 非负数据确保从0或正值开始显示
            min_val = max(0, min_val - margin) if min_val >= 0 else min_val - margin
            max_val += margin

        return min_val, max_val

    def get_class_logger(self):
        return logger



if __name__ == '__main__':

    work_well = DATA_WELL(path_folder=r'F:\logging_workspace\FY1-15')

    path_list_logging = work_well.get_path_list_logging()
    path_list_fmi = work_well.get_path_list_fmi()
    path_list_table = work_well.get_path_list_table()

    print(path_list_logging)
    print(path_list_fmi)
    print(path_list_table)

    logging_data = work_well.combine_logging_table(logging_key=path_list_logging[0], table_key=path_list_table[0],
                                                   curve_names_logging=['DEPTH', 'CON_MEAN_DYNA', 'DIS_MEAN_DYNA', 'HOM_MEAN_DYNA', 'ENG_MEAN_DYNA', 'COR_MEAN_DYNA', 'ASM_MEAN_DYNA', 'ENT_MEAN_DYNA', 'CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA']
                                                   )

    print(logging_data.describe())
    print(logging_data.head(10))
    COLS_ALL = logging_data.columns.to_list()
    print(f"总列数: {len(COLS_ALL)}")
    print(f"所有列名: {COLS_ALL}")
    print(work_well.get_table_replace_dict())

    data_fmi, depth_fmi = work_well.get_FMI(key=path_list_fmi[4])
    data_nmr, depth_nmr = work_well.get_FMI(key=path_list_fmi[-1])
    print(data_fmi.shape, depth_fmi.shape, data_nmr.shape, depth_nmr.shape)
    print(depth_fmi[0], depth_fmi[-1])
    print(depth_nmr[0], depth_nmr[-1])

    depth_config_list = [[3121, 3122],
                         [3121, 3122],
                         [3121, 3122],
                         [3121.5, 3122.5],
                         [3121.5, 3122.5],
                         [3121.5, 3122.5],
                         [3121.5, 3122.5],
                         [3122.5, 3127.5],
                         [3122.5, 3127.5],
                         [3164, 3168],
                         [3164, 3168],]

    LDM = LoggingDataManager(
        logging_data=logging_data,
        # logging_data=pd.DataFrame(),
        fmi_data={'depth': depth_fmi, 'image_data': [data_fmi]},
        nmr_data={'depth': depth_nmr, 'nmr_data': [data_nmr]}
    )
    print(LDM._get_depth_limits())

    # initial_stats = LDM.get_performance_stats()
    # num_iterations = 3
    #
    # import time
    # # 测试常规测井数据
    # print("\n📊 测试常规测井数据...")
    # for i, depth_config in enumerate(depth_config_list):
    #     top_depth, bottom_depth = depth_config
    #     times = []
    #
    #     start_time = time.perf_counter()
    #     data = LDM.get_visible_logging_data(top_depth, bottom_depth)
    #     # print(data.describe())
    #     data = LDM.get_visible_fmi_data(top_depth, bottom_depth)
    #     if data is not None:
    #         print(data['title'])
    #         print(data['image_data'][0].shape)
    #         print(data['depth'].shape)
    #     else:
    #         print(data)
    #     elapsed = (time.perf_counter() - start_time) * 1000
    #     print(f"  范围 [{top_depth}-{bottom_depth}]: 平均 {elapsed:.2f}ms")

    config_plot = LDM.plot_config_check(config_logging={'curves_plot':['CON_MEAN_DYNA', 'DIS_MEAN_DYNA', 'HOM_MEAN_DYNA', 'ENG_MEAN_DYNA', ['COR_MEAN_DYNA', 'ASM_MEAN_DYNA', 'ENT_MEAN_DYNA'], ['CON_SUB_DYNA', 'DIS_SUB_DYNA']]})
    for i, key in enumerate(config_plot):
        print(key)
        print(config_plot[key])

    # print(LDM.get_logging_resolution())

