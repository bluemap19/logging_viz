# from dataclasses import dataclass
# import zlib
# import pickle
# import numpy as np
# import pandas as pd
# from typing import List, Dict, Tuple, Optional, Any, Union
# from collections import OrderedDict
# import logging
#
# logger = logging.getLogger(__name__)
#
#
# @dataclass
# class CacheConfig:
#     """缓存配置类"""
#     enabled: bool = True
#     max_size: int = 100
#     fmi_max_size: int = 100  # FMI缓存较小，因为数据量大
#     nmr_max_size: int = 100  # NMR缓存大小
#     compression_level: int = 1  # 压缩级别1-9
#
#
# class EnhancedWellLogCache:
#     """增强的测井数据缓存系统"""
#
#     def __init__(self, config: CacheConfig = None):
#         self.config = config or CacheConfig()
#
#         # 常规数据缓存
#         self._data_cache = OrderedDict()
#
#         # FMI图像缓存（使用压缩存储）
#         self._fmi_cache = OrderedDict()
#         self._fmi_compression_stats = {'compressed_size': 0, 'original_size': 0}
#
#         # NMR数据缓存
#         self._nmr_cache = OrderedDict()
#         self._nmr_compression_stats = {'compressed_size': 0, 'original_size': 0}
#
#         # 缓存统计
#         self.stats = {
#             'data_hits': 0, 'data_misses': 0,   # 常规测井缓存统计
#             'fmi_hits': 0, 'fmi_misses': 0,     # FMI缓存统计
#             'nmr_hits': 0, 'nmr_misses': 0      # NMR缓存统计
#         }
#
#     def _generate_cache_key(self, depth_range: Tuple[float, float], data_type: str = 'data') -> str:
#         """生成精确的缓存键"""
#         # 使用更高精度避免冲突
#         if data_type == 'fmi':
#             # FMI 缓存键生成
#             return f"fmi_{depth_range[0]:.4f}_{depth_range[1]:.4f}"
#         elif data_type == 'nmr':
#             # NMR, 缓存键生成
#             return f"nmr_{depth_range[0]:.4f}_{depth_range[1]:.4f}"
#         else:
#             # 常规测井, 缓存键生成
#             return f"data_{depth_range[0]:.4f}_{depth_range[1]:.4f}"
#
#     def _compress_data(self, data: Any) -> bytes:
#         """通用数据压缩方法"""
#         original_size = len(pickle.dumps(data))
#         compressed = zlib.compress(pickle.dumps(data), self.config.compression_level)
#         compressed_size = len(compressed)
#         return compressed, original_size, compressed_size
#
#     def _decompress_data(self, compressed_data: bytes) -> Any:
#         """通用数据解压缩方法"""
#         return pickle.loads(zlib.decompress(compressed_data))
#
#     def get_logging_data(self, depth_range: Tuple[float, float]) -> Optional[pd.DataFrame]:
#         """获取常规数据"""
#         if not self.config.enabled:
#             return None
#
#         key = self._generate_cache_key(depth_range, 'data')
#
#         if key in self._data_cache:
#             self._data_cache.move_to_end(key)
#             self.stats['data_hits'] += 1
#             return self._data_cache[key]
#
#         self.stats['data_misses'] += 1
#         return None
#
#     def set_logging_data(self, depth_range: Tuple[float, float], data: pd.DataFrame):
#         """设置常规数据缓存"""
#         if not self.config.enabled:
#             return
#
#         key = self._generate_cache_key(depth_range, 'data')
#         self._data_cache[key] = data
#         self._data_cache.move_to_end(key)
#
#         # 维护缓存大小
#         while len(self._data_cache) > self.config.max_size:
#             self._data_cache.popitem(last=False)
#
#     def _compress_fmi_data(self, image_data: np.ndarray) -> bytes:
#         """ 压缩FMI图像数据 """
#         original_size = image_data.nbytes
#         compressed = zlib.compress(pickle.dumps(image_data), self.config.compression_level)
#         compressed_size = len(compressed)
#
#         self._fmi_compression_stats['compressed_size'] += compressed_size
#         self._fmi_compression_stats['original_size'] += original_size
#
#         compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
#         logger.debug(f"FMI数据压缩: {original_size} -> {compressed_size} (比率: {compression_ratio:.2f})")
#
#         return compressed
#
#     def _decompress_fmi_data(self, compressed_data: bytes) -> np.ndarray:
#         """解压缩FMI图像数据"""
#         return pickle.loads(zlib.decompress(compressed_data))
#
#     def get_fmi_data(self, depth_range: Tuple[float, float]) -> Optional[np.ndarray]:
#         """获取FMI数据"""
#         if not self.config.enabled:
#             return None
#
#         key = self._generate_cache_key(depth_range, 'fmi')
#
#         if key in self._fmi_cache:
#             self._fmi_cache.move_to_end(key)
#             self.stats['fmi_hits'] += 1
#             compressed_data = self._fmi_cache[key]
#             return self._decompress_fmi_data(compressed_data)
#
#         self.stats['fmi_misses'] += 1
#         return None
#
#     def set_fmi_data(self, depth_range: Tuple[float, float], image_data: np.ndarray):
#         """设置FMI数据缓存"""
#         if not self.config.enabled:
#             return
#
#         key = self._generate_cache_key(depth_range, 'fmi')
#         compressed_data = self._compress_fmi_data(image_data)
#         self._fmi_cache[key] = compressed_data
#         self._fmi_cache.move_to_end(key)
#
#         # 维护缓存大小
#         while len(self._fmi_cache) > self.config.fmi_max_size:
#             self._fmi_cache.popitem(last=False)
#
#     def get_nmr_data(self, depth_range: Tuple[float, float]) -> Optional[Dict[float, Dict[str, Any]]]:
#         """获取NMR数据缓存"""
#         if not self.config.enabled:
#             return None
#
#         key = self._generate_cache_key(depth_range, 'nmr')
#
#         if key in self._nmr_cache:
#             self._nmr_cache.move_to_end(key)
#             self.stats['nmr_hits'] += 1
#             compressed_data = self._nmr_cache[key]
#             return self._decompress_data(compressed_data)
#
#         self.stats['nmr_misses'] += 1
#         return None
#
#     def set_nmr_data(self, depth_range: Tuple[float, float], nmr_data: Dict[float, Dict[str, Any]]):
#         """设置NMR数据缓存"""
#         if not self.config.enabled:
#             return
#
#         key = self._generate_cache_key(depth_range, 'nmr')
#         compressed_data, original_size, compressed_size = self._compress_data(nmr_data)
#
#         # 更新压缩统计
#         self._nmr_compression_stats['compressed_size'] += compressed_size
#         self._nmr_compression_stats['original_size'] += original_size
#
#         compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
#         logger.debug(f"NMR数据压缩: {original_size} -> {compressed_size} (比率: {compression_ratio:.2f})")
#
#         self._nmr_cache[key] = compressed_data
#         self._nmr_cache.move_to_end(key)
#
#         # 维护缓存大小
#         while len(self._nmr_cache) > self.config.nmr_max_size:
#             removed_key, removed_data = self._nmr_cache.popitem(last=False)
#             # 更新统计
#             removed_size = len(removed_data)
#             self._nmr_compression_stats['compressed_size'] -= removed_size
#
#     def get_cache_stats(self) -> Dict[str, Any]:
#         """获取缓存统计 - 包含NMR缓存信息"""
#         num_data_hit, num_data_misses = self.stats['data_hits'],  self.stats['data_misses']
#         num_fmi_hit, num_fmi_misses = self.stats['fmi_hits'], self.stats['fmi_misses']
#         num_nmr_hit, num_nmr_misses = self.stats['nmr_hits'], self.stats['nmr_misses']
#
#         data_hit_rate = (num_data_hit/(num_data_hit+num_data_misses) * 100 if (num_data_hit+num_data_misses) > 0 else 0)
#         fmi_hit_rate = (num_fmi_hit/(num_fmi_hit+num_fmi_misses) * 100 if (num_fmi_hit+num_fmi_misses) > 0 else 0)
#         nmr_hit_rate = (num_nmr_hit/(num_nmr_hit+num_nmr_misses) * 100 if (num_nmr_hit+num_nmr_misses) > 0 else 0)
#         fmi_compression_ratio = (self._fmi_compression_stats['original_size']/self._fmi_compression_stats['compressed_size'] if self._fmi_compression_stats['compressed_size'] > 0 else 0)
#         nmr_compression_ratio = (self._nmr_compression_stats['original_size']/self._nmr_compression_stats['compressed_size'] if self._nmr_compression_stats['compressed_size'] > 0 else 0)
#         fmi_memory_saved = (self._fmi_compression_stats['original_size'] - self._fmi_compression_stats['compressed_size']) / (1024 * 1024)
#         nmr_memory_saved = (self._nmr_compression_stats['original_size'] - self._nmr_compression_stats['compressed_size']) / (1024 * 1024)
#
#         return {
#             'data_cache_size': len(self._data_cache),
#             'fmi_cache_size': len(self._fmi_cache),
#             'nmr_cache_size': len(self._nmr_cache),
#             'data_hit_rate': data_hit_rate,
#             'fmi_hit_rate': fmi_hit_rate,
#             'nmr_hit_rate': nmr_hit_rate,
#             'fmi_compression_ratio': fmi_compression_ratio,
#             'nmr_compression_ratio': nmr_compression_ratio,
#             'fmi_memory_saved_mb': fmi_memory_saved,
#             'nmr_memory_saved_mb': nmr_memory_saved,
#             'total_memory_saved_mb': fmi_memory_saved + nmr_memory_saved
#         }
#
#     def clear_cache(self):
#         """清空缓存 - 包含NMR缓存"""
#         self._data_cache.clear()
#         self._fmi_cache.clear()
#         self._nmr_cache.clear()
#         self.stats = {'data_hits': 0, 'data_misses': 0, 'fmi_hits': 0, 'fmi_misses': 0, 'nmr_hits': 0, 'nmr_misses': 0}
#         self._fmi_compression_stats = {'compressed_size': 0, 'original_size': 0}
#         self._nmr_compression_stats = {'compressed_size': 0, 'original_size': 0}


from dataclasses import dataclass
import zlib
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """缓存配置类"""
    enabled: bool = True
    max_size: int = 100
    fmi_max_size: int = 50  # FMI缓存较小，因为数据量大
    nmr_max_size: int = 100  # NMR缓存大小
    compression_level: int = 1  # 压缩级别1-9


class EnhancedWellLogCache:
    """增强的测井数据缓存系统 - 适配新的List格式数据"""

    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()

        # 常规数据缓存 (DataFrame格式)
        self._data_cache = OrderedDict()

        # FMI图像缓存 (List[np.ndarray]格式)
        self._fmi_cache = OrderedDict()
        self._fmi_compression_stats = {'compressed_size': 0, 'original_size': 0}

        # NMR数据缓存 (List[np.ndarray]格式)
        self._nmr_cache = OrderedDict()
        self._nmr_compression_stats = {'compressed_size': 0, 'original_size': 0}

        # 缓存统计
        self.stats = {
            'data_hits': 0, 'data_misses': 0,  # 常规测井缓存统计
            'fmi_hits': 0, 'fmi_misses': 0,  # FMI缓存统计
            'nmr_hits': 0, 'nmr_misses': 0  # NMR缓存统计
        }

    def _generate_cache_key(self, depth_range: Tuple[float, float], data_type: str = 'data') -> str:
        """生成精确的缓存键"""
        if data_type == 'fmi':
            return f"fmi_{depth_range[0]:.4f}_{depth_range[1]:.4f}"
        elif data_type == 'nmr':
            return f"nmr_{depth_range[0]:.4f}_{depth_range[1]:.4f}"
        else:
            return f"data_{depth_range[0]:.4f}_{depth_range[1]:.4f}"

    def _compress_data(self, data: Any) -> Tuple[bytes, int, int]:
        """通用数据压缩方法"""
        original_data = pickle.dumps(data)
        original_size = len(original_data)
        compressed = zlib.compress(original_data, self.config.compression_level)
        compressed_size = len(compressed)
        return compressed, original_size, compressed_size

    def _decompress_data(self, compressed_data: bytes) -> Any:
        """通用数据解压缩方法"""
        return pickle.loads(zlib.decompress(compressed_data))

    # ========== 常规测井数据缓存 (DataFrame格式) ==========
    def get_logging_data(self, depth_range: Tuple[float, float]) -> Optional[pd.DataFrame]:
        """获取常规测井数据缓存"""
        if not self.config.enabled:
            return None

        key = self._generate_cache_key(depth_range, 'data')

        if key in self._data_cache:
            self._data_cache.move_to_end(key)
            self.stats['data_hits'] += 1
            logger.debug(f"常规数据缓存命中: {key}")
            return self._data_cache[key]

        self.stats['data_misses'] += 1
        return None

    def set_logging_data(self, depth_range: Tuple[float, float], data: pd.DataFrame):
        """设置常规测井数据缓存"""
        if not self.config.enabled or data is None or data.empty:
            return

        key = self._generate_cache_key(depth_range, 'data')
        self._data_cache[key] = data
        self._data_cache.move_to_end(key)

        # 维护缓存大小
        while len(self._data_cache) > self.config.max_size:
            removed_key, removed_data = self._data_cache.popitem(last=False)
            logger.debug(f"常规数据缓存溢出，移除: {removed_key}")

    # ========== FMI数据缓存 (List[np.ndarray]格式) ==========
    def get_fmi_data(self, depth_range: Tuple[float, float]) -> Optional[List[np.ndarray]]:
        """获取FMI数据缓存 - 返回List[np.ndarray]格式"""
        if not self.config.enabled:
            return None

        key = self._generate_cache_key(depth_range, 'fmi')

        if key in self._fmi_cache:
            self._fmi_cache.move_to_end(key)
            self.stats['fmi_hits'] += 1
            compressed_data = self._fmi_cache[key]
            fmi_list = self._decompress_data(compressed_data)

            # 验证数据格式
            if not isinstance(fmi_list, list) or not all(isinstance(arr, np.ndarray) for arr in fmi_list):
                logger.warning("FMI缓存数据格式异常，已清除")
                del self._fmi_cache[key]
                return None

            logger.debug(f"FMI缓存命中: {key}, 包含{len(fmi_list)}个图像")
            return fmi_list

        self.stats['fmi_misses'] += 1
        return None

    def set_fmi_data(self, depth_range: Tuple[float, float], fmi_data: List[np.ndarray]):
        """设置FMI数据缓存 - 接收List[np.ndarray]格式"""
        if not self.config.enabled or not fmi_data:
            return

        # 验证输入数据格式
        if not isinstance(fmi_data, list):
            logger.warning("FMI数据必须是列表格式")
            return

        valid_arrays = [arr for arr in fmi_data if isinstance(arr, np.ndarray)]
        if not valid_arrays:
            logger.warning("FMI数据列表中未找到有效的numpy数组")
            return

        key = self._generate_cache_key(depth_range, 'fmi')

        try:
            # 压缩整个列表
            compressed_data, original_size, compressed_size = self._compress_data(valid_arrays)

            # 更新压缩统计
            self._fmi_compression_stats['compressed_size'] += compressed_size
            self._fmi_compression_stats['original_size'] += original_size

            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
            logger.debug(f"FMI数据压缩: {original_size} -> {compressed_size} (比率: {compression_ratio:.2f})")

            self._fmi_cache[key] = compressed_data
            self._fmi_cache.move_to_end(key)

            # 维护缓存大小
            while len(self._fmi_cache) > self.config.fmi_max_size:
                removed_key, removed_data = self._fmi_cache.popitem(last=False)
                removed_size = len(removed_data)
                self._fmi_compression_stats['compressed_size'] -= removed_size
                logger.debug(f"FMI缓存溢出，移除: {removed_key}")

        except Exception as e:
            logger.error(f"FMI数据压缩失败: {e}")

    # ========== NMR数据缓存 (List[np.ndarray]格式) ==========
    def get_nmr_data(self, depth_range: Tuple[float, float]) -> Optional[List[np.ndarray]]:
        """获取NMR数据缓存 - 返回List[np.ndarray]格式"""
        if not self.config.enabled:
            return None

        key = self._generate_cache_key(depth_range, 'nmr')

        if key in self._nmr_cache:
            self._nmr_cache.move_to_end(key)
            self.stats['nmr_hits'] += 1
            compressed_data = self._nmr_cache[key]
            nmr_list = self._decompress_data(compressed_data)

            # 验证数据格式
            if not isinstance(nmr_list, list) or not all(isinstance(arr, np.ndarray) for arr in nmr_list):
                logger.warning("NMR缓存数据格式异常，已清除")
                del self._nmr_cache[key]
                return None

            logger.debug(f"NMR缓存命中: {key}, 包含{len(nmr_list)}个数据道")
            return nmr_list

        self.stats['nmr_misses'] += 1
        return None

    def set_nmr_data(self, depth_range: Tuple[float, float], nmr_data: List[np.ndarray]):
        """设置NMR数据缓存 - 接收List[np.ndarray]格式"""
        if not self.config.enabled or not nmr_data:
            return

        # 验证输入数据格式
        if not isinstance(nmr_data, list):
            logger.warning("NMR数据必须是列表格式")
            return

        valid_arrays = [arr for arr in nmr_data if isinstance(arr, np.ndarray)]
        if not valid_arrays:
            logger.warning("NMR数据列表中未找到有效的numpy数组")
            return

        key = self._generate_cache_key(depth_range, 'nmr')

        try:
            # 压缩整个列表
            compressed_data, original_size, compressed_size = self._compress_data(valid_arrays)

            # 更新压缩统计
            self._nmr_compression_stats['compressed_size'] += compressed_size
            self._nmr_compression_stats['original_size'] += original_size

            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1
            logger.debug(f"NMR数据压缩: {original_size} -> {compressed_size} (比率: {compression_ratio:.2f})")

            self._nmr_cache[key] = compressed_data
            self._nmr_cache.move_to_end(key)

            # 维护缓存大小
            while len(self._nmr_cache) > self.config.nmr_max_size:
                removed_key, removed_data = self._nmr_cache.popitem(last=False)
                removed_size = len(removed_data)
                self._nmr_compression_stats['compressed_size'] -= removed_size
                logger.debug(f"NMR缓存溢出，移除: {removed_key}")

        except Exception as e:
            logger.error(f"NMR数据压缩失败: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取详细的缓存统计信息"""
        # 计算命中率
        data_hits, data_misses = self.stats['data_hits'], self.stats['data_misses']
        fmi_hits, fmi_misses = self.stats['fmi_hits'], self.stats['fmi_misses']
        nmr_hits, nmr_misses = self.stats['nmr_hits'], self.stats['nmr_misses']

        data_hit_rate = (data_hits / (data_hits + data_misses) * 100
                         if (data_hits + data_misses) > 0 else 0)
        fmi_hit_rate = (fmi_hits / (fmi_hits + fmi_misses) * 100
                        if (fmi_hits + fmi_misses) > 0 else 0)
        nmr_hit_rate = (nmr_hits / (nmr_hits + nmr_misses) * 100
                        if (nmr_hits + nmr_misses) > 0 else 0)

        # 计算压缩比率
        fmi_compression_ratio = (self._fmi_compression_stats['original_size'] /
                                 self._fmi_compression_stats['compressed_size']
                                 if self._fmi_compression_stats['compressed_size'] > 0 else 0)
        nmr_compression_ratio = (self._nmr_compression_stats['original_size'] /
                                 self._nmr_compression_stats['compressed_size']
                                 if self._nmr_compression_stats['compressed_size'] > 0 else 0)

        # 计算内存节省(MB)
        fmi_memory_saved = (self._fmi_compression_stats['original_size'] -
                            self._fmi_compression_stats['compressed_size']) / (1024 * 1024)
        nmr_memory_saved = (self._nmr_compression_stats['original_size'] -
                            self._nmr_compression_stats['compressed_size']) / (1024 * 1024)

        return {
            # 缓存大小
            'data_cache_size': len(self._data_cache),
            'fmi_cache_size': len(self._fmi_cache),
            'nmr_cache_size': len(self._nmr_cache),
            'max_sizes': {
                'data': self.config.max_size,
                'fmi': self.config.fmi_max_size,
                'nmr': self.config.nmr_max_size
            },

            # 命中率统计
            'hit_rates': {
                'data': round(data_hit_rate, 2),
                'fmi': round(fmi_hit_rate, 2),
                'nmr': round(nmr_hit_rate, 2)
            },

            # 命中次数
            'hit_counts': {
                'data': data_hits,
                'fmi': fmi_hits,
                'nmr': nmr_hits
            },

            # 压缩统计
            'compression_ratios': {
                'fmi': round(fmi_compression_ratio, 2),
                'nmr': round(nmr_compression_ratio, 2)
            },

            # 内存节省
            'memory_saved_mb': {
                'fmi': round(fmi_memory_saved, 2),
                'nmr': round(nmr_memory_saved, 2),
                'total': round(fmi_memory_saved + nmr_memory_saved, 2)
            },

            # 原始数据大小(MB)
            'original_sizes_mb': {
                'fmi': round(self._fmi_compression_stats['original_size'] / (1024 * 1024), 2),
                'nmr': round(self._nmr_compression_stats['original_size'] / (1024 * 1024), 2)
            }
        }

    def clear_cache(self):
        """清空所有缓存"""
        self._data_cache.clear()
        self._fmi_cache.clear()
        self._nmr_cache.clear()

        # 重置统计
        self.stats = {
            'data_hits': 0, 'data_misses': 0,
            'fmi_hits': 0, 'fmi_misses': 0,
            'nmr_hits': 0, 'nmr_misses': 0
        }
        self._fmi_compression_stats = {'compressed_size': 0, 'original_size': 0}
        self._nmr_compression_stats = {'compressed_size': 0, 'original_size': 0}

        logger.info("所有缓存已清空")

    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况(MB)"""
        memory_usage = {}

        # 估算常规数据缓存内存
        data_memory = sum(df.memory_usage(deep=True).sum() for df in self._data_cache.values())
        memory_usage['data_mb'] = data_memory / (1024 * 1024)

        # FMI和NMR缓存使用压缩后大小
        memory_usage['fmi_mb'] = self._fmi_compression_stats['compressed_size'] / (1024 * 1024)
        memory_usage['nmr_mb'] = self._nmr_compression_stats['compressed_size'] / (1024 * 1024)
        memory_usage['total_mb'] = sum(memory_usage.values())

        return memory_usage