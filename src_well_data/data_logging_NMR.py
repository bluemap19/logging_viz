import os
import numpy as np
import pandas as pd
from src_data_process.cal_data_glcm_texture import cal_images_texture
import logging
from typing import Optional, List, Dict, Tuple
from enum import Enum

from src_fmi.fmi_fractal_dimension_extended_calculate import  trans_NMR_as_Ciflog_file_type
from src_plot.TEMP_9 import WellLogVisualizer

# 完整显示describe的全部信息不省略
#set_option函数可以解决数据显示不全问题，比如自动换行、显示……这种，本题不设置就会报错
pd.set_option('display.float_format', lambda x:'%.4f'%x) # 小数点后面保留3位小数，诸如此类，按需修改吧
pd.set_option('display.max_columns', None)# 显示所有的列，而不是以……显示
pd.set_option('display.max_rows', None)# 显示所有的行，而不是以……显示
pd.set_option('display.width', None) # 不自动换行显示


class NMRException(Exception):
    """
    NMR数据异常类
    用于处理核磁数据相关的特定异常情况
    """
    pass

class FileFormat(Enum):
    """
    文件格式枚举类
    定义支持的NMR数据文件格式
    """
    CSV = '.csv'
    TEXT = '.txt'
    UNKNOWN = 'unknown'


class DataNMR:
    """
    NMR核磁数据管理核心类

    功能概述：
    1. 支持CSV和TXT格式的NMR数据读取
    4. 管理深度数据和核磁数据的对应关系

    核磁数据特点：
    - 数据量大：每个深度点对应该深度下测量计算得到的核磁T2谱
    - 深度连续性：数据按深度严格排序

    设计原则：
    - 数据封装：内部数据状态受保护，通过方法访问
    - 异常安全：完善的错误处理和数据验证
    """

    def __init__(self, path_nmr: str = '', well_name: str = '', nmr_charter: str = ''):
        """
        初始化nmr数据对象

        Args:
            path_nmr: nmr数据文件路径，支持.csv和.txt格式
            well_name: 井名标识，用于数据标识和日志记录
            nmr_charter: nmr仪器标识，用于特征命名区分

        Attributes:
            _table_2: 保留属性，用于与其他数据格式兼容
            _well_name: 井名标识
            _data_nmr: 存储原始nmr核磁数据（二维numpy数组）
            _resolution: nmr数据分辨率（深度采样间隔）
            _data_depth: 存储深度数据（一维numpy数组）
            path_nmr: 数据文件路径
            nmr_charter: nmr仪器标识
            _logger: 日志记录器实例
            _is_data_loaded: 数据加载状态标志
        """
        # 数据存储属性
        self._table_2: pd.DataFrame = pd.DataFrame()  # 保留属性，用于兼容性
        self._data_nmr: np.ndarray = np.array([])  # nmr核磁数据体
        self._data_depth: np.ndarray = np.array([])  # 深度数据

        # 配置参数
        self._resolution: float = 0.0025  # 默认分辨率
        self._well_name: str = well_name
        self.path_nmr: str = path_nmr
        if len(nmr_charter) == 0:
            if path_nmr.upper().__contains__('NMR'):
                nmr_charter = 'NMR'
            else:
                nmr_charter = 'UNKNOWN'
        self.nmr_charter: str = nmr_charter
        if self._well_name == '':
            self._well_name = self.path_nmr.split('\\')[-2]

        # 状态标志
        self._is_data_loaded: bool = False

        # 初始化日志系统
        self._logger = self._setup_logger()

        # 检查文件是否存在
        if path_nmr and not os.path.isfile(path_nmr):
            self._logger.error(f"文件不存在或无法访问: {path_nmr}")

    def _setup_logger(self) -> logging.Logger:
        """
        设置并配置日志记录器

        Returns:
            配置好的logging.Logger实例
        """
        logger = logging.getLogger(f"DataNMR_{self._well_name}")

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger

    def _detect_file_format(self, file_path: str) -> FileFormat:
        """
        检测文件格式

        Args:
            file_path: 文件路径

        Returns:
            检测到的文件格式枚举值
        """
        if file_path.endswith('.csv'):
            return FileFormat.CSV
        elif file_path.endswith('.txt'):
            return FileFormat.TEXT
        else:
            return FileFormat.UNKNOWN

    def read_data(self, file_path: str = '') -> None:
        """
        读取NMR核磁数据文件

        Args:
            file_path: 数据文件路径，为空时使用对象初始化路径

        Raises:
            NMRException: 文件读取失败或格式不支持时抛出

        Workflow:
            1. 确定文件路径并检查存在性
            2. 根据文件格式调用相应的读取方法
            3. 提取深度数据和核磁数据
            4. 更新数据加载状态
        """
        # 确定文件路径
        file_path = file_path or self.path_nmr

        if not file_path:
            raise NMRException("未提供文件路径")

        if not os.path.isfile(file_path):
            raise NMRException(f"文件不存在: {file_path}")

        try:
            # 检测文件格式并读取数据
            file_format = self._detect_file_format(file_path)
            self._logger.info(f"检测到文件格式: {file_format.value}")

            if file_format == FileFormat.CSV:
                self._read_csv_file(file_path)
            elif file_format == FileFormat.TEXT:
                self._read_text_file(file_path)
            else:
                raise NMRException(f"不支持的文件格式: {file_path}")

            # 验证数据完整性
            if self._data_nmr.size == 0 or self._data_depth.size == 0:
                raise NMRException("读取到的数据为空")

            # 更新加载状态
            self._is_data_loaded = True
            self._logger.info(f"成功加载NMR数据，形状: {self._data_nmr.shape}")

        except Exception as e:
            self._logger.error(f"读取NMR数据失败: {e}")
            raise

    def _read_csv_file(self, file_path: str) -> None:
        """
        读取CSV格式的NMR数据文件
        Args:
            file_path: CSV文件路径
        Note:
            CSV格式假设第一列为深度，其余列为电极测量值
        """
        try:
            # 读取CSV文件，第一列作为索引（通常是深度）
            df = pd.read_csv(file_path, index_col=0)

            # 提取数据：第一列之后的所有列为NMR数据
            self._data_nmr = df.values
            # 索引列作为深度数据
            self._data_depth = df.index.values

            self._logger.debug(f"CSV文件读取成功，数据形状: {self._data_nmr.shape}")

        except Exception as e:
            raise NMRException(f"CSV文件读取失败: {e}")

    def _read_text_file(self, file_path: str) -> None:
        """
        读取TXT格式的NMR数据文件（通常为LAS格式）
        Args:
            file_path: TXT文件路径
        Note:
            TXT 格式通常有文件头，需要跳过前几行
        """
        try:
            # 跳过前8行（LAS文件头），使用制表符分隔
            data_file = np.loadtxt(file_path, delimiter='\t', skiprows=8)

            # 第一列为深度，其余列为NMR数据
            self._data_depth = data_file[:, 0]
            self._data_nmr = data_file[:, 1:]

            self._logger.debug(f"TXT文件读取成功，数据形状: {self._data_nmr.shape}")

        except Exception as e:
            raise NMRException(f"TXT文件读取失败: {e}")

    def get_data(self, depth: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取NMR核磁数据和对应的深度数据
        Args:
            depth: 指定深度范围，为None时返回全部数据
        Returns:
            Tuple 包含NMR数据数组和深度数据数组
        Workflow:
            1. 惰性加载：如果数据未加载，先读取数据
            2. 深度筛选：如果指定深度范围，进行数据筛选
            3. 返回请求的数据
        """
        # 惰性加载数据
        if not self._is_data_loaded:
            self.read_data()

        if depth is None or len(depth) == 0:
            # 返回全部数据
            return self._data_nmr, self._data_depth
        else:
            # 根据深度范围筛选数据（待实现）
            depth_min = min(depth)
            depth_max = max(depth)
            idx = (self._data_depth >= depth_min) & (self._data_depth <= depth_max)
            return self._data_nmr[idx], self._data_depth[idx]


    def get_summary(self) -> Dict[str, any]:
        """
        获取NMR数据摘要信息

        Returns:
            包含各类统计信息的字典
        """
        summary = {
            'well_name': self._well_name,
            'nmr_charter': self.nmr_charter,
            'file_path': self.path_nmr,
            'is_loaded': self._is_data_loaded,
            'resolution': self._resolution
        }

        if self._is_data_loaded:
            summary.update({
                'data_shape': self._data_nmr.shape,
                'depth_range': (np.min(self._data_depth), np.max(self._data_depth)),
                'data_type': str(self._data_nmr.dtype)
            })

        return summary

    def get_data_info(self) -> str:
        """
        获取NMR数据的详细信息字符串

        Returns:
            格式化的数据信息字符串
        """
        if not self._is_data_loaded:
            return "数据未加载"

        info_lines = [
            f"井名: {self._well_name}",
            f"仪器: {self.nmr_charter}",
            f"数据形状: {self._data_nmr.shape}",
            f"深度范围: {np.min(self._data_depth):.2f} - {np.max(self._data_depth):.2f}",
            f"数据类型: {self._data_nmr.dtype}",
            f"分辨率: {self._resolution} 米/点"
        ]

        return "\n".join(info_lines)



def user_specific_test():
    """
    用户特定测试 - 使用用户提供的文件路径
    """
    print("\n" + "=" * 60)
    print("用户特定测试")
    print("=" * 60)

    # 用户提供的测试用例
    test_case = {
        'path_nmr': r'F:\logging_workspace\FY1-15\丰页1-15HF_DYNA_ORIGIN_TEST.txt',
        'nmr_charter': 'NMR'
    }

    print(f"测试文件: {test_case['path_nmr']}")
    print(f"仪器: {test_case['nmr_charter']}")
    print("-" * 50)

    try:
        # 创建nmr处理器实例
        test_NMR = DataNMR(
            path_nmr=test_case['path_nmr'],
            nmr_charter=test_case['nmr_charter']
        )

        print(">>> 获取数据...")
        nmr_data, depth_data = test_NMR.get_data()

        NMR_DICT = {
            'depth': depth_data,
            'nmr_data': nmr_data
        }

        print(f"nmr数据形状: {nmr_data.shape}")
        print(f"深度数据形状: {depth_data.shape}")

        if depth_data.size > 0:
            print(f"深度范围: {depth_data.min():.3f} - {depth_data.max():.3f}")

        # 显示数据摘要
        print("\n>>> 数据摘要:")
        summary = test_NMR.get_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")

        visualizer = WellLogVisualizer()
        # 执行可视化
        visualizer.visualize(
            logging_dict=None,
            NMR_dict=[NMR_DICT],
            NMR_CONFIG={'X_LOG': [True, True],
                        'NMR_TITLE': ['α-fα-DYNA', 'α-fα-STAT'],
                        'X_LIMIT': [[0, 6.4], [0, 6.4]],
                        'Y_scaling_factor': 4},
            # depth_limit_config=[320, 380],                      # 深度限制
            figsize=(12, 10)  # 图形尺寸
        )

        # 显示性能统计
        stats = visualizer.get_performance_stats()
        print("性能统计:", stats)

    except FileNotFoundError:
        print(f"文件不存在: {test_case['path_nmr']}")
        print("跳过该测试用例...")
    except Exception as e:
        print(f"测试失败: {e}")
        print("错误详情:", str(e))



if __name__ == '__main__':
    """
    主程序入口
    执行顺序：
    1. 综合测试（使用模拟数据）
    2. 用户特定测试（使用用户提供的文件路径）
    """

    # 设置日志级别
    logging.basicConfig(level=logging.INFO)

    # 执行用户特定测试
    user_specific_test()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)

