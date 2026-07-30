import os
import warnings
import numpy as np
import pandas as pd
import logging
from typing import Optional, List, Dict, Union, Tuple
from enum import Enum
from src_logging.curve_preprocess import get_resolution_by_depth
from src_table.table_process import table_2_to_3, get_replace_dict, table_3_to_2


class TableFormat(Enum):
    """
    表格格式枚举类
    定义支持的测井表格数据格式
    """
    UNKNOWN = 0          # 未知格式
    DEPTH_DATA = 1       # 深度-类型格式 (n×2)
    START_END_DATA = 2   # 开始深度-截止深度-类型格式 (n×3)


class DataTableException(Exception):
    """
    自定义异常类
    用于处理测井表格数据相关的特定异常情况
    """
    pass


class DataTableNumercial:
    """
    测井表格管理核心类

    功能概述：
    1. 支持两种测井数据格式的读取和互转：
       - 深度-类型格式（n×m）：深度值、岩心实验结果
       - 开始深度-截止深度-类型格式（n×3）：开始深度、截止深度、岩心实验结果
    2. 提供统一的数据访问接口

    设计原则：
    - 数据封装：内部数据状态受保护，通过方法访问
    - 惰性加载：数据在需要时才进行转换和处理
    - 异常安全：完善的错误处理和数据验证
    """

    def __init__(self, path: str = '', well_name: str = '', resolution: float = 0.0025, table_formate:int = TableFormat.DEPTH_DATA,):
        """
        初始化测井表格对象

        Args:
            path: 数据文件路径，支持.csv和.xlsx格式
            well_name: 井名标识，用于日志记录和数据标识
            resolution: 表格分辨率（深度采样间隔），单位：米，默认0.0025米
            table_formate：表格类型

        Attributes:
            _table_2: 存储原始的深度-岩心数据格式数据（2列DataFrame）
            _table_3: 存储原始的开始-截止-岩心数据格式数据（3列DataFrame）
            _table_resolution: 数据分辨率参数
            _file_path: 源数据文件路径
            _well_name: 井名标识
            _raw_data: 从文件直接读取的原始数据（不做格式处理）
            _is_data_loaded: 数据加载状态标志
        """
        # 数据存储属性
        self._table_2: pd.DataFrame = pd.DataFrame()  # 深度-岩心数据格式原始数据
        self._table_3: pd.DataFrame = pd.DataFrame()  # 开始-结束-岩心数据格式原始数据

        # 配置参数
        self._table_resolution: float = resolution  # 深度采样分辨率
        self._file_path: str = path  # 数据文件路径
        self._well_name: str = well_name  # 井名标识

        # 原始数据和状态
        self._raw_data: pd.DataFrame = pd.DataFrame()  # 从文件读取的原始数据
        self._is_data_loaded: bool = False  # 数据是否已加载标志
        self.table_formate = table_formate

        # 初始化日志系统
        self._logger = self._setup_logger()


        if not os.path.exists(self._file_path):
            self._logger.error('file_path {} does not exist!'.format(self._file_path))

        # 列名常量定义 - 确保整个类中列名使用的一致性
        self.COLUMN_NAMES_2 = []  # 2列表格的列名
        self.COLUMN_NAMES_3 = []  # 3列表格的列名

    def _setup_logger(self) -> logging.Logger:
        """
        设置并配置日志记录器

        Returns:
            配置好的logging.Logger实例

        Note:
            - 每个井使用独立的logger，便于区分不同井的日志
            - 日志格式包含时间、井名、日志级别和消息
        """
        # 创建以井名命名的logger，便于区分不同井的日志
        logger = logging.getLogger(f"DataTable_{self._well_name}")

        # 避免重复添加handler（防止多次调用时产生重复日志）
        if not logger.handlers:
            handler = logging.StreamHandler()  # 输出到控制台
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)  # 设置日志级别为INFO

        return logger

    def read_data(self, file_path: str = '', table_format: Optional[TableFormat] = None) -> None:
        """
        读取测井数据文件的主入口方法

        Args:
            file_path: 数据文件路径，为空时使用对象初始化路径
            table_format: 强制指定表格格式，为None时自动检测

        Raises:
            DataTableException: 文件读取失败、数据为空或格式处理异常时抛出

        Workflow:
            1. 检查数据是否已加载（避免重复加载）
            2. 确定文件路径和表格名称
            3. 读取文件内容到_raw_data
            4. 处理数据格式识别和转换
            5. 提取类型替换字典
            6. 更新加载状态标志
        """
        # 检查避免重复加载数据
        if self._is_data_loaded:
            self._logger.info("数据已加载，跳过重复读取")
            return

        try:
            # 步骤1: 确定文件路径（优先使用参数路径，其次使用初始化路径）
            file_path = file_path or self._file_path
            if not file_path:
                raise DataTableException("未提供文件路径")

            # 步骤2: 读取文件内容
            self._raw_data = self._read_file(file_path)
            if self._raw_data.empty:
                raise DataTableException("读取到的数据为空")

            # 步骤3: 处理数据格式识别和转换
            self._process_data_format(table_format)

            # 步骤5: 更新数据加载状态
            self._is_data_loaded = True
            self._logger.info(f"成功加载数据，格式: {table_format.name}")

        except Exception as e:
            self._logger.error(f"读取数据失败: {str(e)}")
            raise  # 重新抛出异常，让调用者处理

    def _read_file(self, file_path: str) -> pd.DataFrame:
        """
        文件读取的具体实现，支持CSV和Excel格式

        Args:
            file_path: 文件路径

        Returns:
            读取的DataFrame数据

        Raises:
            DataTableException: 文件格式不支持或读取失败时抛出
        """
        if file_path.endswith('.csv'):
            # CSV文件：尝试多种编码格式读取
            encodings = ['utf-8-sig', 'gbk', 'utf-8', 'latin-1']
            for encoding in encodings:
                try:
                    return pd.read_csv(file_path, encoding=encoding)
                except UnicodeDecodeError:
                    continue  # 尝试下一种编码
            raise DataTableException(f"无法解码CSV文件: {file_path}")

        elif file_path.endswith(('.xlsx', '.xls')):
            # Excel文件：尝试读取指定工作表
            try:
                # 确定工作表名称：默认第一个工作表(0)
                sheet_name = 0
                return pd.read_excel(file_path, sheet_name=sheet_name)
            except Exception as e:
                self._logger.warning(f"读取指定sheet失败: {e}，尝试读取第一个sheet")
                return pd.read_excel(file_path, sheet_name=0)

        else:
            raise DataTableException(f"不支持的文件格式: {file_path}")

    def _process_data_format(self, table_format: Optional[TableFormat] = None) -> None:
        """
        处理数据格式识别和相应的初始化转换

        Args:
            table_format: 强制指定的格式，为None时自动检测
        """
        # 确定数据格式（强制格式或自动检测）
        if table_format is not None:
            format_type = table_format
        else:
            format_type = self.table_formate

        if format_type == TableFormat.DEPTH_DATA:
            # 处理2列格式数据
            self._table_2 = self._raw_data.copy()
            self._check_table_2()  # 数据完整性检查

            # 自动计算分辨率
            self._table_resolution = get_resolution_by_depth(self._table_2.iloc[:, 0].values)

        elif format_type == TableFormat.START_END_DATA:
            # 处理3列格式数据
            self._table_3 = self._raw_data.copy()
            self._check_table_3()  # 数据完整性检查

            # 如果分辨率未设置或无效，使用默认值
            if self._table_resolution <= 0:
                self._table_resolution = 0.1
                self._logger.info("重置表格分辨率为0.1米")

            # 转换为2列格式
            self._convert_3_to_2()

    def _check_table_2(self) -> None:
        """
        深度-类型格式数据的完整性验证

        Checks:
            1. 数据表是否为空
            2. 列数是否为2
            3. 是否存在空值
            4. 深度值是否严格单调递增

        Raises:
            DataTableException: 当数据验证失败时抛出
        """
        # 检查1: 数据表是否为空
        if self._table_2.empty:
            raise DataTableException("深度-类型表格数据为空")

        # 检查2: 列数是否正确
        if self._table_2.shape[1] < 2:
            raise DataTableException(f"深度-类型表格应为2列，实际为{self._table_2.shape[1]}列")

        # 检查3: 空值处理
        null_mask = self._table_2.isnull().any(axis=1)
        if null_mask.any():
            null_count = null_mask.sum()
            self._logger.warning(f"发现{null_count}行空值数据，将被丢弃")
            self._table_2 = self._table_2.dropna().reset_index(drop=True)

        # 检查4: 深度单调性验证
        depths = self._table_2.iloc[:, 0].values  # 第一列永远是深度列
        if len(depths) > 1:
            depth_diffs = np.diff(depths)
            if not np.all(depth_diffs > 0):
                # 找到非递增的位置
                non_increasing_indices = np.where(depth_diffs <= 0)[0]
                raise DataTableException(f"深度值非单调递增，问题位置: {non_increasing_indices}")

        self.COLUMN_NAMES_2 = self._table_2.columns.tolist()

    def _check_table_3(self) -> None:
        """
        开始-结束-类型格式数据的完整性验证

        Checks:
            1. 数据表是否为空
            2. 列数是否为3
            3. 是否存在空值
            4. 开始深度是否小于结束深度
            5. 深度区间是否连续（警告级别检查）

        Raises:
            DataTableException: 当严重数据问题发现时抛出
        """
        # 检查1: 数据表是否为空
        if self._table_3.empty:
            raise DataTableException("开始-结束-类型表格数据为空")

        # 检查2: 列数验证
        if self._table_3.shape[1] < 3:
            raise DataTableException(f"开始-结束-类型表格应为3列，实际为{self._table_3.shape[1]}列")

        # 检查3: 空值处理
        null_mask = self._table_3.isnull().any(axis=1)
        if null_mask.any():
            null_count = null_mask.sum()
            self._logger.warning(f"发现{null_count}行空值数据，将被丢弃")
            self._table_3 = self._table_3.dropna().reset_index(drop=True)

        # 检查4: 深度区间合法性（开始深度 < 结束深度）
        depth_starts = self._table_3.iloc[:, 0].values
        depth_ends = self._table_3.iloc[:, 1].values

        invalid_intervals = depth_starts > depth_ends
        if invalid_intervals.any():
            invalid_indices = np.where(invalid_intervals)[0]
            raise DataTableException(f"开始深度不小于结束深度，问题行: {invalid_indices}")

        # 检查5: 深度区间连续性（相邻区间不应有重叠或过大间隙）
        if len(depth_ends) > 1:
            for i in range(len(depth_ends) - 1):
                if depth_ends[i] > depth_starts[i + 1]:
                    self._logger.warning(
                        f"深度区间重叠或间隙: 行{i}结束深度{depth_ends[i]} > "
                        f"行{i + 1}开始深度{depth_starts[i + 1]}"
                    )

        self.COLUMN_NAMES_3 = self._table_3.columns.tolist()


    def _convert_3_to_2(self, resolution: Optional[float] = None) -> None:
        """
        将开始-结束-类型格式(3列)转换为深度-类型格式(2列)

        Args:
            resolution: 采样分辨率，为None时使用对象分辨率

        Note:
            - 只有在_table_2为空且_table_3不为空时才执行转换
            - 转换结果存储在_table_2中
        """
        if self._table_2.empty and not self._table_3.empty:
            # 确定使用的分辨率
            resolution = resolution or self._table_resolution
            # 调用转换函数
            table_2_array = table_3_to_2(self._table_3.values, step=resolution)

            self.COLUMN_NAMES_2 = [r'DEP_START', r'DEP_END'] + list(table_2_array.columns)[1:]
            # 将转换结果重新封装为DataFrame
            self._table_2 = pd.DataFrame(table_2_array, columns=self.COLUMN_NAMES_2)
            self._logger.debug("完成3列到2列表格转换")

    def get_table_2(self, curve_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取深度-类型格式(2列)数据
        Args:
            curve_names: 指定返回的列名列表，为None时返回所有列
        Returns:
            深度-类型格式的DataFrame（2列）
        Workflow:
            1. 如果数据未加载，先读取数据
            2. 如果2列数据为空，从3列数据转换
            3. 按指定列名返回数据
        """
        # 惰性加载：如果数据未加载，先读取数据
        if not self._is_data_loaded:
            self.read_data()

        # 惰性转换：如果2列数据为空，从3列数据转换
        if self._table_2.empty:
            self._convert_3_to_2()

        # 列名过滤逻辑
        if curve_names and len(curve_names) == 2:
            return self._table_2[curve_names]

        # 默认返回深度列和类型列
        return self._table_2.iloc[:, [0, -1]]

    def set_resolution(self, resolution: float) -> None:
        """
        设置表格分辨率并重新转换数据

        Args:
            resolution: 新的分辨率值（必须为正数）

        Raises:
            DataTableException: 当分辨率非正时抛出
        """
        if resolution <= 0:
            raise DataTableException("分辨率必须为正数")

        self._table_resolution = resolution

        # 分辨率改变后，如果存在3列数据，需要重新转换为2列格式
        if not self._table_3.empty:
            self._convert_3_to_2(resolution)

    def get_summary(self) -> Dict[str, any]:
        """
        获取数据摘要信息

        Returns:
            包含各类统计信息的字典
        """
        return {
            'well_name': self._well_name,
            'file_path': self._file_path,
            'resolution': self._table_resolution,
            'is_loaded': self._is_data_loaded,
            'table_2_rows': len(self._table_2),
            'table_3_rows': len(self._table_3),
        }



def user_specific_test():
    """
    根据用户提供的初始化代码进行特定测试
    """
    print("\n" + "=" * 60)
    print("基于提供的文件路径")
    print("=" * 60)

    # 用户提供的测试用例
    test_cases = [
        {
            'path': r'Z:\logging_workspace\姬119H2\姬119H2导眼井全岩实验数据_table.csv',
            'well_name': '姬119H2',
            'description': '姬119H2井岩性类型数据'
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. 测试用例: {test_case['description']}")
        print(f"   文件路径: {test_case['path']}")
        print(f"   井名: {test_case['well_name']}")
        print("-" * 50)

        try:
            # 创建DataTable实例
            test_table = DataTableNumercial(
                path=test_case['path'],
                well_name=test_case['well_name']
            )

            print("\n>>> 2列表格数据统计:")
            table_2_data = test_table.get_table_2()
            print(table_2_data.describe())

            print("\n>>> 数据摘要信息:")
            summary = test_table.get_summary()
            for key, value in summary.items():
                print(f"   {key}: {value}")

        except FileNotFoundError:
            print(f"   文件不存在: {test_case['path']}")
            print("   跳过该测试用例...")
        except Exception as e:
            print(f"   测试失败: {e}")
            print("   错误详情:", str(e))

if __name__ == '__main__':
    """
    主程序入口
    执行顺序：
    1. 综合测试（使用模拟数据）
    2. 用户特定测试（使用用户提供的文件路径）
    """

    # 执行用户特定测试
    user_specific_test()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)