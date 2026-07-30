import os
import re
import logging
from enum import Enum
from typing import Optional, List, Dict, Union, Tuple

import numpy as np
import pandas as pd

# 完整显示 describe 的全部信息不省略
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


class CoreException(Exception):
    """
    岩心数据异常类
    用于处理岩心实验数据相关的特定异常情况
    """
    pass


class FileFormat(Enum):
    """
    文件格式枚举类
    定义支持的岩心数据文件格式
    """
    CSV = '.csv'
    EXCEL = '.xlsx'
    TEXT = '.txt'
    UNKNOWN = 'unknown'


class DataCore:
    """
    岩心实验数据管理核心类

    功能概述：
    1. 支持多种格式的岩心实验数据读取（CSV、Excel、TXT）
    2. 自动进行列名清理（去空格、大写化、去开头下划线）
    3. 自动计算深度分辨率
    4. 提供数据完整性和深度单调性验证
    5. 支持按深度范围筛选数据

    数据特点：
    - 稀疏性：深度点不连续，采样间隔远大于测井分辨率
    - 多维属性：每个深度点对应多种矿物/岩性组分数据
    - 深度有序：深度值严格单调递增

    设计原则：
    - 惰性加载：数据在需要时才进行读取和处理
    - 异常安全：完善的错误处理和数据验证
    - 配置外部化：列名映射通过标准化处理
    """


    def __init__(self, path: str = '', well_name: str = ''):
        """
        初始化岩心数据对象

        Args:
            path: 岩心数据文件路径，支持 CSV、Excel 和 TXT 格式
            well_name: 井名标识，用于数据标识和日志记录

        Attributes:
            _data: 存储原始岩心数据的 DataFrame
            _curve_names: 岩心数据列名列表（含 #Depth 深度列）
            _file_path: 数据文件路径
            _well_name: 井名标识
            _logger: 日志记录器实例
            _is_data_loaded: 数据加载状态标志
        """
        # 数据存储属性
        self._data: pd.DataFrame = pd.DataFrame()

        # 配置和元数据属性
        self._curve_names: List[str] = []
        self._file_path: str = path
        self._well_name: str = well_name
        self._is_data_loaded: bool = False

        # 初始化日志系统
        self._logger = self._setup_logger()

        # 检查文件是否存在
        if path and not os.path.isfile(path):
            self._logger.warning(f"文件不存在或无法访问: {path}")

    def _setup_logger(self) -> logging.Logger:
        """
        设置并配置日志记录器

        Returns:
            配置好的 logging.Logger 实例

        Note:
            - 每个井使用独立的 logger，便于区分不同井的日志
            - 日志格式包含时间、井名、日志级别和消息
        """
        logger = logging.getLogger(f"DataCore_{self._well_name}")

        # 避免重复添加 handler
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
        elif file_path.endswith('.xlsx'):
            return FileFormat.EXCEL
        elif file_path.endswith('.txt'):
            return FileFormat.TEXT
        else:
            return FileFormat.UNKNOWN

    def read_data(self, file_path: str = '') -> None:
        """
        读取岩心数据文件

        Args:
            file_path: 数据文件路径，为空时使用对象初始化路径

        Raises:
            CoreException: 文件读取失败或格式不支持时抛出

        Workflow:
            1. 检查数据是否已加载（避免重复加载）
            2. 确定文件路径和格式
            3. 根据格式调用相应的读取方法
            4. 初始化曲线名称和分辨率
            5. 更新加载状态标志
        """
        if self._is_data_loaded:
            self._logger.info("数据已加载，跳过重复读取")
            return

        try:
            # 确定文件路径
            file_path = file_path or self._file_path
            if not file_path:
                raise CoreException("未提供文件路径")

            # 检查文件是否存在
            if not os.path.isfile(file_path):
                raise CoreException(f"文件不存在: {file_path}")

            # 检测文件格式
            file_format = self._detect_file_format(file_path)
            self._logger.info(f"检测到文件格式: {file_format.value}")

            # 根据格式读取数据
            if file_format == FileFormat.CSV:
                # CSV 文件：尝试多种编码格式读取
                encodings = ['utf-8-sig', 'gbk', 'gb2312', 'utf-8', 'latin-1']
                data_read = False
                for encoding in encodings:
                    try:
                        self._data = pd.read_csv(file_path, encoding=encoding)
                        data_read = True
                        break
                    except UnicodeDecodeError:
                        continue  # 尝试下一种编码
                if not data_read:
                    raise CoreException(f"无法解码 CSV 文件: {file_path}")

            elif file_format == FileFormat.EXCEL:
                self._data = pd.read_excel(file_path, sheet_name=0)

            elif file_format == FileFormat.TEXT:
                # TXT 文件：尝试制表符分隔读取
                encodings = ['utf-8-sig', 'gbk', 'utf-8', 'latin-1']
                data_read = False
                for encoding in encodings:
                    try:
                        self._data = pd.read_csv(file_path, delimiter='\t', encoding=encoding)
                        data_read = True
                        break
                    except Exception:
                        continue
                if not data_read:
                    raise CoreException(f"无法解码 TXT 文件: {file_path}")

            else:
                raise CoreException(f"不支持的文件格式: {file_path}")

            # 检查数据是否成功读取
            if self._data.empty:
                raise CoreException("读取到的数据为空")

            # 删除所有列名中的空格，并将所有字符进行大写化
            self._data = self.columns_preprocess(self._data, to_uppercase=True, remove_all_spaces=True)

            # 初始化曲线名称
            self._curve_names = list(self._data.columns)
            self._logger.info(f"成功读取数据，包含 {len(self._curve_names)} 列")

            # 数据完整性验证
            self._validate_data()

            # 更新加载状态
            self._is_data_loaded = True

        except CoreException:
            raise
        except Exception as e:
            self._logger.error(f"读取数据失败: {e}")
            raise

    def columns_preprocess(self, dataframe: Optional[pd.DataFrame] = None,
                           remove_all_spaces: bool = False,
                           to_lowercase: bool = False,
                           to_uppercase: bool = False,
                           inplace: bool = True) -> pd.DataFrame:
        """
        高级列名清理函数，提供多种清理选项

        Args:
            dataframe: 要处理的 DataFrame，为 None 时处理 self._data
            remove_all_spaces: 是否删除所有空格（包括中间空格）
            to_lowercase: 是否转换为小写
            to_uppercase: 是否转换为大写
            inplace: 是否原地修改

        Returns:
            处理后的 DataFrame

        Example:
            输入: [' DEPTH  ', '石英', '钾长石']
            remove_all_spaces + to_uppercase: ['DEPTH', '石英', '钾长石']
        """
        # 确定要处理的数据框
        if dataframe is None:
            dataframe = self._data
            target_df = self._data
        else:
            target_df = dataframe if inplace else dataframe.copy()

        if target_df.empty:
            self._logger.warning("DataFrame 为空，跳过列名清理")
            return target_df

        # 保存原始列名
        original_columns = list(target_df.columns)
        cleaned_columns = []

        for col in original_columns:
            # 去除两端空格
            col_clean = str(col).strip()

            # 去除开头的下划线
            col_clean = col_clean.lstrip('_')

            # 处理空格选项
            if remove_all_spaces:
                # 删除所有空格
                col_clean = col_clean.replace(' ', '')
            else:
                # 仅合并连续空格
                col_clean = re.sub(r'\s+', ' ', col_clean)

            # 处理大小写选项
            if to_lowercase and not to_uppercase:
                col_clean = col_clean.lower()
            elif to_uppercase and not to_lowercase:
                col_clean = col_clean.upper()

            cleaned_columns.append(col_clean)

        # 检查列名是否唯一
        if len(cleaned_columns) != len(set(cleaned_columns)):
            duplicates = [col for col in cleaned_columns if cleaned_columns.count(col) > 1]
            self._logger.error(f"列名清理后出现重复: {duplicates}")
            raise CoreException(f"列名清理后出现重复列名: {duplicates}")

        # 应用新的列名
        target_df.columns = cleaned_columns

        # 记录变化
        changes = [(orig, clean) for orig, clean in zip(original_columns, cleaned_columns) if orig != clean]
        if changes:
            self._logger.info(f"高级列名清理完成，修改 {len(changes)} 个列名")
            for orig, clean in changes[:5]:
                self._logger.debug(f"'{orig}' -> '{clean}'")

        # 如果是处理 self._data，需要更新曲线名称列表
        if dataframe is None and hasattr(self, '_curve_names'):
            self._curve_names = cleaned_columns

        return target_df if inplace else target_df

    def _validate_data(self) -> None:
        """
        数据完整性验证

        Checks:
            1. 数据表是否为空
            2. 是否存在空值（严重警告）
            3. 深度值是否严格单调递增
            4. 第一列是否为深度列（列名含 DEPTH）

        Raises:
            CoreException: 当严重数据问题发现时抛出
        """
        # 检查1: 数据表是否为空
        if self._data.empty:
            raise CoreException("岩心数据表为空")

        # 检查2: 空值统计
        null_counts = self._data.isnull().sum()
        if null_counts.any():
            null_cols = null_counts[null_counts > 0]
            self._logger.warning(f"发现空值列: {dict(null_cols)}")

        # 检查3: 深度单调性验证（第一列默认为深度列）
        depth_col = self._data.columns[0]
        depths = self._data[depth_col].values

        if len(depths) > 1:
            depth_diffs = np.diff(depths)
            if not np.all(depth_diffs > 0):
                non_increasing_indices = np.where(depth_diffs <= 0)[0]
                raise CoreException(f"深度值非单调递增，问题位置: {non_increasing_indices}")

        # 检查4: 深度列名检测（软警告）
        if 'DEPTH' not in depth_col.upper():
            self._logger.warning(f"第一列 '{depth_col}' 可能不是深度列，请确认数据格式")

    def get_data(self, curve_names: Optional[List[str]] = None,
                 depth_range: Optional[List[float]] = None) -> pd.DataFrame:
        """
        获取岩心数据

        Args:
            curve_names: 指定要获取的列名列表，为 None 时获取所有列
            depth_range: 指定深度范围 [min_depth, max_depth]，为 None 时不限制

        Returns:
            包含指定列的岩心数据 DataFrame

        Workflow:
            1. 惰性加载：如果数据未加载，先读取数据
            2. 列名处理：确定要获取的列列表
            3. 深度过滤：如果指定了深度范围，进行数据筛选
        """
        # 步骤1: 惰性加载数据
        if not self._is_data_loaded:
            self.read_data()

        # 步骤2: 确定列名列表
        if curve_names is None or len(curve_names) == 0:
            curve_names = self._curve_names.copy()

        # 验证列名是否存在
        missing_cols = set(curve_names) - set(self._data.columns)
        if missing_cols:
            raise CoreException(f"列不存在: {missing_cols}")

        # 步骤3: 深度过滤
        if depth_range is not None and len(depth_range) == 2:
            depth_col = self._data.columns[0]
            depth_min, depth_max = min(depth_range), max(depth_range)
            mask = (self._data[depth_col] >= depth_min) & (self._data[depth_col] <= depth_max)
            data_result = self._data.loc[mask, curve_names].copy()
            self._logger.info(f"深度筛选 [{depth_min:.2f}, {depth_max:.2f}]，"
                              f"原始 {len(self._data)} 行 → 筛选后 {len(data_result)} 行")
        else:
            data_result = self._data[curve_names].copy()

        return data_result

    def get_summary(self) -> Dict[str, any]:
        """
        获取数据摘要信息

        Returns:
            包含各类统计信息的字典
        """
        summary = {
            'well_name': self._well_name,
            'file_path': self._file_path,
            'is_loaded': self._is_data_loaded,
            'data_shape': self._data.shape if not self._data.empty else (0, 0),
            'curve_count': len(self._curve_names),
            'columns': self._curve_names
        }

        if self._is_data_loaded and not self._data.empty:
            depth_col = self._data.columns[0]
            summary.update({
                'depth_min': float(self._data[depth_col].min()),
                'depth_max': float(self._data[depth_col].max()),
                'depth_range': (float(self._data[depth_col].min()),
                                float(self._data[depth_col].max())),
                'sample_count': len(self._data)
            })

        return summary

    def get_curve_names(self) -> List[str]:
        """
        获取所有列名（曲线名称）

        Returns:
            列名列表
        """
        if not self._is_data_loaded:
            self.read_data()
        return self._curve_names.copy()


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("岩心数据读取测试")
    print("=" * 60)

    test_path = r'Z:\logging_workspace\姬119H2\姬119H2导眼井全岩实验数据_core.csv'

    try:
        # 创建实例
        data_core = DataCore(path=test_path, well_name='姬119H2')

        # 测试数据读取
        print(">>> 原始数据读取:")
        df = data_core.get_data()
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
        print(df.head(10).to_string())

        # 测试按列获取
        print("\n>>> 按列获取 (石英、钾长石):")
        df_subset = data_core.get_data(curve_names=['DEPTH', '石英', '钾长石'])
        print(df_subset.head(5).to_string())

        # 测试按深度范围筛选
        print("\n>>> 深度范围筛选 [2730, 2745]:")
        df_filtered = data_core.get_data(depth_range=[2730, 2745])
        print(df_filtered.to_string())

        # 显示数据摘要
        print("\n>>> 数据摘要:")
        summary = data_core.get_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")

    except CoreException as e:
        print(f"岩心数据异常: {e}")
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
