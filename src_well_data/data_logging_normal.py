import os
import re
import sys
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Union, Tuple
from enum import Enum
import logging
from src_logging.curve_preprocess import data_Normalized, find_mode

# 完整显示describe的全部信息不省略
#set_option函数可以解决数据显示不全问题，比如自动换行、显示……这种，本题不设置就会报错
pd.set_option('display.float_format', lambda x:'%.4f'%x) # 小数点后面保留3位小数，诸如此类，按需修改吧
pd.set_option('display.max_columns', None)# 显示所有的列，而不是以……显示
pd.set_option('display.max_rows', None)# 显示所有的行，而不是以……显示
pd.set_option('display.width', None) # 不自动换行显示

class DataLoggingException(Exception):
    """
    自定义异常类
    用于处理测井数据相关的特定异常情况
    """
    pass


class FileFormat(Enum):
    """
    文件格式枚举类
    定义支持的测井数据文件格式
    """
    CSV = '.csv'
    EXCEL = '.xlsx'
    TEXT = '.txt'
    UNKNOWN = 'unknown'


class DataLogging:
    """
    测井数据管理核心类

    功能概述：
    1. 支持多种格式的测井数据读取（CSV、Excel、TXT）
    2. 自动进行曲线名称映射和标准化
    3. 提供数据归一化处理功能
    4. 自动计算测井数据的分辨率
    5. 支持数据与岩性类型的合并处理

    设计原则：
    - 配置外部化：曲线名称映射通过XML文件配置
    - 惰性加载：数据在需要时才进行读取和处理
    - 异常安全：完善的错误处理和数据验证
    - 类型安全：使用类型注解提高代码可读性
    """

    # 类常量定义
    DEFAULT_RESOLUTION = -1.0  # 默认分辨率值（未计算状态）
    CONFIG_FILE_NAME = "COLS_MAPPING.xml"  # 配置文件名称
    DEFAULT_CONFIG_PATH = r"D:\GitHub\zfhlog\src_well_data"  # 默认配置文件路径

    def __init__(self, path: str = '', well_name: str = ''):
        """
        初始化测井数据对象

        Args:
            path: 测井数据文件路径，支持CSV、Excel和TXT格式
            well_name: 井名标识，用于数据标识和日志记录

        Attributes:
            _data: 存储原始测井数据的DataFrame
            _data_normed: 存储归一化后的测井数据
            _data_with_type: 存储测井数据与岩性类型数据的合并结果
            _data_normed_with_type: 存储归一化后的测井数据与岩性类型数据的合并结果
            _curve_names: 测井曲线名称列表
            _file_path: 数据文件路径
            _logging_name: 井名标识
            _resolution: 测井数据分辨率（深度采样间隔）
            mapping_dict: 曲线名称映射字典
            _logger: 日志记录器实例
            _is_data_loaded: 数据加载状态标志
        """
        # 数据存储属性初始化
        self._data: pd.DataFrame = pd.DataFrame()
        self._data_normed: pd.DataFrame = pd.DataFrame()
        self._data_with_type: pd.DataFrame = pd.DataFrame()
        self._data_normed_with_type: pd.DataFrame = pd.DataFrame()

        # 配置和元数据属性
        self._curve_names: List[str] = []
        self._file_path: str = path
        self._logging_name: str = well_name
        self._resolution: float = self.DEFAULT_RESOLUTION
        self._is_data_loaded: bool = False

        # 初始化日志系统
        self._logger = self._setup_logger()

        # 加载曲线名称映射配置
        try:
            self.mapping_dict: Dict[str, List[str]] = self._load_config(self.CONFIG_FILE_NAME)
            self._logger.info(f"成功加载曲线名称映射配置，包含{len(self.mapping_dict)}种曲线类型")
        except Exception as e:
            self._logger.error(f"加载映射配置失败: {e}")
            # 使用默认映射字典作为fallback
            self.mapping_dict = self._get_default_mapping_dict()
            self._logger.warning("使用默认映射字典作为替代")

    def _setup_logger(self) -> logging.Logger:
        """
        设置并配置日志记录器
        Returns:
            配置好的logging.Logger实例
        Note:
            - 每个井使用独立的logger，便于区分不同井的日志
            - 日志格式包含时间、井名、日志级别和消息
        """
        logger = logging.getLogger(f"DataLogging_{self._logging_name}")

        # 避免重复添加handler
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger

    def _get_default_mapping_dict(self) -> Dict[str, List[str]]:
        """
        获取默认的曲线名称映射字典
        Returns:
            默认映射字典，当XML配置文件不可用时使用
        Note:
            这是一个fallback方法，确保即使配置文件缺失也能正常运行
        """
        return {
            'depth': ['#Depth', '#DEPTH', 'Depth', 'DEPTH', 'depth'],
            'CAL': ['CAL', 'CALC', 'CALX', 'CALY'],
            'SP': ['SP', 'Sp'],
            'GR': ['GR', 'GRC'],
            'CNL': ['CNL', 'CN'],
            'DEN': ['DEN', 'DENC'],
            'DT': ['DT', 'DT24', 'DTC', 'AC', 'Ac'],
            'RXO': ['RXO', 'Rxo', 'RXo', 'RS', 'Rs', 'Rlls', 'RLLS'],
            'RD': ['RD', 'Rd', 'Rt', 'RT', 'Rlld', 'RLLD'],
        }

    def _load_config(self, config_name: str) -> Dict[str, List[str]]:
        """
        从XML配置文件加载曲线名称映射字典

        Args:
            config_name: 配置文件名

        Returns:
            曲线名称映射字典，键为标准曲线名，值为别名列表

        Raises:
            DataLoggingException: 当配置文件不存在或格式错误时抛出

        Workflow:
            1. 构建配置文件完整路径
            2. 检查文件是否存在
            3. 解析XML文件结构
            4. 提取曲线类型和别名信息
            5. 构建并返回映射字典
        """
        # 构建配置文件路径
        config_path = os.path.join(self.DEFAULT_CONFIG_PATH, config_name)

        # 检查文件是否存在
        if not os.path.exists(config_path):
            raise DataLoggingException(f"配置文件不存在: {config_path}")

        try:
            # 解析XML文件
            tree = ET.parse(config_path)
            root = tree.getroot()

            # 验证根元素
            if root.tag != "LogAliasMapping":
                raise DataLoggingException(f"XML根元素应为'LogAliasMapping'，实际为'{root.tag}'")

            # 创建映射字典
            mapping_dict = {}

            # 遍历所有LogType元素
            for log_type_elem in root.findall("LogType"):
                # 获取曲线类型名称
                type_name = log_type_elem.get("name")
                if not type_name:
                    self._logger.warning("发现未命名的LogType元素，已跳过")
                    continue

                # 提取所有别名
                aliases = []
                for alias_elem in log_type_elem.findall("Alias"):
                    if alias_elem.text:
                        aliases.append(alias_elem.text)

                if not aliases:
                    self._logger.warning(f"曲线类型'{type_name}'没有定义别名")
                    continue

                # 添加到映射字典
                mapping_dict[type_name] = aliases
                self._logger.debug(f"加载曲线类型: {type_name} -> {aliases}")

            if not mapping_dict:
                raise DataLoggingException("配置文件中未找到有效的曲线类型定义")

            return mapping_dict

        except ET.ParseError as e:
            raise DataLoggingException(f"XML文件解析错误: {e}")
        except Exception as e:
            raise DataLoggingException(f"加载配置文件失败: {e}")

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

    def read_data(self, file_path: str = '', table_name: str = '') -> None:
        """
        读取测井数据文件

        Args:
            file_path: 数据文件路径，为空时使用对象初始化路径
            table_name: 表格名称（主要用于Excel的sheet名）

        Raises:
            DataLoggingException: 文件读取失败或格式不支持时抛出

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
                raise DataLoggingException("未提供文件路径")

            # 检查文件是否存在
            if not os.path.isfile(file_path):
                raise DataLoggingException(f"文件不存在: {file_path}")

            # 检测文件格式
            file_format = self._detect_file_format(file_path)
            self._logger.info(f"检测到文件格式: {file_format.value}")

            # 根据格式读取数据
            if file_format == FileFormat.CSV:
                # CSV文件：尝试多种编码格式读取
                encodings = ['utf-8-sig', 'gbk', 'utf-8', 'latin-1']
                for encoding in encodings:
                    try:
                        self._data = pd.read_csv(file_path, encoding=encoding)
                    except UnicodeDecodeError:
                        continue  # 尝试下一种编码
                if self._data.empty:
                    raise DataLoggingException(f"无法解码CSV文件: {file_path}")

            elif file_format == FileFormat.EXCEL:
                # 确定工作表名称
                sheet_name = table_name or self._logging_name or 0
                self._data = pd.read_excel(file_path, sheet_name=sheet_name)
            elif file_format == FileFormat.TEXT:
                # TXT文件通常为LAS格式，跳过前几行头信息
                self._data = pd.DataFrame(np.loadtxt(file_path, skiprows=8))
                # # 设置默认列名，实际应用中应根据文件内容设置
                # self._data.columns = ['Depth', 'Curve1']  # 简化处理
            else:
                raise DataLoggingException(f"不支持的文件格式: {file_path}")

            # 检查数据是否成功读取
            if self._data.empty:
                raise DataLoggingException("读取到的数据为空")

            # 删除所有数据头columns中的空格，并将所有字符进行大写化
            self._data = self.columns_preprocess(self._data, to_uppercase=True, remove_all_spaces=True)

            # 初始化曲线名称
            self._curve_names = list(self._data.columns)
            self._logger.info(f"成功读取数据，包含{len(self._curve_names)}条曲线")

            # 计算分辨率
            self._resolution = self._calculate_resolution()
            self._logger.info(f"计算得到分辨率: {self._resolution:.6f}米")

            # 更新加载状态
            self._is_data_loaded = True


        except Exception as e:
            self._logger.error(f"读取数据失败: {e}")
            raise

    def columns_preprocess(self, dataframe: Optional[pd.DataFrame] = None,
                                    remove_all_spaces: bool = False,
                                    to_lowercase: bool = False,
                                    to_uppercase: bool = False,
                                    inplace: bool = True) -> pd.DataFrame:
        """
        高级列名清理函数，提供更多清理选项

        Args:
            dataframe: 要处理的DataFrame，为None时处理self._data
            remove_all_spaces: 是否删除所有空格（包括中间空格）
            to_lowercase: 是否转换为小写
            to_uppercase: 是否转换为大写
            inplace: 是否原地修改

        Returns:
            处理后的DataFrame

        Example:
            输入: [' GRC  ', 'cn Log', 'DT-24 ']
            基本清理: ['GRC', 'cn Log', 'DT-24']
            删除所有空格: ['GRC', 'cnLog', 'DT-24']
            转为大写: ['GRC', 'CN LOG', 'DT-24']
        """
        # 确定要处理的数据框
        if dataframe is None:
            dataframe = self._data
            target_df = self._data
        else:
            target_df = dataframe if inplace else dataframe.copy()

        if target_df.empty:
            self._logger.warning("DataFrame为空，跳过列名清理")
            return target_df

        # 保存原始列名
        original_columns = list(target_df.columns)
        cleaned_columns = []

        for col in original_columns:
            # 去除两端空格
            col_clean = col.strip()

            # 🆕 新增：去除开头的下划线
            # 使用lstrip('_')去除所有开头连续的下划线
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
            # 如果同时设置为True，优先使用大写

            cleaned_columns.append(col_clean)

        # 检查列名是否唯一
        if len(cleaned_columns) != len(set(cleaned_columns)):
            duplicates = [col for col in cleaned_columns if cleaned_columns.count(col) > 1]
            self._logger.error(f"列名清理后出现重复: {duplicates}")
            raise DataLoggingException(f"列名清理后出现重复列名: {duplicates}")

        # 应用新的列名
        target_df.columns = cleaned_columns

        # 记录变化
        changes = [(orig, clean) for orig, clean in zip(original_columns, cleaned_columns) if orig != clean]
        if changes:
            self._logger.info(f"高级列名清理完成，修改{len(changes)}个列名")
            for orig, clean in changes[:5]:  # 只显示前5个变化
                self._logger.debug(f"'{orig}' -> '{clean}'")
            if len(changes) > 5:
                self._logger.debug(f"... 还有{len(changes) - 5}个修改")

        # 如果是处理 self._data，需要更新曲线名称列表
        if dataframe is None and hasattr(self, '_curve_names'):
            self._curve_names = cleaned_columns

        return target_df if inplace else target_df

    def _calculate_resolution(self) -> float:
        """
        计算测井数据的分辨率（深度采样间隔）

        Returns:
            计算得到的深度分辨率

        Algorithm:
            1. 提取深度列数据
            2. 计算相邻深度点的差值
            3. 使用众数作为分辨率估计值
        """
        try:
            depth_array = self._data.iloc[:, 0].values
            depth_array = depth_array.ravel()  # 确保为一维数组

            if len(depth_array) < 2:
                self._logger.warning("深度数据点不足，使用默认分辨率")
                return 0.0025
            # 计算深度差值
            depth_diff = np.diff(depth_array)
            # 使用众数作为分辨率估计
            resolution = find_mode(depth_diff)
            return resolution
        except Exception as e:
            self._logger.error(f"计算分辨率失败: {e}")
            return 0.0025  # 返回默认分辨率

    def input_cols_mapping(self, input_cols: List[str], target_cols: List[str]) -> List[str]:
        """
        曲线名称映射函数：将输入的曲线名称别名映射为标准名称

        Args:
            input_cols: 输入的曲线名称列表（可能包含别名）
            target_cols: 目标曲线名称列表（实际存在的曲线名）

        Returns:
            映射后的曲线名称列表

        Algorithm:
            1. 复制输入列表，避免修改原始数据
            2. 找出输入列表中不在目标列表中的曲线（需要映射的曲线）
            3. 从后向前遍历需要映射的曲线（避免索引变化影响）
            4. 在映射字典中查找对应的标准名称
            5. 使用集合交集确定实际存在的标准名称
            6. 替换别名为标准名称
            7. 检查并处理无法映射的曲线

        Example:
            输入: ['GRC', 'CN', 'DT24']
            目标: ['Depth', 'GR', 'CNL', 'AC']
            映射: 'GRC'->'GR', 'CN'->'CNL', 'DT24'->'AC'
            输出: ['Depth', 'GR', 'CNL', 'AC']
        """
        # 步骤1: 复制输入列表，避免修改原始数据
        input_mapping_cols = input_cols.copy()

        # 步骤2: 找出需要映射的曲线（不在目标列表中的曲线）
        unindex_result = [col for col in input_cols if col not in target_cols]

        if not unindex_result:
            self._logger.debug("所有曲线名称均存在，无需映射")
            return input_mapping_cols

        self._logger.debug(f"发现需要映射的曲线: {unindex_result}")

        # 步骤3: 记录已处理的和未处理的曲线
        processed_cols = []
        unprocessed_cols = []

        # 步骤4: 从后向前遍历需要映射的曲线（避免索引变化影响）
        for col_name in reversed(unindex_result):
            try:
                # 获取当前曲线在列表中的索引
                idx = input_mapping_cols.index(col_name)

                # 标记是否找到映射
                mapping_found = False

                # 步骤5: 在映射字典中查找对应的标准名称
                for standard_name, aliases in self.mapping_dict.items():
                    if col_name in aliases:
                        # 步骤6: 使用集合交集确定实际存在的标准名称
                        intersection = list(set(aliases) & set(target_cols))

                        if intersection:
                            # 找到映射，替换别名
                            original_name = input_mapping_cols[idx]
                            input_mapping_cols[idx] = intersection[0]
                            processed_cols.append(col_name)
                            mapping_found = True

                            self._logger.debug(f"曲线映射: {original_name} -> {intersection[0]}")
                            break
                        else:
                            self._logger.warning(f"曲线'{col_name}'的映射目标'{standard_name}'不在目标列表中")

                # 步骤7: 检查是否找到映射
                if not mapping_found:
                    unprocessed_cols.append(col_name)
                    self._logger.warning(f"未找到曲线'{col_name}'的映射规则")

            except ValueError:
                # 曲线名称不在列表中（理论上不会发生）
                unprocessed_cols.append(col_name)
                self._logger.warning(f"曲线'{col_name}'不在输入列表中")

        # 步骤8: 处理无法映射的曲线
        if unprocessed_cols:
            error_msg = f"存在无法处理的曲线: {unprocessed_cols}"
            self._logger.error(error_msg)
            raise DataLoggingException(error_msg)

        self._logger.info(f"曲线映射完成: 处理{len(processed_cols)}条曲线")
        return input_mapping_cols

    def get_data(self, curve_names: Optional[List[str]] = None,
                 depth_delete: Optional[List[float]] = None) -> pd.DataFrame:
        """
        获取测井数据

        Args:
            curve_names: 指定要获取的曲线名称列表，为None时获取所有曲线
            depth_delete: 要删除的深度点列表，为None时不删除

        Returns:
            包含指定曲线的测井数据DataFrame

        Workflow:
            1. 惰性加载：如果数据未加载，先读取数据
            2. 曲线名称处理：确定要获取的曲线列表
            3. 深度列确保：确保深度列包含在结果中
            4. 曲线名称映射：将别名映射为标准名称
            5. 深度过滤：删除指定的深度点（如果提供）
        """
        # 步骤1: 惰性加载数据
        if not self._is_data_loaded:
            self.read_data()

        # 步骤2: 确定曲线名称列表
        if curve_names is None or len(curve_names) == 0:
            curve_names = self._curve_names.copy()

        # 步骤3: 确保深度列包含在结果中
        depth_col = self._find_depth_column(curve_names)
        if not depth_col and self._curve_names:
            curve_names = [self._curve_names[0]] + curve_names
            self._logger.debug(f"自动添加深度列: {self._curve_names[0]}")

        # 步骤4: 曲线名称映射
        mapped_curve_names = self.input_cols_mapping(curve_names, self._curve_names)

        # 验证映射后的曲线名称是否存在
        missing_curves = set(mapped_curve_names) - set(self._curve_names)
        if missing_curves:
            raise DataLoggingException(f"曲线不存在: {missing_curves}")

        # 步骤5: 深度过滤
        if depth_delete:
            data_result = self._data[mapped_curve_names].copy()
            # 删除指定的深度点
            depth_col_name = mapped_curve_names[0]  # 假设第一列为深度
            mask = ~data_result[depth_col_name].isin(depth_delete)
            data_result = data_result[mask]
            self._logger.info(f"删除了{len(depth_delete) - mask.sum()}个深度点")
        else:
            data_result = self._data[mapped_curve_names]

        return data_result

    def _find_depth_column(self, curve_names: List[str]) -> Optional[str]:
        """
        查找深度列名称

        Args:
            curve_names: 曲线名称列表

        Returns:
            深度列名称，如果未找到返回None
        """
        # 查找标准深度列名称
        depth_aliases = self.mapping_dict.get('depth', [])
        for col in curve_names:
            if col in depth_aliases:
                return col
        return None

    def get_data_normed(self, curve_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        获取归一化后的测井数据

        Args:
            curve_names: 指定要获取的曲线名称列表
        Returns:
            归一化后的测井数据DataFrame
        Workflow:
            1. 如果归一化数据为空，进行归一化处理
            2. 如果指定了曲线名称，检查是否已存在
            3. 如果不存在，重新进行归一化处理
        """
        # 步骤1: 获取原始数据
        data_target = self.get_data(curve_names)

        # 步骤2: 检查是否需要重新归一化
        need_renormalize = False
        if self._data_normed.empty:
            need_renormalize = True
        elif curve_names and not set(curve_names).issubset(set(self._data_normed.columns)):
            need_renormalize = True
            self._logger.debug("指定曲线未归一化，重新进行归一化处理")

        # 步骤3: 归一化处理
        if need_renormalize:
            cols_new_temp = list(data_target.columns)
            data_normed_temp = data_Normalized(
                data_target.values,
                DEPTH_USE=True,
                logging_range=[-999, 9999],
                max_ratio=0.01,
                local_normalized=False
            )
            self._data_normed = pd.DataFrame(data_normed_temp, columns=cols_new_temp)
            self._logger.info("完成数据归一化处理")

        return self._data_normed

        # # 如果归一化数据为零，则初始化归一化数据
        # if self._data_normed.empty:
        #     data_target = self.get_data(curve_names)
        #     cols_new_temp = list(data_target.columns)
        #     data_normed_temp = data_Normalized(data_target.values, DEPTH_USE=True, logging_range=[-999, 9999],
        #                                        max_ratio=0.01, local_normalized=False)
        #     self._data_normed = pd.DataFrame(data_normed_temp, columns=cols_new_temp)
        # else:
        #     # 如果 self._data_normed 中含有 curve_names，则直接返回
        #     if set(curve_names) < set(list(self._data_normed.columns)):
        #         return self._data_normed[curve_names]
        #     # 否则的话，重新进行正则化
        #     else:
        #         data_target = self.get_data(curve_names)
        #         cols_new_temp = list(data_target.columns)
        #         data_normed_temp = data_Normalized(data_target.values, DEPTH_USE=True, logging_range=[-999, 9999],
        #                                            max_ratio=0.01, local_normalized=False)
        #         self._data_normed = pd.DataFrame(data_normed_temp, columns=cols_new_temp)
        #
        # return self._data_normed

    def get_curve_names(self) -> List[str]:
        """
        获取所有曲线名称

        Returns:
            曲线名称列表
        """
        if not self._curve_names:
            self.read_data()
        return self._curve_names.copy()

    def get_summary(self) -> Dict[str, any]:
        """
        获取数据摘要信息

        Returns:
            包含各类统计信息的字典
        """
        return {
            'well_name': self._logging_name,
            'file_path': self._file_path,
            'resolution': self._resolution,
            'is_loaded': self._is_data_loaded,
            'curve_count': len(self._curve_names),
            'data_shape': self._data.shape if not self._data.empty else (0, 0),
            'mapping_dict_size': len(self.mapping_dict)
        }


# 测试代码
if __name__ == '__main__':
    # 创建测试实例
    test_data = DataLogging(
        # path=r'F:\logging_workspace\FY1-15\FY1-15_texture_logging_data.csv',
        path=r'F:\logging_workspace\FY1-15\FY1-15_logging_data.csv',
        well_name='FY1-15'
    )

    # 测试数据获取
    print("=== 原始数据统计 ===")
    print(test_data.get_data().describe())

    print("\n=== 映射字典内容 ===")
    print(test_data.mapping_dict)

    print("\n=== 曲线名称列表 ===")
    print(test_data.get_curve_names())

    print("\n=== 分辨率信息 ===")
    print(test_data._resolution)

    print("\n=== 数据摘要 ===")
    summary = test_data.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

    # 测试曲线名称映射
    print("\n=== 曲线名称映射测试 ===")
    # test_curves = ['GRC', 'CN', 'DT', 'SP']
    test_curves = []
    try:
        print(f"原始曲线: {test_curves}")
        data_logging = test_data.get_data(curve_names=test_curves)
        print(f"映射后曲线: {data_logging.columns}")
        # mapped_curves = test_data.input_cols_mapping(test_curves, test_data.get_curve_names())
    except Exception as e:
        print(f"映射测试失败: {e}")