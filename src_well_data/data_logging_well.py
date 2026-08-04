import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from src_data_process.data_depth_delete import process_depth_segment
from src_data_process.data_linear_regression import MultiVariateLinearRegressor
from src_file_op.dir_operation import search_files_by_criteria
from src_logging.logging_combine import combine_logging_table
from src_well_data.data_logging_FMI import DataFMI
from src_well_data.data_logging_normal import DataLogging
from src_well_data.data_logging_table import DataTable
from src_well_data.data_logging_NMR import DataNMR
from src_well_data.data_logging_core import DataCore


class DATA_WELL:
    """
    井数据统一管理器（Facade 门面模式）：
    - 日常曲线测井数据 DataLogging
    - 电成像 FMI DataFMI
    - 表格类型数据 DataTable
    - 岩心实验数据 DataCore
    - 未来拓展 NMR 数据

    设计原则：
    - 惰性加载：各子模块实例在首次访问时才创建，同一路径不重复实例化
    - 统一访问：通过 get_* 系列方法以相同接口访问不同数据类型
    - 文件扫描：构造函数自动扫描井目录，识别各类数据文件路径
    """

    # =============== 基础初始化 ==================
    def __init__(self, path_folder: str = '', WELL_NAME: str = ''):

        # ---- 数据容器（惰性单例模式） ----
        self.logging_dict: Dict[str, DataLogging] = {}   # 常规测井曲线
        self.table_dict: Dict[str, DataTable] = {}       # 岩性类型表
        self.FMI_dict: Dict[str, DataFMI] = {}           # 电成像数据
        self.NMR_dict: Dict[str, Any] = {}               # 核磁共振数据
        self.core_dict: Dict[str, DataCore] = {}          # 岩心实验数据

        # ---- 路径容器（由 scan_files() 扫描填充） ----
        self.path_list_logging: List[str] = []
        self.path_list_table: List[str] = []
        self.path_list_fmi: List[str] = []
        self.path_list_nmr: List[str] = []
        self.path_list_core: List[str] = []             # 岩心实验数据文件路径列表

        # 根路径
        self.well_path = path_folder

        # ---- 井名判定 ----
        if WELL_NAME:
            self.WELL_NAME = WELL_NAME
        else:
            self.WELL_NAME = os.path.basename(path_folder)

        # ---- 文件识别关键字 ----
        self.LOGGING_KW = ['logging']
        self.TABLE_KW = ['table', 'LITHO_TYPE']
        self.FMI_KW = ['DYNA', 'STAT']
        self.NMR_KW = ['NMR']
        self.CORE_KW = ['core']                         # 岩心数据文件关键字（文件名需含 'core'）

        # 初始化路径扫描
        self.scan_files()

    # =========================================================================
    #                          文件扫描模块
    # =========================================================================
    def scan_files(self):
        """扫描井目录，识别各类文件路径"""
        if not os.path.exists(self.well_path):
            print(f"[WARN] 路径不存在: {self.well_path}")
            return

        # 常规测井曲线文件（.xlsx / .csv）
        self.path_list_logging = search_files_by_criteria(
            self.well_path,
            name_keywords=self.LOGGING_KW,
            file_extensions=['.xlsx', '.csv'],
            all_keywords=False
        )

        # 岩性类型表文件（.xlsx / .csv）
        self.path_list_table = search_files_by_criteria(
            self.well_path,
            name_keywords=self.TABLE_KW,
            file_extensions=['.xlsx', '.csv'],
            all_keywords=False
        )

        # 电成像 FMI 文件（.txt）
        self.path_list_fmi = search_files_by_criteria(
            self.well_path,
            name_keywords=self.FMI_KW,
            file_extensions=['.txt'],
            all_keywords=False
        )

        # 核磁共振 NMR 文件（.csv / .txt）
        self.path_list_nmr = search_files_by_criteria(
            self.well_path,
            name_keywords=self.NMR_KW,
            file_extensions=['.csv', '.txt'],
            all_keywords=False
        )

        # 岩心实验数据文件（.csv / .xlsx / .txt）
        self.path_list_core = search_files_by_criteria(
            self.well_path,
            name_keywords=self.CORE_KW,
            file_extensions=['.csv', '.xlsx', '.txt'],
            all_keywords=False
        )

    # =========================================================================
    #                          内部辅助函数
    # =========================================================================
    def _get_default_obj(self, data_dict: Dict, key: str = ''):
        """
        字典数据获取辅助函数

        优先级规则：
        1. dict 为空  → 返回 None
        2. key 为空    → 返回第一个对象
        3. key 模糊匹配 → 返回匹配到的第一个对象
        4. 无匹配       → 返回 None
        """
        if not data_dict:
            print("\033[33m[WARN] 数据未初始化\033[0m")
            return None

        if not key:
            return next(iter(data_dict.values()))  # 返回第一个对象

        # 支持模糊匹配（key 是文件名子串即可）
        for k in data_dict.keys():
            if key in k:
                return data_dict[k]

        # 完全匹配失败
        return None


    def search_file_path_list(self, name_keywords=[], file_extensions=['.csv', '.xlsx', '.txt']):
        """
        按关键字精确搜索常规测井文件路径（AND 匹配）

        Args:
            name_keywords: 搜索关键字列表，所有关键字均需出现在文件名中

        Returns:
            符合条件的文件路径列表
        """
        path_list_logging = search_files_by_criteria(
            self.well_path,
            name_keywords=name_keywords,
            file_extensions=file_extensions,
            all_keywords=True
        )
        return path_list_logging

    # =========================================================================
    #                          数据初始化模块
    # =========================================================================
    def init_logging(self, path: str = ''):
        """
        初始化常规测井数据对象（惰性单例模式）

        Args:
            path: 测井数据文件路径，为空时自动取 scan_files() 扫描到的第一个文件
        """
        if not path:
            if not self.path_list_logging:
                return
            path = self.path_list_logging[0]

        if path not in self.logging_dict:
            self.logging_dict[path] = DataLogging(path=path, well_name=self.WELL_NAME)

    def init_table(self, path: str = ''):
        """
        初始化岩性类型表对象（惰性单例模式）

        Args:
            path: 类型表文件路径，为空时自动取 scan_files() 扫描到的第一个文件
        """
        if not path:
            if not self.path_list_table:
                return
            path = self.path_list_table[0]

        if path not in self.table_dict:
            self.table_dict[path] = DataTable(path=path, well_name=self.WELL_NAME)

    def init_FMI(self, path: str = ''):
        """
        初始化电成像 FMI 数据对象（惰性单例模式）

        Args:
            path: FMI 文件路径，为空时自动取 scan_files() 扫描到的第一个文件
        """
        if not path:
            if not self.path_list_fmi:
                return
            path = self.path_list_fmi[0]

        if path not in self.FMI_dict:
            self.FMI_dict[path] = DataFMI(path_fmi=path)

    def init_NMR(self, path: str = ''):
        """
        初始化核磁共振 NMR 数据对象（惰性单例模式）

        Args:
            path: NMR 文件路径，为空时自动取 scan_files() 扫描到的第一个文件
        """
        if not path:
            if not self.path_list_nmr:
                return
            path = self.path_list_nmr[0]

        if path not in self.NMR_dict:
            self.NMR_dict[path] = DataNMR(path_nmr=path)

    def init_core(self, path: str = ''):
        """
        初始化岩心实验数据对象（惰性单例模式）

        Args:
            path: 岩心数据文件路径，为空时自动取 scan_files() 扫描到的第一个文件

        Note:
            - 同一路径不会重复创建实例
            - 支持 .csv / .xlsx / .txt 三种格式
            - 文件名需包含 'core' 关键字（由 CORE_KW 控制）
        """
        if not path:
            if not self.path_list_core:
                print("\033[33m[WARN] 未扫描到岩心数据文件\033[0m")
                return
            path = self.path_list_core[0]

        if path not in self.core_dict:
            self.core_dict[path] = DataCore(path=path, well_name=self.WELL_NAME)

    # =========================================================================
    #                          统一访问接口
    # =========================================================================
    def get_logging(self, key: str = '',
                    curve_names: List[str] = None,
                    norm: bool = False,
                    depth_limit: List[float] = []):
        """
        获取常规测井数据

        Args:
            key: 文件路径或关键字，'' → 取第一个扫描到的文件
            curve_names: 指定要获取的曲线列表，None → 获取所有曲线
            norm: 是否返回归一化后的数据
            depth_limit: 深度限制 [min_depth, max_depth]

        Returns:
            测井数据 DataFrame
        """
        self.init_logging(key)
        obj = self._get_default_obj(self.logging_dict, key)
        if obj is None:
            return pd.DataFrame()
        df_logging = obj.get_data_normed(curve_names) if norm else obj.get_data(curve_names)
        if depth_limit:
            df_logging = process_depth_segment(
                df=df_logging,
                depth_config=[depth_limit],
                drop=False
            )

        return df_logging

    def get_table(self, key: str = '', mode='3', replaced=False, replace_dict=None, new_col='Type_Replaced'):
        """
        获取岩性类型表数据

        Args:
            key: 文件路径或关键字，'' → 取第一个扫描到的文件
            mode: '3' 返回三列格式 (depth_start, depth_end, type)，
                  '2' 返回两列格式 (depth, type)
            replaced: 是否应用类型替换
            replace_dict: 类型替换字典
            new_col: 替换后的新列名
        """
        self.init_table(key)
        obj = self._get_default_obj(self.table_dict, key)
        if obj is None:
            return pd.DataFrame()

        if replaced and replace_dict:
            obj._apply_type_replacement(replace_dict=new_col)

        return obj.get_table_3() if mode == '3' else obj.get_table_2()

    def get_FMI(self, key: str = '', depth: Optional[List[float]] = None):
        """
        获取电成像 FMI 数据

        Args:
            key: 文件路径或关键字，'' → 取第一个扫描到的文件
            depth: 深度范围 [min_depth, max_depth]，None → 不限制

        Returns:
            FMI 图像数据（numpy array）和深度数组
        """
        self.init_FMI(key)
        obj = self._get_default_obj(self.FMI_dict, key)
        if obj is None:
            return None
        return obj.get_data(depth)

    def get_NMR(self, key: str = '', depth: Optional[List[float]] = None):
        """
        获取核磁共振 NMR 数据

        Args:
            key: 文件路径或关键字，'' → 取第一个扫描到的文件
            depth: 深度范围 [min_depth, max_depth]，None → 不限制
        """
        self.init_NMR(key)
        obj = self._get_default_obj(self.NMR_dict, key)
        if obj is None:
            return None
        return obj.get_data(depth)

    def get_core(self, key: str = '', curve_names: Optional[List[str]] = None,
                 depth_range: Optional[List[float]] = None) -> pd.DataFrame:
        """
        获取岩心实验数据（DataFrame 格式）

        Args:
            key: 文件路径或关键字，'' → 取第一个扫描到的岩心文件
            curve_names: 指定要获取的列名列表，None → 获取所有列
                       典型列名: ['DEPTH', '石英', '钾长石', '斜长石', '黄铁矿', '黏土矿物']
            depth_range: 深度范围 [min_depth, max_depth]，None → 不限制

        Returns:
            包含指定列和深度范围的岩心数据 DataFrame

        Note:
            - 数据稀疏：深度点不连续，采样间隔约 1m，远大于测井分辨率
            - 支持多矿物/多组分列同时获取
        """
        self.init_core(key)
        obj = self._get_default_obj(self.core_dict, key)
        if obj is None:
            return pd.DataFrame()
        return obj.get_data(curve_names=curve_names, depth_range=depth_range)

    def get_FMI_texture(self, key: str = '', texture_config: Optional[Dict] = None):
        """
        获得 FMI 电成像数据的纹理特征数据

        Args:
            key: 文件路径或关键字
            texture_config: 纹理计算配置字典
        """
        self.init_FMI(key)
        obj = self._get_default_obj(self.FMI_dict, key)
        if obj is None:
            return None
        texture = obj.get_texture(texture_config, fmi_texture_path='')
        return texture

    def get_FMI_textures(self, texture_config: Optional[Dict] = None, path_config={}):
        """
        获取动静态电成像的合并纹理特征数据

        优先从缓存文件读取；缓存不存在时重新计算并保存

        Args:
            texture_config: 纹理计算配置（level, distance, angles, windows_length, windows_step）
            path_config: 路径配置（可选，包含 path_dyna 和 path_stat）
        """
        path_texture_all = self.well_path + f'\\{self.WELL_NAME}_texture_logging_{texture_config["windows_length"]}.csv'

        # 缓存命中：直接读取
        if os.path.exists(path_texture_all):
            print('纹理文件已存在，直接进行读取', path_texture_all)
            return pd.read_csv(path_texture_all)

        # 缓存未命中：计算动静态纹理并合并
        if 'path_dyna' in path_config:
            path_dyna = path_config['path_dyna']
            if path_dyna not in self.path_list_fmi:
                raise FileNotFoundError("file {} not found".format(path_dyna))
        else:
            path_dyna = self.search_file_path_list(name_keywords=[self.FMI_KW[0]])[0]

        if 'path_stat' in path_config:
            path_stat = path_config['path_stat']
            if path_stat not in self.path_list_fmi:
                raise FileNotFoundError("file {} not found".format(path_stat))
        else:
            path_stat = self.search_file_path_list(name_keywords=[self.FMI_KW[1]])[0]

        texture_dyna = self.get_FMI_texture(key=path_dyna, texture_config=texture_config)
        texture_stat = self.get_FMI_texture(key=path_stat, texture_config=texture_config)

        TEXTURE_ALL = pd.concat([texture_stat, texture_dyna.iloc[:, 1:]], axis=1)
        print('saving all textures to file:', path_texture_all)
        TEXTURE_ALL.to_csv(path_texture_all, index=False)
        return TEXTURE_ALL

    def get_FMI_fde(self, key: str = '', fde_config: Optional[Dict] = None):
        """
        获取指定 FMI 文件的分形维数谱（FDE）数据

        Args:
            key: 文件路径或关键字
            fde_config: FDE 计算配置字典
        """
        self.init_FMI(key)
        obj = self._get_default_obj(self.FMI_dict, key)
        if obj is None:
            return None
        fmi_fde = obj.get_fmi_fde(config_fde=fde_config)
        return fmi_fde

    def get_FMI_fdes(self, fde_config: Optional[Dict] = None):
        """
        获取动静态电成像的 FDE 数据（tuple 格式）

        Args:
            fde_config: FDE 计算配置字典

        Returns:
            (fde_dyna, fde_stat) 元组
        """
        path_dyna_list = self.search_file_path_list(name_keywords=[self.FMI_KW[0]])
        path_stat_list = self.search_file_path_list(name_keywords=[self.FMI_KW[1]])
        path_dyna = path_dyna_list[0] if path_dyna_list else ''
        path_stat = path_stat_list[0] if path_stat_list else ''
        fde_dyna = self.get_FMI_fde(key=path_dyna, fde_config=fde_config)
        fde_stat = self.get_FMI_fde(key=path_stat, fde_config=fde_config)

        return fde_dyna, fde_stat


    # =========================================================================
    #                          数据概览接口
    # =========================================================================
    def well_summary(self) -> Dict[str, Any]:
        """
        获取井数据总览信息

        Returns:
            包含井名、路径、各类文件数量和路径列表的字典
        """
        return {
            "well": self.WELL_NAME,
            "path": self.well_path,
            "paths_logging": self.path_list_logging,
            "paths_fmi": self.path_list_fmi,
            "paths_table": self.path_list_table,
            "paths_nmr": self.path_list_nmr,
            "paths_core": self.path_list_core,
            "logging_files_num": len(self.path_list_logging),
            "fmi_files_num": len(self.path_list_fmi),
            "table_files_num": len(self.path_list_table),
            "nmr_files_num": len(self.path_list_nmr),
            "core_files_num": len(self.path_list_core),
        }

    def __repr__(self):
        return (f"<DATA_WELL {self.WELL_NAME} | "
                f"logging={len(self.logging_dict)}, "
                f"fmi={len(self.FMI_dict)}, "
                f"table={len(self.table_dict)}, "
                f"core={len(self.core_dict)}>")

    # =========================================================================
    #                          数据综合获取函数
    # =========================================================================
    def combine_logging_table(
            self,
            logging_key='',
            curve_names_logging=None,
            table_key='',
            replace_dict=None,
            new_col='Type',
            norm=False,
            depth_limit: Optional[List[float]] = None,
    ):
        """
        将连续曲线 logging 与类型表（3列或2列）合并

        生成 (depth + curves + lithology_label) 格式的 DataFrame，
        常用于后续的岩性分类或相关性分析

        Args:
            logging_key: 测井数据文件路径或关键字
            curve_names_logging: 要保留的曲线列名列表
            table_key: 类型表文件路径或关键字
            replace_dict: 类型替换字典
            new_col: 替换后的新列名
            norm: 是否对测井曲线归一化
            tolerance: 深度合并容差（米）

        Returns:
            合并后的 DataFrame (depth + curves + lithology_label)
        """
        # 1 获取曲线数据
        df_log = self.get_logging(
            key=logging_key,
            curve_names=curve_names_logging,
            norm=norm,
            depth_limit=depth_limit,
        )
        depth_col = df_log.columns[0]

        # 2 获取类型表
        self.init_table(table_key)
        table_obj = self._get_default_obj(self.table_dict, table_key)

        if replace_dict:
            table_obj._apply_type_replacement(replace_dict=replace_dict, new_col=new_col)

        df_tab = table_obj.get_table_2_replaced()

        # 排序
        df_log = df_log.sort_values(depth_col)
        df_tab = df_tab.sort_values(df_tab.columns[0])

        logging_columns = list(df_log.columns)
        table_columns = list(df_tab.columns)
        array_logging = df_log.values.astype(np.float32)
        array_table = df_tab.values.astype(np.float32)
        array_merge = combine_logging_table(array_logging, array_table, drop=False, tolerance=table_obj._table_resolution+0.001)

        data_columns = logging_columns + [table_columns[-1]]
        df_merge = pd.DataFrame(array_merge, columns=data_columns)
        df_merge.dropna(inplace=True)
        df_merge[table_columns[-1]] = df_merge[table_columns[-1]].astype(int)

        if new_col != '' or new_col is not None:
            df_merge.rename(columns={table_columns[-1]: new_col}, inplace=True)

        return df_merge

    def combine_logging_core(
            self,
            logging_key: str = '',
            curve_names_logging: Optional[List[str]] = None,
            norm: bool = False,
            tolerance: float = 0,
            core_key: str = '',
            curve_names_core: Optional[List[str]] = None,
            depth_limit: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        将 logging、table、core 三类数据按深度最近邻合并

        以 logging 深度轴为主基准，通过 cKDTree 最近邻匹配将 table 和 core
        的数据合并进来（table 和 core 可二选一、同时存在或都不传）。

        Args:
            logging_key : 测井数据文件路径或关键字，'' → 取 scan 到的第一个文件
            curve_names_logging : 要保留的 logging 曲线列名列表，None → 全部曲线
            norm : 是否对测井曲线做归一化
            tolerance : 深度合并容差（米），≤0 → 自动计算（logging 分辨率/2 + 0.001）
            core_key : 岩心数据文件路径或关键字，'' → 跳过 core 合并
            curve_names_core : 要保留的 core 列名列表，None → 全部列
            depth_limit : 全局深度限制 [min_depth, max_depth]，None → 不限制

        Returns:
            合并后的 DataFrame (depth + logging_curves + table_label + core_curves)
            无匹配处留 NaN（保留全部 logging 数据行）。

        Note:
            - core 数据为稀疏采样（间隔约 1m），合并后大部分行 core 列仍为 NaN，
              这正是设计预期，调用方可在 NaN 处做插值或直接丢弃。
            - 若 table_key 和 core_key 均为 ''，退化为普通 logging 获取
              （含 curve_names_logging / norm / depth_limit 过滤）。
        """
        # =================================================================
        # Step 1: 获取 logging 数据（主基准）
        # =================================================================
        df_log = self.get_logging(
            key=logging_key,
            curve_names=curve_names_logging,
            norm=norm,
            depth_limit=depth_limit if depth_limit else []
        )
        if df_log.empty:
            print('[WARN] logging 数据为空，返回空 DataFrame')
            return pd.DataFrame()

        depth_col = df_log.columns[0]
        df_log = df_log.sort_values(depth_col).reset_index(drop=True)

        df_current = df_log.copy()


        # =================================================================
        # Step 3: 合并 core（如有）
        # =================================================================
        df_core = self.get_core(
            key=core_key,
            curve_names=curve_names_core,
            depth_range=depth_limit
        )
        if df_core.empty:
            print('[WARN] core 数据为空，跳过 core 合并')
        else:
            # core DataFrame 深度列名通常为 'DEPTH'（列名清理后统一）
            core_depth_col = df_core.columns[0]
            df_core = df_core.sort_values(core_depth_col).reset_index(drop=True)

            # core 的曲线列（不含深度列）
            core_curve_cols = [c for c in df_core.columns if c != core_depth_col]

            arr_log2 = df_current.values.astype(np.float32)
            arr_core = df_core.values.astype(np.float32)
            arr_merged2 = combine_logging_table(
                arr_log2, arr_core, drop=False, tolerance=tolerance
            )
            # 合并后的列名：logging 列 + core 曲线列（深度列已含在 logging 中，不重复）
            log_cols2 = list(df_current.columns)
            core_out_cols = log_cols2 + core_curve_cols
            df_current = pd.DataFrame(arr_merged2, columns=core_out_cols)
            df_current.dropna(subset=[depth_col], inplace=True)
            print(f'[INFO] core 合并完成，最终 DataFrame shape: {df_current.shape}')
            print(f'[INFO] core 列: {core_curve_cols}')

        return df_current

    def combine_logging_table_core(
            self,
            logging_key: str = '',
            curve_names_logging: Optional[List[str]] = None,
            table_key: str = '',
            replace_dict: Optional[Dict] = None,
            new_col: str = 'Type',
            norm: bool = False,
            tolerance: float = 0,
            core_key: str = '',
            curve_names_core: Optional[List[str]] = None,
            depth_limit: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """
        将 logging、table、core 三类数据按深度最近邻合并

        以 logging 深度轴为主基准，通过 cKDTree 最近邻匹配将 table 和 core
        的数据合并进来（table 和 core 可二选一、同时存在或都不传）。

        Args:
            logging_key : 测井数据文件路径或关键字，'' → 取 scan 到的第一个文件
            curve_names_logging : 要保留的 logging 曲线列名列表，None → 全部曲线
            table_key : 类型表文件路径或关键字，'' → 跳过 table 合并
            replace_dict : 类型替换字典，None → 不做类型替换
            new_col : 替换/重命名后的标签列名
            norm : 是否对测井曲线做归一化
            tolerance : 深度合并容差（米），≤0 → 自动计算（logging 分辨率/2 + 0.001）
            core_key : 岩心数据文件路径或关键字，'' → 跳过 core 合并
            curve_names_core : 要保留的 core 列名列表，None → 全部列
            depth_limit : 全局深度限制 [min_depth, max_depth]，None → 不限制

        Returns:
            合并后的 DataFrame (depth + logging_curves + table_label + core_curves)
            无匹配处留 NaN（保留全部 logging 数据行）。

        Note:
            - core 数据为稀疏采样（间隔约 1m），合并后大部分行 core 列仍为 NaN，
              这正是设计预期，调用方可在 NaN 处做插值或直接丢弃。
            - 若 table_key 和 core_key 均为 ''，退化为普通 logging 获取
              （含 curve_names_logging / norm / depth_limit 过滤）。
        """
        # =================================================================
        # Step 1: 获取 logging 数据（主基准）
        # =================================================================
        df_log = self.get_logging(
            key=logging_key,
            curve_names=curve_names_logging,
            norm=norm,
            depth_limit=depth_limit if depth_limit else []
        )
        if df_log.empty:
            print('[WARN] logging 数据为空，返回空 DataFrame')
            return pd.DataFrame()

        depth_col = df_log.columns[0]
        df_log = df_log.sort_values(depth_col).reset_index(drop=True)

        # 若 table 和 core 均未指定，直接返回过滤后的 logging
        if not table_key and not core_key:
            print(f'[INFO] 未指定 table 和 core，直接返回 logging 数据 ({df_log.shape[0]} 行)')
            return df_log

        df_current = df_log.copy()

        # =================================================================
        # Step 2: 合并 table（如有）
        # =================================================================
        if table_key:
            df_current_logging = self.combine_logging_table(
                logging_key=logging_key,
                curve_names_logging=curve_names_logging,
                table_key=table_key,
                replace_dict=replace_dict,
                new_col=new_col,
                norm=norm,
                depth_limit=depth_limit,
            )

        # =================================================================
        # Step 3: 合并 core（如有）
        # =================================================================
        if core_key:
            df_current_core = self.combine_logging_core(
                logging_key=logging_key,
                curve_names_logging=curve_names_logging,
                core_key=core_key,
                depth_limit=depth_limit,
                tolerance=tolerance,
                curve_names_core = curve_names_core if curve_names_core else []
            )


        if table_key and core_key:
            cols_logging = set(list(df_current_logging.columns))
            cols_core = set(list(df_current_core.columns))
            # common_cols = list(cols_logging | cols_core)
            key_cols = list(cols_logging & cols_core)  # 交集作为连接键
            # print("并集列:", common_cols)
            # df_current = pd.merge(df_current_logging, df_current_core, on=common_cols, how='inner')
            df_current = pd.merge(df_current_logging, df_current_core, on=key_cols, how='outer')
        elif table_key and not core_key:
            df_current = df_current_logging
        elif not table_key and core_key:
            df_current = df_current_core
        else:
            print('NO KEY AVAILABLE, ERROR')
            exit(0)

        return df_current

    # =========================================================================
    #                          表格数据的replace_dict获取
    # =========================================================================
    def get_table_replace_dict(self, table_key=''):
        """获取类型表的替换字典"""
        self.init_table(table_key)
        table_obj = self._get_default_obj(self.table_dict, table_key)
        return table_obj.get_replace_dict()





# =========================================================================
#                              测试代码
# =========================================================================
if __name__ == '__main__':
    well = DATA_WELL(path_folder=r'Z:\logging_workspace\姬119H2')

    # =====================================================================
    # 1. 文件扫描与总览
    # =====================================================================
    print("\n" + "=" * 70)
    print("【1】well_summary() — 井数据总览")
    print("=" * 70)
    summary = well.well_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # =====================================================================
    # 3. search_*_path_list 系列 — 关键字精确搜索
    # =====================================================================
    print("\n" + "=" * 70)
    print("【3】search_*_path_list 系列 — 关键字精确搜索")
    print("=" * 70)
    p_logging = well.search_file_path_list(name_keywords=['姬119H2', '常规测井'])
    p_table = well.search_file_path_list(name_keywords=['table'])
    p_dyna = well.search_file_path_list(name_keywords=['DYNA'])
    p_stat = well.search_file_path_list(name_keywords=['STAT'])
    p_nmr = well.search_file_path_list(name_keywords=['NMR'])
    p_core = well.search_file_path_list(name_keywords=['CORE'])

    print("search_logging_path_list(['姬119H2','常规测井']):", p_logging)
    print("search_table_path_list(['table']):", p_table)
    print("search_fmi_path_list(['DYNA']):", p_dyna)
    print("search_fmi_path_list(['STAT']):", p_stat)
    print("search_nmr_path_list(['NMR']):", p_nmr)
    print("search_core_path_list(['CORE']):", p_core)

    # =====================================================================
    # 6. get_table — 岩性类型表（mode='2' 通常更稳定）
    # =====================================================================
    print("\n" + "=" * 70)
    print("【6】get_table — 岩性类型表")
    print("=" * 70)
    try:
        df_tab2 = well.get_table(key=p_table[0])
        print(f"  mode='2' Shape: {df_tab2.shape}")
        print("  Head:\n" + df_tab2.head(5).to_string())
    except Exception as e:
        print(f"  mode='2' FAIL: {e}")
    try:
        df_tab3 = well.get_table(key=p_table[1])
        print(f"  mode='3' Shape: {df_tab3.shape}")
        print("  Head:\n" + df_tab3.head(5).to_string())
    except Exception as e:
        print(f"  mode='3' FAIL: {e}")

    try:
        rep_dict = well.get_table_replace_dict(table_key=p_table[0])
        print("  get_table_replace_dict():", rep_dict)
    except Exception as e:
        print(f"  get_table_replace_dict() FAIL: {e}")

    # =====================================================================
    # 11. combine_logging_table — logging + table 合并
    # =====================================================================
    print("\n" + "=" * 70)
    print("【11】combine_logging_table — logging + table 合并")
    print("=" * 70)
    try:
        df_merged_tab = well.combine_logging_table(
            logging_key=p_logging[0],
            curve_names_logging=['#DEPTH', 'GR', 'DEN', 'CNL'],
            table_key=p_table[0],
            new_col='Type',
        )
        print(f"  Shape: {df_merged_tab.shape}")
        print(f"  Columns: {list(df_merged_tab.columns)}")
        print("  Head:\n" + df_merged_tab.head(5).to_string())
        print("  Describe:\n" + df_merged_tab.describe().to_string())
    except Exception as e:
        print(f"  FAIL: {e}")

    # =====================================================================
    # 12. combine_logging_tables — logging + core 合并
    # =====================================================================
    print("\n" + "=" * 70)
    print("【12】combine_logging_tables — logging + core 合并")
    print("=" * 70)
    try:
        df_merged_core = well.combine_logging_core(
            logging_key=p_logging[0] if p_logging else '',
            curve_names_logging=['#DEPTH', 'GR', 'DEN', 'CNL'],
            core_key=p_core[0] if p_core else '',
            # depth_limit=[2726, 2760],
            tolerance=0.5,
        )
        print(f"  Shape: {df_merged_core.shape}")
        print(f"  Columns: {list(df_merged_core.columns)}")
        matched = df_merged_core.dropna(subset=['石英'])
        print(f"  有矿物数据的行: {len(matched)} / {len(df_merged_core)}")
        print("  矿物数据示例:")
        print(matched[['#DEPTH', 'GR', '石英', '钾长石', '黄铁矿']].head(5).to_string(index=False))
        print("  Describe:\n" + df_merged_core.describe().to_string())
    except Exception as e:
        print(f"  FAIL: {e}")

    # =====================================================================
    # 13. combine_logging_tables — 仅 logging（退化场景）
    # =====================================================================
    print("\n" + "=" * 70)
    print("【13】combine_logging_tables — 所有数据齐全")
    print("=" * 70)
    df_log_table_core = well.combine_logging_table_core(
        logging_key=p_logging[0] if p_logging else '',
        curve_names_logging=['#DEPTH', 'GR', 'DEN'],
        # depth_limit=[2730, 2745],
        table_key=p_table[0],
        core_key=p_core[0],
        tolerance=0
    )
    df_log_table_core_dropna = df_log_table_core.dropna()
    print(f"  Shape: {df_log_table_core_dropna.shape}")
    print(df_log_table_core_dropna.head(9).to_string())
    print(df_log_table_core_dropna.describe().to_string())
