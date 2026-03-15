import logging
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from src_data_process.data_correction_analysis import feature_influence_analysis
from src_data_process.data_depth_delete import process_depth_segment
from src_data_process.data_dilute import dilute_dataframe
from src_data_process.data_linear_regression import calculate_predictions
from src_data_process.data_linear_regression_2 import MultiVariateLinearRegressor
from src_file_op.dir_operation import search_files_by_criteria
from src_logging.logging_combine import data_combine_table2col
from src_plot.plot_matrxi_scatter import plot_matrxi_scatter
from src_well_data.data_logging_FMI import DataFMI
from src_well_data.data_logging_normal import DataLogging
from src_well_data.data_logging_table import DataTable
from src_well_data.data_logging_NMR import DataNMR

class DATA_WELL:
    """
    井数据统一管理器：
    - 日常曲线测井数据 DataLogging
    - 电成像 FMI DataFMI
    - 表格类型数据 DataTable
    - 未来拓展 NMR 数据
    """

    # =============== 基础初始化 ==================
    def __init__(self, path_folder: str = '', WELL_NAME: str = ''):

        # ---- 数据容器 ----
        self.logging_dict: Dict[str, DataLogging] = {}
        self.table_dict: Dict[str, DataTable] = {}
        self.FMI_dict: Dict[str, DataFMI] = {}
        self.NMR_dict: Dict[str, Any] = {}

        # ---- 路径容器 ----
        self.path_list_logging: List[str] = []
        self.path_list_table: List[str] = []
        self.path_list_fmi: List[str] = []
        self.path_list_nmr: List[str] = []

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
        self.NMR_KW = ['nmr']

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

        self.path_list_logging = search_files_by_criteria(
            self.well_path,
            name_keywords=self.LOGGING_KW,
            file_extensions=['.xlsx', '.csv'],
            all_keywords=False
        )

        self.path_list_table = search_files_by_criteria(
            self.well_path,
            name_keywords=self.TABLE_KW,
            file_extensions=['.xlsx', '.csv'],
            all_keywords=False
        )

        self.path_list_fmi = search_files_by_criteria(
            self.well_path,
            name_keywords=self.FMI_KW,
            file_extensions=['.txt'],
            all_keywords=False
        )

        self.path_list_nmr = search_files_by_criteria(
            self.well_path,
            name_keywords=self.NMR_KW,
            file_extensions=['.csv'],
            all_keywords=False
        )

    # =========================================================================
    #                          内部辅助函数
    # =========================================================================
    def _get_default_obj(self, data_dict: Dict, key: str = ''):
        """
        dict 为空 → 返回空
        key为空  → 返回第一个
        key匹配  → 返回对应对象
        """
        if not data_dict:
            print("\033[33m[WARN] 数据未初始化\033[0m")
            return None

        if not key:
            return next(iter(data_dict.values()))  # 返回第一个对象

        # 支持模糊匹配
        for k in data_dict.keys():
            if key in k:
                return data_dict[k]

        # 完全匹配失败
        return None

    # =========================================================================
    #                          数据初始化模块
    # =========================================================================
    def init_logging(self, path: str = ''):
        """初始化普通测井数据"""
        if not path:
            if not self.path_list_logging:
                return
            path = self.path_list_logging[0]

        if path not in self.logging_dict:
            self.logging_dict[path] = DataLogging(path=path, well_name=self.WELL_NAME)

    def init_table(self, path: str = ''):
        """初始化表格数据"""
        if not path:
            if not self.path_list_table:
                return
            path = self.path_list_table[0]

        if path not in self.table_dict:
            self.table_dict[path] = DataTable(path=path, well_name=self.WELL_NAME)

    def init_FMI(self, path: str = ''):
        """初始化电成像数据（stat/dyna均可）"""
        if not path:
            if not self.path_list_fmi:
                return
            path = self.path_list_fmi[0]

        if path not in self.FMI_dict:
            self.FMI_dict[path] = DataFMI(path_fmi=path)

    def init_NMR(self, path: str = ''):
        """初始化核磁数据"""
        if not path:
            if not self.path_list_fmi:
                return
            path = self.path_list_fmi[0]

        if path not in self.NMR_dict:
            self.NMR_dict[path] = DataNMR(path_nmr=path)

    # =========================================================================
    #                          统一访问接口
    # =========================================================================
    def get_logging(self, key: str = '',
                    curve_names: List[str] = None,
                    norm: bool = False,
                    depth_limit: List[float] = []):
        """
        获取测井数据 DataFrame

        :param key: 文件名或关键字
        :param curve_names: 需要的曲线列表
        :param norm: 是否归一化
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
        mode='3': depth_start, depth_end, type
        mode='2': depth, type
        """
        self.init_table(key)
        obj = self._get_default_obj(self.table_dict, key)
        if obj is None:
            return pd.DataFrame()

        if replaced and replace_dict:
            obj._apply_type_replacement(replace_dict=new_col)

        return obj.get_table_3() if mode == '3' else obj.get_table_2()

    def get_FMI(self, key: str = '', depth: Optional[List[float]] = None):
        """获得 FMI 电成像数据"""
        self.init_FMI(key)
        obj = self._get_default_obj(self.FMI_dict, key)
        if obj is None:
            return None
        return obj.get_data(depth)

    def get_NMR(self, key: str = '', depth: Optional[List[float]] = None):
        self.init_NMR(key)
        obj = self._get_default_obj(self.NMR_dict, key)
        if obj is None:
            return None
        return obj.get_data(depth)

    def get_FMI_texture(self, key: str = '', texture_config: Optional[Dict] = None):
        """获得 FMI 电成像数据的纹理数据"""
        self.init_FMI(key)
        obj = self._get_default_obj(self.FMI_dict, key)
        if obj is None:
            return None
        texture = obj.get_texture(texture_config, fmi_texture_path='')
        return texture

    def get_path_texture_all(self, texture_config):
        return self.well_path + f'\\{self.WELL_NAME}_texture_logging_{texture_config["windows_length"]}.csv'

    def get_FMI_textures(self, texture_config: Optional[Dict] = None, path_config={}):
        """获得 FMI 电成像数据的纹理数据，这个获取的是动静态成像的纹理特征"""
        # 计算保存全部纹理数据的路径
        path_texture_all = self.get_path_texture_all(texture_config)

        # 如果存在则直接进行计算并保存
        if os.path.exists(path_texture_all):
            print('纹理文件已存在，直接进行读取', path_texture_all)
            return pd.read_csv(path_texture_all)

        # 不存在的话，重新计算动静态电成像纹理数据
        if 'path_dyna' in path_config:
            path_dyna = path_config['path_dyna']
            if path_dyna not in self.path_list_fmi:
                raise FileNotFoundError("file {} not found".format(path_dyna))
        else:
            path_dyna = self.search_fmi_path_list(new_kw=[self.FMI_KW[0]])[0]
        if 'path_stat' in path_config:
            path_stat = path_config['path_stat']
            if path_stat not in self.path_list_fmi:
                raise FileNotFoundError("file {} not found".format(path_stat))
        else:
            path_stat = self.search_fmi_path_list(new_kw=[self.FMI_KW[1]])[0]

        texture_dyna = self.get_FMI_texture(key=path_dyna, texture_config=texture_config)
        texture_stat = self.get_FMI_texture(key=path_stat, texture_config=texture_config)

        TEXTURE_ALL = pd.concat([texture_stat, texture_dyna.iloc[:, 1:]], axis=1)
        print(path_texture_all)
        TEXTURE_ALL.to_csv(path_texture_all, index=False)
        return TEXTURE_ALL

    def get_FMI_fde(self, key: str = '', fde_config: Optional[Dict] = None):
        """获得 FMI 电成像数据的fde图谱数据，这个获取指定路径下的电成像数据"""
        self.init_FMI(key)
        obj = self._get_default_obj(self.FMI_dict, key)
        if obj is None:
            return None
        fmi_fde = obj.get_fmi_fde(config_fde=fde_config)
        return fmi_fde

    def get_FMI_fdes(self, fde_config: Optional[Dict] = None):
        """获得 FMI 电成像数据的fde图谱数据，这个获取的是动静态成像的fde数据"""
        # 不存在的话，重新计算动静态电成像纹理数据
        path_dyna = self.search_data_path(keywords=[self.FMI_KW[0]], path_list=self.path_list_fmi)
        path_stat = self.search_data_path(keywords=[self.FMI_KW[1]], path_list=self.path_list_fmi)
        fde_dyna = self.get_FMI_fde(key=path_dyna, fde_config=fde_config)
        fde_stat = self.get_FMI_fde(key=path_stat, fde_config=fde_config)

        return fde_dyna, fde_stat


    # =========================================================================
    #                          数据概览接口
    # =========================================================================
    def well_summary(self) -> Dict[str, Any]:
        return {
            "well": self.WELL_NAME,
            "path": self.well_path,
            "paths_logging": self.path_list_logging,
            "paths_fmi": self.path_list_fmi,
            "paths_table": self.path_list_table,
            "paths_nmr": self.path_list_nmr,
            "logging_files_num": len(self.path_list_logging),
            "fmi_files_num": len(self.path_list_fmi),
            "table_files_num": len(self.path_list_table),
            "nmr_files_num": len(self.path_list_nmr),
        }

    def __repr__(self):
        return f"<DATA_WELL {self.WELL_NAME} | logging={len(self.logging_dict)}, fmi={len(self.FMI_dict)}, table={len(self.table_dict)}>"

    def combine_logging_table(
            self,
            logging_key='',
            curve_names_logging=None,
            table_key='',
            replace_dict=None,
            new_col='Type',
            norm=False,
    ):
        """
        将连续曲线logging与类型表（3列或2列）合并
        生成 (depth + curves + lithology_label)
        """

        # 1 获取曲线数据
        df_log = self.get_logging(logging_key, curve_names_logging, norm)
        depth_col = df_log.columns[0]

        # 2 获取 table
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
        array_merge = data_combine_table2col(array_logging, array_table, drop=True)

        data_columns = logging_columns + [table_columns[-1]]
        df_merge = pd.DataFrame(array_merge, columns=data_columns)
        df_merge.dropna(inplace=True)
        df_merge[table_columns[-1]] = df_merge[table_columns[-1]].astype(int)

        if new_col != '' or new_col is None:
            ##### 重命名
            df_merge.rename(columns={table_columns[-1]: new_col}, inplace=True)

        return df_merge

    def get_table_replace_dict(self, table_key=''):
        self.init_table(table_key)
        table_obj = self._get_default_obj(self.table_dict, table_key)
        return table_obj.get_replace_dict()

    def get_path_list_logging(self):
        return self.path_list_logging

    def get_path_list_fmi(self):
        return self.path_list_fmi

    def get_path_list_table(self):
        return self.path_list_table

    def search_logging_path_list(self, new_kw=[]):
        path_list_logging = search_files_by_criteria(
            self.well_path,
            name_keywords=new_kw,
            file_extensions=['.xlsx', '.csv'],
            all_keywords=True
        )
        return path_list_logging

    def search_table_path_list(self, new_kw=[]):
        path_list_table = search_files_by_criteria(
            self.well_path,
            name_keywords=new_kw,
            file_extensions=['.xlsx', '.csv'],
            all_keywords=True
        )
        return path_list_table

    def search_fmi_path_list(self, new_kw=[]):
        path_list_fmi = search_files_by_criteria(
            self.well_path,
            name_keywords=new_kw,
            file_extensions=['.txt'],
            all_keywords=True
        )
        return path_list_fmi

    def search_nmr_path_list(self, new_kw=[]):
        path_list_nmr = search_files_by_criteria(
            self.well_path,
            name_keywords=new_kw,
            file_extensions=['.csv'],
            all_keywords=True
        )
        return path_list_nmr



if __name__ == '__main__':
    # well = DATA_WELL("F:\logging_workspace\桃镇1H")
    # well = DATA_WELL(r'F:\logging_workspace\禄探')
    well = DATA_WELL(r'F:\logging_workspace\云安012-X18')

    summary_temp = well.well_summary()
    for k, val in summary_temp.items():
        print(k, val)

    logging_data_temp = well.get_logging()
    print(logging_data_temp.describe())

    path_list_fmi = well.get_path_list_fmi()
    print(path_list_fmi)

    path_list_logging = well.get_path_list_logging()
    print(path_list_logging)

    path_logging_target = well.search_logging_path_list(new_kw=['120', 'TEXTURE', 'logging'])
    print(path_logging_target)
    path_table_target = well.search_table_path_list(new_kw=['table'])
    print(path_table_target)
    path_fmi_dyna_target = well.search_fmi_path_list(new_kw=["DYNA"])
    print(path_fmi_dyna_target)
    path_fmi_stat_target = well.search_fmi_path_list(new_kw=['STAT'])
    print(path_fmi_stat_target)


    # # well.get_FMI_fdes(fde_config={'windows_length': 160, 'windows_step': 40, 'processing_method': 'original'})
    #
    # # # texture_dyna = well.get_FMI_texture(key='F:\\logging_workspace\\云安012-X18\\云安012-X18-DYNA.txt', texture_config = {
    # # #         'level': 16,  # 灰度级别
    # # #         'distance': [2, 4],  # 像素距离
    # # #         'angles': [0, np.pi / 2],  # 角度方向
    # # #         'windows_length': 80,  # 窗口长度
    # # #         'windows_step': 10  # 滑动步长
    #
    # # # })
    # # # texture_stat = well.get_FMI_texture(key='F:\\logging_workspace\\云安012-X18\\云安012-X18-STAT.txt', texture_config = {
    # # #         'level': 16,  # 灰度级别
    # # #         'distance': [2, 4],  # 像素距离
    # # #         'angles': [0, np.pi / 2],  # 角度方向
    # # #         'windows_length': 80,  # 窗口长度
    # # #         'windows_step': 10  # 滑动步长
    # # # })
    # # # print(texture_dyna.describe())
    # # # print(texture_stat.describe())
    #
    # texture_all = well.get_FMI_textures(texture_config={
    #         'level': 16,  # 灰度级别
    #         'distance': [2, 4],  # 像素距离
    #         'angles': [0, np.pi / 2],  # 角度方向
    #         'windows_length': 120,  # 窗口长度
    #         'windows_step': 10  # 滑动步长
    # })
    # print(texture_all.describe())
    #
    # # input_cols = ['AC', 'CAL', 'CNL', 'DEN', 'DTS', 'GR', 'RT', 'RXO']
    # # input_cols = ['CON_MEAN_STAT', 'DIS_MEAN_STAT', 'HOM_MEAN_STAT', 'ENG_MEAN_STAT', 'COR_MEAN_STAT', 'ASM_MEAN_STAT', 'ENT_MEAN_STAT', 'CON_SUB_STAT', 'DIS_SUB_STAT', 'HOM_SUB_STAT', 'ENG_SUB_STAT', 'COR_SUB_STAT', 'ASM_SUB_STAT', 'ENT_SUB_STAT', 'CON_X_STAT', 'DIS_X_STAT', 'HOM_X_STAT', 'ENG_X_STAT', 'COR_X_STAT', 'ASM_X_STAT', 'ENT_X_STAT', 'CON_Y_STAT', 'DIS_Y_STAT', 'HOM_Y_STAT', 'ENG_Y_STAT', 'COR_Y_STAT', 'ASM_Y_STAT', 'ENT_Y_STAT']
    # # input_cols = ['COR_MEAN_STAT', 'COR_Y_STAT', 'ASM_SUB_STAT', 'HOM_SUB_STAT', 'COR_X_STAT', 'ENG_SUB_STAT', 'ENT_SUB_STAT', 'HOM_X_STAT']
    # # input_cols = ['CON_MEAN_DYNA', 'DIS_MEAN_DYNA', 'HOM_MEAN_DYNA', 'ENG_MEAN_DYNA', 'COR_MEAN_DYNA', 'ASM_MEAN_DYNA', 'ENT_MEAN_DYNA', 'CON_SUB_DYNA', 'DIS_SUB_DYNA', 'HOM_SUB_DYNA', 'ENG_SUB_DYNA', 'COR_SUB_DYNA', 'ASM_SUB_DYNA', 'ENT_SUB_DYNA', 'CON_X_DYNA', 'DIS_X_DYNA', 'HOM_X_DYNA', 'ENG_X_DYNA', 'COR_X_DYNA', 'ASM_X_DYNA', 'ENT_X_DYNA', 'CON_Y_DYNA', 'DIS_Y_DYNA', 'HOM_Y_DYNA', 'ENG_Y_DYNA', 'COR_Y_DYNA', 'ASM_Y_DYNA', 'ENT_Y_DYNA']
    # # input_cols = ['ENT_SUB_DYNA', 'HOM_X_DYNA', 'ENG_SUB_DYNA', 'HOM_SUB_DYNA', 'ASM_SUB_DYNA', 'CON_SUB_DYNA', 'CON_X_DYNA', 'DIS_SUB_DYNA']
    input_cols = ['HOM_X_STAT', 'HOM_X_DYNA', 'ENT_SUB_DYNA', 'ASM_SUB_STAT', 'TEXTURE_LEVEL']

    # target_col = 'Type'
    # logging_data_temp_type = well.combine_logging_table(
    #         logging_key=path_logging_target[0],
    #         curve_names_logging=input_cols,
    #         table_key=path_table_target[0],
    #         replace_dict={},
    #         new_col=target_col,
    #         norm=False,
    # )
    # print(list(logging_data_temp_type.columns))
    # print(logging_data_temp_type.describe())
    logging_data_temp_type = well.get_logging(key=path_logging_target[0], curve_names=input_cols)
    print('form path {} read data :{}'.format(path_logging_target[0], logging_data_temp_type.columns))

    # 创建模型实例
    model = MultiVariateLinearRegressor(fit_intercept=True)
    # # 训练模型（使用x1, x2, x3预测y1, y2）
    # model.fit(logging_data_temp_type, x_cols=['HOM_X_STAT', 'HOM_X_DYNA', 'ENT_SUB_DYNA', 'ASM_SUB_STAT'], y_cols=['Type'])
    coef_matrix = np.array([-2.82736617, 1.65717694, -1.32970234, -15.03774471]).reshape((1, 4))
    intercept_matrix = np.array([2.08278412]).reshape((1, 1))
    print('intercept_matrix is :', intercept_matrix.shape, 'coef_matrix is :', coef_matrix.shape)
    model.set_fit_paras(x_cols=['HOM_X_STAT', 'HOM_X_DYNA', 'ENT_SUB_DYNA', 'ASM_SUB_STAT'], y_cols=['Type'], intercept_matrix=intercept_matrix, coef_matrix=coef_matrix)
    ############## 进行预测
    logging_data_temp_type.loc[:, 'TEXTURE_LEVEL'] = model.predict(logging_data_temp_type)

    def classify_texture(value):
        if value < 1.0:
            return 0
        elif 1.0 <= value < 1.3:
            return 1
        elif 1.5 <= value < 1.78:
            return 2
        else:
            return 3
    logging_data_temp_type['TYPE_PRED'] = logging_data_temp_type['TEXTURE_LEVEL'].apply(classify_texture)

    # logging_data_temp_type.to_csv(well.well_path+'\\'+well.WELL_NAME+'_result.csv', index=False)

    # ############## 计算评估指标
    # test_metrics = model.score(logging_data_temp_type)
    # print(type(model.coef_matrix), type(model.intercept_matrix), model.coef_matrix.shape, model.intercept_matrix.shape, model.coef_matrix, model.intercept_matrix)

    # pearson_result, pearson_sorted, rf_result, rf_sorted = feature_influence_analysis(
    #     df_input=logging_data_temp_type,
    #     input_cols=input_cols,
    #     target_col=target_col,
    #     regressor_use=False,
    #     replace_dict={},
    # )
    # print("\n按皮尔逊系数排序的属性:", pearson_sorted)
    # print("\n按随机森林特征重要性排序的属性:", rf_sorted)

    # # 按组抽稀，每个组保留50%的数据
    # logging_data_dilute = dilute_dataframe(logging_data_temp_type, ratio=5, method='random', group_by=target_col)
    # print(f"按组抽稀50%后形状: {logging_data_dilute.shape}")
    # plot_matrxi_scatter(logging_data_dilute, ['HOM_X_STAT', 'HOM_X_DYNA', 'ENT_SUB_DYNA', 'ASM_SUB_STAT', 'CON_X_DYNA', 'CON_SUB_DYNA', 'COR_MEAN_STAT'], target_col, target_col_dict={})

    depth_config = [logging_data_temp_type['DEPTH'].min(), logging_data_temp_type['DEPTH'].max()]
    fmi_dynamic, depth_dyna = well.get_FMI(key=path_fmi_dyna_target[0], depth=depth_config)
    fmi_static, depth_stat = well.get_FMI(key=path_fmi_stat_target[0], depth=depth_config)
    print('fmi depth from {} to {}, and fmi data shape is :{}'.format(depth_dyna[0,], depth_dyna[-1], fmi_static.shape))
