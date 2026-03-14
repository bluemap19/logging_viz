import os
import numpy as np
import pandas as pd
from src_file_op.dir_operation import get_all_subfolder_paths
from src_table.table_process import get_replace_dict
from src_well_data.DATA_WELL import DATA_WELL


class LOGGING_PROJECT:
    def __init__(self, project_path=r'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV'):
        self.PROJECT_PATH=project_path
        self.WELL_NAMES = []        ### ['元543',  '珠74', '珠79', '珠80']
        self.WELL_PATH = {}         ### {'珠80': 'C:\\Users\\ZFH\\Desktop\\算法测试-长庆数据收集\\logging_CSV\\珠80', '元543': 'C:\\Users\\ZFH\\Desktop\\算法测试-长庆数据收集\\logging_CSV\\元543', '珠74': 'C:\\Users\\ZFH\\Desktop\\算法测试-长庆数据收集\\logging_CSV\\珠74', '珠79': 'C:\\Users\\ZFH\\Desktop\\算法测试-长庆数据收集\\logging_CSV\\珠79'}
        # 存放的是每一个井名对应的文件路径
        self.well_logging_path_dict = {}        ### {'珠80':['C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV\珠80\Texture_File\珠80_Texture_ALL_logging_50_5.csv', 'C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV\珠80\珠80_logging_data.csv']}
        self.well_table_path_dict = {}          ### {'珠80':['C:\Users\ZFH\Desktop\算法测试-长庆数据收集\logging_CSV\珠80\珠80__LITHO_TYPE.csv'}
        self.well_FMI_path_dict = {}
        self.well_NMR_path_dict={}

        self.target_curves = {}                 ### 做整体任务时，需要用到输入测井资料的配置
        self.WELL_DATA = {}                     ### {'珠80':WELL(path='珠80'), '元543'=WELL(path='元543'), '珠74':WELL(path='珠74'), '珠79':WELL(path='珠79')}
        self.replace_dict_all = {}              ### 做整体任务时，需要用到的替换字典配置
        self.data_vc = pd.DataFrame()           ### 做整体任务时，数据垂直链接到一起的整体数据

        self.init_well_path()                   ### 初始化工作区间数据，主要包括初始化井名，井类Class：WELL

    def init_well_path(self):
        # 便利self.PROJECT_PATH项目路径文件夹
        path_all = get_all_subfolder_paths(self.PROJECT_PATH)
        # 判断self.WELL_NAMES是否为空，如果为空，就把 项目文件夹 下所有的文件夹，初始化成其井名
        if not self.WELL_NAMES:
            self.WELL_NAMES = [os.path.basename(p) for p in path_all]
            self.WELL_PATH = {name: path for name, path in zip(self.WELL_NAMES, path_all)}
        else:       # 如果 self.WELL_NAMES 不为空，就使用指定的文件夹作为初始路径
            valid_paths = {}
            WELL_NAMES = []
            for p in path_all:
                folder_name = os.path.basename(p)
                if folder_name in self.WELL_NAMES:
                    valid_paths[folder_name] = p
                    WELL_NAMES.append(folder_name)
            self.WELL_PATH = valid_paths
            self.WELL_NAMES = WELL_NAMES

        # 根据self.WELL_PATH初始化所有的井信息，初始化所有的WELL()类Class
        for well_name in self.WELL_PATH.keys():
            well_temp = DATA_WELL(path_folder=self.WELL_PATH[well_name], WELL_NAME=well_name)
            self.well_logging_path_dict[well_name] = well_temp.get_logging_path_list()
            self.well_table_path_dict[well_name] = well_temp.get_table_path_list()
            # self.well_FMI_path_dict[well_name] = well_temp.get_FMI_path_list()
            # self.well_NMR_path_dict = well_temp.get_NMR_path_list()

            self.WELL_DATA[well_name] = well_temp

        # print(self.WELL_NAMES)
        # print(self.WELL_PATH)
        # print(self.WELL_DATA)

        # print(self.well_logging_path_dict)
        # print(self.well_table_path_dict)
        # print(self.well_FMI_path_dict)
        # print(self.well_NMR_path_dict)

    # 这个是获取字典元素，如果key_default为空，就取第一个Value元素，否则取key_default对应的元素
    # 这个主要是用来对各种字典进行元素提取的
    def get_default_dict(self, dict={}, key_default=''):
        if not dict:
            print('Empty dictionary get')
            return
        if key_default == '':
            key_default = list(dict.keys())[0]
            value_default = dict[key_default]
        else:
            value_default = dict[key_default]
        return value_default

    # 获取井数据，如果井名为空，就取第一个井名，如果文件路径为空，就取第一个文件路径，如果曲线名为空，就取第一个曲线名
    def get_well_data(self, well_name='', file_path='', curve_names=[], Norm=False):
        # 判断传进来的井名是否在self.WELL_NAMES中，如果不在，判断是否为空，如果在，继续根据井名获取井数据
        if well_name not in self.WELL_NAMES:
            if len(well_name) == 0:
                well_name = self.WELL_NAMES[0]
            else:
                print('Error well name:{}'.format(well_name))
                exit(0)

        if file_path == '':
            file_path = self.well_logging_path_dict[well_name][0]
            print('current file path set as: {}'.format(file_path))

        WELL_temp = self.get_default_dict(self.WELL_DATA, well_name)
        data = WELL_temp.get_logging_data(well_key=file_path, curve_names=curve_names, Norm=Norm)
        return data

    # # 根据井名、文件路径关键字特征、目标文件类型 获得目标文件的数据dataframe
    def get_well_data_by_charters(self, well_name='', target_path_feature=['Texture_ALL', '_20_5'],
                                  target_file_type='logging', curve_names=[], Norm=False):
        """
        :param well_name: 井名-字符串 string 例如：城96
        :param target_path_feature: 搜索的目标字符串集合 例如：['Texture_ALL', '_20_5', 'csv']
        :param target_file_type: 搜索哪一种数据，字符串string，只能从以下四种数据里面进行选择：logging, table, FMI, NMR
        :param curve_names: 曲线名称对应list，例如：['GR', 'RHOB', 'NPHI']
        :param Norm: 是否正则化，布尔值，True为正则化，False为不正则化
        :return:返回目标数据dataframe
        """
        WELL = self.get_default_dict(self.WELL_DATA, well_name)
        target_file = WELL.search_logging_data_by_charters(target_path_feature=target_path_feature,
                                                        target_file_type=target_file_type,
                                                        curve_names=curve_names, Norm=Norm)
        return target_file

    # 根据井名、文件路径关键字特征、目标文件类型 获得目标文件的路径
    def search_target_file_path(self, well_name='', target_path_feature=['Texture_ALL', '_20_5'], target_file_type='logging'):
        WELL = self.get_default_dict(self.WELL_DATA, well_name)
        path = WELL.search_data_path_by_charters(target_path_feature=target_path_feature, target_file_type=target_file_type)
        return path

    # 获取指定井的类别数据，
    def get_table_3_data(self, well_name='', file_path='', curve_names=[]):
        WELL_temp = self.get_default_dict(self.WELL_DATA, well_name)
        type_3 = WELL_temp.get_type_3(table_key=file_path, curve_names=curve_names)
        return type_3
    def get_table_2_data(self, well_name='', file_path='', curve_names=[]):
        WELL_temp = self.get_default_dict(self.WELL_DATA, well_name)
        type_2 = WELL_temp.get_type_2(table_key=file_path, curve_names=curve_names)
        return type_2
    def get_table_3_data_replaced(self, well_name='', file_path='', curve_names=[], replace_dict={}):
        if not replace_dict:
            replace_dict = self.replace_dict_all
        WELL_temp = self.get_default_dict(self.WELL_DATA, well_name)
        data_logging_type = WELL_temp.get_type_3_replaced
        return data_logging_type
    def get_table_2_data_replaced(self, well_name='', file_path='', curve_names=[], replace_dict={}):
        if not replace_dict:
            replace_dict = self.replace_dict_all
        WELL_temp = self.get_default_dict(self.WELL_DATA, well_name)
        data_logging_type = WELL_temp.get_type_2_replaced
        return data_logging_type


    # 获取指定井的3列类别数据，这个主要是用来统计工作区间都是包含那些类的
    def get_table_3_all_data(self, well_names=[], file_path={}, curve_names=[]):
        if well_names == []:
            well_names = self.WELL_NAMES

        if not file_path:
            for well_name in well_names:
                file_path[well_name] = ''

        table_3_list = []
        for well_name in well_names:
            table_3_df_t = self.get_table_3_data(well_name=well_name, file_path=file_path[well_name], curve_names=curve_names)
            table_3_list.append(table_3_df_t)
        self.data_vc = self.data_vertical_cohere(data_list=table_3_list, well_names=well_names)
        return self.data_vc

    # dataframe的垂直合并，将多个井的数据合并为一个dataframe，输入为需要进行垂直合并的dataframe数据list及其对应的井名数据，井名数据需要作为一个新的列加入进去
    def data_vertical_cohere(self, data_list=[], well_names=[]):
        if well_names == []:
            well_names = self.WELL_NAMES

        for i in range(len(data_list)):
            well_temp = data_list[i]
            # 对每一口井都先添加井名，再进行合并
            well_temp['Well_Name'] = well_names[i]
            if i == 0:
                data_vertical_combined = well_temp
            else:
                # 合并并添加来源标识
                data_vertical_combined = pd.concat(
                    [data_vertical_combined, well_temp], axis=0
                )
            # print('well_temp:\n{}'.format(well_temp))

        print('data vertical combined as shape:{}'.format(data_vertical_combined.shape))
        # print(data_vertical_combined.describe())
        # print(data_vertical_combined.shape)
        # print(data_vertical_combined.head())
        # exit(0)
        data_vertical_combined.reset_index(drop=True, inplace=True)
        return data_vertical_combined

    # 获得所有数据的合并table3，并利用table3计算replace_dict
    def get_all_table_replace_dict(self, well_names=[], file_path={}, curve_names=[], Type_col_name=''):
        table_value = self.get_table_3_all_data(well_names=well_names, file_path=file_path, curve_names=curve_names)
        if Type_col_name == '':
            Type_col_name = list(table_value.columns)[-2]       # -1为井名列，get_table_3_all_data函数会为table的最后一列加一个井名列
        self.replace_dict_all = get_replace_dict(table_value[Type_col_name])
        return self.replace_dict_all

    # 设置replace_dict
    def set_all_table_replace_dict(self, well_names=[], well_names_key={}, dict={}, type_new_col='Type'):
        if dict:
            self.replace_dict_all = dict
            print('set all table replace dict as:{}'.format(self.replace_dict_all))
        else:
            print('No replace dict')
            dict = self.replace_dict_all

        if not well_names_key:
            for well_name in self.WELL_NAMES:
                well_names_key[well_name] = ''

        for well_name in well_names:
            WELL_TEMP = self.get_default_dict(self.WELL_DATA, well_name)
            WELL_TEMP.reset_table_replace_dict(table_key=well_names_key[well_name], replace_dict=dict, new_col=type_new_col)


    # 合并
    def combined_all_logging_with_type(self, well_names=[], file_path_logging={}, file_path_table={}, curve_names_logging=[], curve_names_type=[], replace_dict={}, type_new_col='', Norm=False):
        """
        :param well_names:          list [] 井名列表
        :param file_path_logging:   文件路径dict {} 每一个井名对应的测井数据文件路径
        :param file_path_table:     表格路径dict {} 每一个井名对应的分类表格文件路径
        :param curve_names_logging: list [] 测井数据文件 要取的曲线名称list
        :param curve_names_type:    list [] 分类表格文件 要取的分类头header
        :param replace_dict:        分类表格文件 分类资料如何进行映射
        :param type_new_col:        分类替换后的新列的名称
        :param Norm:                是否对测井数据进行归一化
        :return:
        """
        if well_names == []:
            well_names = self.WELL_NAMES
        if not file_path_logging:
            for well_name in well_names:
                file_path_logging[well_name] = ''
        if not file_path_table:
            for well_name in well_names:
                file_path_table[well_name] = ''

        if not replace_dict:
            replace_dict = self.replace_dict_all

        data_logging_type_list = []
        for well_name in well_names:
            WELL_temp = self.get_default_dict(self.WELL_DATA, well_name)
            print('current processing well data:{}'.format(well_name))
            data_logging_type = WELL_temp.combine_logging_table(well_key=file_path_logging[well_name],
                                                                curve_names_logging=curve_names_logging,
                                                                table_key=file_path_table[well_name],
                                                                curve_names_table=curve_names_type,
                                                                replace_dict=replace_dict, new_col=type_new_col, Norm=Norm)
            data_logging_type_list.append(data_logging_type)
            # print(data_logging_type.describe())
        data_final = self.data_vertical_cohere(data_list=data_logging_type_list, well_names=well_names)
        # print(data_final.head(10))
        return data_final

    # 通过参数配置计算电成像纹理数据
    def get_fmi_texture(self, well_names=['SIMU4'], file_path_logging={}, Mode='ALL', texture_config={'level':16, 'distance':[2,4], 'angles':[0, np.pi/4, np.pi/2, np.pi*3/4], 'windows_length':80, 'windows_step':5}):
        if well_names == []:
            well_names = self.WELL_NAMES
        if not file_path_logging:
            for well_name in well_names:
                file_path_logging[well_name] = ''

        data_texture_dict = {}
        for well_name in well_names:
            WELL_temp = self.get_default_dict(self.WELL_DATA, well_name)
            print('current processing well data:{}'.format(well_name))
            data_texture = WELL_temp.get_fmi_texture(Mode=Mode, texture_config=texture_config)
            name_temp = well_name + '_' + Mode + '_texture'
            data_texture_dict[name_temp] = data_texture
            # print(data_logging_type.describe())

        return data_texture_dict




if __name__ == '__main__':
    TEST_LOGGING_PROJECT = LOGGING_PROJECT(project_path=r'F:\桌面\算法测试-长庆数据收集\logging_CSV')

    print(TEST_LOGGING_PROJECT.WELL_NAMES)

    data_logging_test = TEST_LOGGING_PROJECT.get_well_data('城96', file_path='', curve_names=[], Norm=False)

    print(data_logging_test.describe())

    data_logging_target_test = TEST_LOGGING_PROJECT.get_well_data_by_charters(well_name='城96', target_path_feature=['Texture_ALL', '_80_5'],
                                  target_file_type='logging', curve_names=['DYNA_CON', 'STAT_XY_ASM', 'STAT_XY_ENT'], Norm=False)
    print(data_logging_target_test.describe())

    path_logging_target = TEST_LOGGING_PROJECT.search_target_file_path(well_name='城96', target_path_feature=['Texture_ALL', '_120_5'], target_file_type='logging')
    print(path_logging_target)

    path_table_target = TEST_LOGGING_PROJECT.search_target_file_path(well_name='城96',
                                                                      target_path_feature=['litho'],
                                                                      target_file_type='table')
    print(path_table_target)

    table_3_data = TEST_LOGGING_PROJECT.get_table_3_all_data(well_names=['元543', '元552', '城96', '悦235', '悦88', '悦92',
                                                                         '珠201', '珠202', '珠23', '珠45', '珠74', '珠79',
                                                                         '珠80', '白159', '白291', '白292', '白294', '白300', '白75'])
    print(table_3_data.describe())

    table_3_replace_dict = TEST_LOGGING_PROJECT.get_all_table_replace_dict(well_names=['元543', '元552', '城96', '悦235', '悦88', '悦92',
                                                                         '珠201', '珠202', '珠23', '珠45', '珠74', '珠79',
                                                                         '珠80', '白159', '白291', '白292', '白294', '白300', '白75'])
    print(table_3_replace_dict)

    # 更新替换目标井的replace_dict
    TEST_LOGGING_PROJECT.set_all_table_replace_dict(well_names=['元543', '元552', '城96', '悦235', '悦88', '悦92',
                                                                         '珠201', '珠202', '珠23', '珠45', '珠74', '珠79',
                                                                         '珠80', '白159', '白291', '白292', '白294', '白300', '白75'],
                                                    dict={'中GR长英黏土质': 0, '中GR长英黏土质（泥岩）': 1,
                                                          '中低GR长英质': 0, '中低GR长英质（砂岩）': 2,
                                                          '富有机质长英质': 0, '富有机质长英质页岩': 1,
                                                          '富有机质黏土质': 0, '富有机质黏土质页岩': 2,
                                                          '高GR富凝灰长英质': 0, '高GR富凝灰长英质（沉凝灰岩）': 1})
    # 合并 目标井-目标测井数据 的 综合数据 镜柜replace_dict替换后的
    logging_data_with_type_all = TEST_LOGGING_PROJECT.combined_all_logging_with_type(well_names=['元543', '元552', '城96', '悦235', '悦88', '悦92',
                                                                         '珠201', '珠202', '珠23', '珠45', '珠74', '珠79',
                                                                         '珠80', '白159', '白291', '白292', '白294', '白300', '白75'],
                                                                         Norm=False,
                                                                         curve_names_logging=['AC', 'GR', 'DEN', 'CNL'],
                                                                         type_new_col='Type_litho')

    print(logging_data_with_type_all.describe())