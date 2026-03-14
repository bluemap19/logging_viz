# xlsx 类型文件信息读取
import pandas as pd


def get_data_from_pathlist(file_list, Charter_Config_Curve):
    # 判断有没有直接的分层信息，有的话，直接从层序信息读取分层信息
    if ('path_directly' in Charter_Config_Curve) and (len(Charter_Config_Curve['path_directly']) > 4):
        table_info = pd.read_excel(Charter_Config_Curve['path_directly'], sheet_name=Charter_Config_Curve['sheet_name'])[Charter_Config_Curve['curve_name']]
        return table_info

    target_file = []
    for i in range(len(file_list)):
        # xlsx文件，特征参数对比,寻找个参数均合适的excel文件
        if (
                Charter_Config_Curve['file_name'] in file_list[i]  # 更直观的包含判断
                and (file_list[i].lower().endswith('.xlsx') or file_list[i].lower().endswith('.csv'))# 增加点号并统一大小写
                and '~' not in file_list[i]
        ):
            target_file.append(file_list[i])
    if len(target_file) == 0:
        print('Cant find the file as charter:{}'.format(Charter_Config_Curve['file_name']))
        exit(0)
    elif len(target_file) == 1:
        print('current file: ', target_file[0], '\tSheet name: ', Charter_Config_Curve['sheet_name'], '\tCurve_name: ', Charter_Config_Curve['curve_name'])
        if target_file[0].endswith('.xlsx'):
            ALL_DATA = pd.read_excel(target_file[0], sheet_name=Charter_Config_Curve['sheet_name'])
        elif target_file[0].endswith('.csv'):
            ALL_DATA = pd.read_csv(target_file[0])
        else:
            ALL_DATA = pd.DataFrame([])
        pd.set_option('display.max_columns', None)      # 设置DataFrame显示所有列
        # pd.set_option('display.max_rows', None)         # 设置DataFrame显示所有行
        # pd.set_option('max_colwidth', 400)                  # 设置value的显示长度为100，默认为50
        # print('Data Frame All describe:\n{}'.format(ALL_DATA.describe()))
        curve_org = ALL_DATA[Charter_Config_Curve['curve_name']]
        # 把 除最后一列 的所有信息转换为数字格式
        # curve_org.iloc[:, 0:-1] = curve_org.iloc[:, 0:-1].apply(pd.to_numeric, errors='coerce')
        print('Data Frame Choice describe:\n{}'.format(curve_org.describe()))

        if 'depth' in Charter_Config_Curve['curve_name'][0].lower():
            curve_depth = curve_org.iloc[:, 0].values.reshape((-1, 1))
            # 计算 常规九条测井数据的分辨率
            LEV_normal = (curve_depth[-1, 0] - curve_depth[0, 0]) / curve_depth.shape[0]
            print('self.curve_org shape :{}, logging resolution:{:.2f}, Depth is from {} to {}'.format(
                curve_org.shape, LEV_normal, curve_depth[0, 0], curve_depth[-1, 0]))

        return curve_org
    elif len(target_file) > 1:
        print('Error file charter:{},there is multi target file:{}'.format(Charter_Config_Curve['file_name'], target_file))
        exit(0)