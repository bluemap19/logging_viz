import os
import re
from pathlib import Path
from typing import Union, Iterable, List


# 检查是否存在指定文件夹，如果不存在，则建立新的文件夹
def check_and_make_dir(dir_path):
    """
    :param dir_path: string -- folder path needed to check
    :return: NULL
    """
    if os.path.exists(dir_path):
        return
    else:
        os.makedirs(dir_path)
        print('successfully create dir:{}'.format(dir_path))
        assert os.path.exists(dir_path), dir_path


# # 遍历文件夹，返回的是文件夹下的目录以及文件
# def traverse_folder(path):
#     """
#     :param path: string -- target folder path need to be traversed
#     :return: list -- all file path under target folder（include sub-folder）
#     """
#     FilePath = []
#     for filepath, dirnames, filenames in os.walk(path):
#         for filename in filenames:
#             # print(os.path.join(filepath, filename).replace('\\', '/'))
#             FilePath.append(os.path.join(filepath, filename).replace('\\', '/'))
#     return FilePath
# # a = traverseFolder('D:\GitHubProj\Pytorch-DDPM\src_ele')
# # print(a)
#
#
# # 遍历目标文件夹内所有的子文件夹路径
# def traverse_folder_folder(path):
#     """
#     :param path: string -- target folder path need to be traversed
#     :return: list -- all folder path under target folder（include sub-folder）
#     """
#     path_folder = []
#     for filepath, dirnames, filenames in os.walk(path):
#         for dir_name in dirnames:
#             path_folder.append(os.path.join(filepath, dir_name).replace('\\', '/'))
#
#     return path_folder
# # a = traverseFolder_folder(r'D:\GitHubProj\Pytorch-DDPM')
# # print(a)
#
#
# # 在指定文件夹内 根据关键词搜寻目标文件
# def folder_search_by_charter(path=r'D:/1111/Input/fanxie184', target_file_charter=['data_org'], end_token=[]):
#     """
#     :param path: string -- target folder path need to be searched
#     :param target_file_charter: list -- file string character needed
#     :param end_token: list -- file end character needed
#     :return: list -- target file path
#     """
#
#     file_list = traverse_folder(path)
#     file_list_tmp = []
#     if end_token == []:
#         for i in file_list:
#             for j in end_token:
#                 if i.endswith(j):
#                     file_list_tmp.append(i)
#     else:
#         file_list_tmp = file_list
#
#     target_file_list = []
#     for i in range(len(target_file_charter)):
#         for j in range(len(file_list_tmp)):
#             if file_list_tmp[j].__contains__(target_file_charter[i]):
#                 target_file_list.append(file_list_tmp[j])
#
#     return target_file_list
def get_all_file_paths(root_dir: Union[str, Path]) -> List[str]:
    """获取目录及其子目录下所有文件路径
    Args:
        root_dir: 需要遍历的根目录路径
    Returns:
        包含所有文件绝对路径的列表，路径使用正斜杠格式
    Examples:
        >>> paths = get_all_file_paths('D:/projects')
        >>> print(paths[:2])
        ['D:/projects/README.md', 'D:/projects/utils/__init__.py']
    """
    root_path = Path(root_dir)
    file_paths = []

    for current_dir, _, files in os.walk(root_path):
        for filename in files:
            # 统一转换为正斜杠路径
            full_path = Path(current_dir) / filename
            file_paths.append(str(full_path))

    return file_paths


def get_all_subfolder_paths(root_dir: Union[str, Path]) -> List[str]:
    """获取目录及其所有子目录路径
    Args:
        root_dir: 需要遍历的根目录路径
    Returns:
        包含所有子目录绝对路径的列表，路径使用正斜杠格式
    Examples:
        >>> folders = get_all_subfolder_paths('D:/projects')
        >>> print(folders[:2])
        ['D:/projects', 'D:/projects/src']
    """
    root_path = Path(root_dir)
    folder_paths = []
    for current_dir, dirs, _ in os.walk(root_path):
        # 添加子目录
        for dir_name in dirs:
            folder_paths.append(str(Path(current_dir) / dir_name))

    return sorted(set(folder_paths))  # 去重并排序


def search_files_by_criteria(
        search_root: Union[str, Path],
        name_keywords: Iterable[str] = (),
        file_extensions: Iterable[str] = (),
        all_keywords: bool = True,
) -> List[str]:
    """根据名称关键字和文件扩展名搜索文件
    Args:
        search_root: 需要搜索的根目录
        name_keywords: 文件名需要包含的关键字序列
        file_extensions: 需要匹配的文件扩展名序列（如 .txt）
    Returns:
        匹配文件的绝对路径列表，按字母顺序排列
    Examples:
        >>> find_files = search_files_by_criteria(
        ...     'D:/data',
        ...     name_keywords=['log', '2023'],
        ...     file_extensions=['.csv', '.xlsx']
        ... )
    """
    all_files = get_all_file_paths(search_root)
    matched_files = []

    # 转换大小写不敏感的扩展名集合
    if len(file_extensions) > 0:
        ext_set = {ext.lower().strip('.') for ext in file_extensions}
    else:
        ext_set = []

    for file_path in all_files:
        path_obj = Path(file_path)

        # 排除临时文件（两种常见临时文件格式）
        if path_obj.name.startswith('~') or '~' in path_obj.name:
            continue

        if len(ext_set)>0:
            # 检查扩展名
            if path_obj.suffix.lstrip('.').lower() not in ext_set:
                continue

        # 检查名称关键字,首先判断是严格对照所有关键词，还是只对照其中一项关键词
        filename = path_obj.stem.lower()
        if all_keywords:
            contains_all_keywords = all(
                keyword.lower() in filename
                for keyword in name_keywords
            )
        else:
            contains_all_keywords = any(
                keyword.lower() in filename
                for keyword in name_keywords
            )

        if contains_all_keywords:
            matched_files.append(file_path)

    return sorted(matched_files)

############ 根据关键字list：target_path_feature=[]在文件路径list：path_list中进行搜索，只有满足所有target_path_feature的文件路径才能被筛选出来，筛选出所有的path并返回
def search_target_path(path_list=[], target_path_feature=[]):
    matched_paths = []
    for path in path_list:
        # 将路径拆分为组件
        components = set(path.lower().split('/'))
        components |= set(path.lower().split('\\'))

        # 检查所有特征是否都在组件中
        if all(any(feature.lower() in comp for comp in components) for feature in target_path_feature):
            matched_paths.append(path)

    return matched_paths

# def search_target_path(path_list=None, target_path_feature=None):
#     """
#     改进版路径搜索函数：基于特征词精确匹配文件路径
#
#     参数:
#     path_list: 待搜索路径列表 (默认空列表)
#     target_path_feature: 路径特征词列表 (默认空列表)
#
#     返回:
#     匹配的路径列表
#     """
#     # 初始化参数
#     path_list = path_list or []
#     target_path_feature = target_path_feature or []
#
#     # 规范化特征词：小写+去除空格
#     norm_features = [feat.strip().lower() for feat in target_path_feature]
#     if not norm_features:
#         return path_list  # 无特征时返回所有路径
#
#     matched_paths = []
#
#     for path in path_list:
#         # 1. 路径规范化处理
#         norm_path = os.path.normcase(path)  # 系统兼容的大小写处理
#         norm_path = norm_path.replace('\\', '/')  # 统一分隔符
#
#         # 2. 提取纯文件名（不含扩展名）
#         filename = os.path.splitext(os.path.basename(norm_path))[0].lower()
#
#         # 3. 组合检查范围：文件名 + 父目录名
#         check_components = [filename]
#         parent_dirs = os.path.dirname(norm_path).split('/')
#         check_components.extend([d for d in parent_dirs if d])  # 过滤空目录名
#
#         # 4. 特征词精确匹配（词边界限定）
#         match_all = True
#         for feature in norm_features:
#             feature_found = any(
#                 # 精确匹配文件名组件（支持带下划线的特征）
#                 feature == comp_part or
#                 # 词边界匹配完整路径
#                 re.search(rf'\b{re.escape(feature)}\b', comp, re.IGNORECASE)
#                 for comp in check_components
#                 for comp_part in comp.split('_')  # 拆分文件名中的下划线
#             )
#             if not feature_found:
#                 match_all = False
#                 break
#
#         if match_all:
#             matched_paths.append(path)
#
#     return matched_paths


# if __name__ == '__main__':
#     save_dir = r'C:\Users\Administrator\Desktop\算法测试-长庆数据收集'
#     path_list = get_all_subfolder_paths(save_dir)
#     # print(path_list)
#     # check_and_make_dir(save_dir)
#     # print(traverseFolder_folder(save_dir).__len__())
#     # save_file_as_xlsx([np.zeros((10, 10)), np.zeros((20, 20)), np.zeros((7, 7))], sheet_name=['ssdf', 'sasfa'])
#     target_file_list = search_files_by_criteria(save_dir, name_keywords=['珠23'], file_extensions=['.xlsx'])
#     print(target_file_list)


# def get_path_from_pathlist(file_list, Charter_Config_Curve):
#     target_file = []
#     for i in range(len(file_list)):
#         # xlsx文件，特征参数对比,寻找个参数均合适的excel文件
#         if (
#                 Charter_Config_Curve['file_name'] in file_list[i]  # 更直观的包含判断
#                 and file_list[i].lower().endswith('.xlsx')  # 增加点号并统一大小写
#                 and '~' not in file_list[i]
#         ):
#             target_file.append(file_list[i])
#     if len(target_file) == 0:
#         print('Cant find the file as charter:{}'.format(Charter_Config_Curve['file_name']))
#         exit(0)
#     elif len(target_file) == 1:
#         print('current file: ', target_file[0], '\tSheet name: ', Charter_Config_Curve['sheet_name'], '\tCurve_name: ', Charter_Config_Curve['curve_name'])
#         ALL_DATA = pd.read_excel(target_file[0], sheet_name=Charter_Config_Curve['sheet_name'])
#         pd.set_option('display.max_columns', None)      # 设置DataFrame显示所有列
#         # pd.set_option('display.max_rows', None)         # 设置DataFrame显示所有行
#         # pd.set_option('max_colwidth', 400)                  # 设置value的显示长度为100，默认为50
#         # print('Data Frame All describe:\n{}'.format(ALL_DATA.describe()))
#         curve_org = ALL_DATA[Charter_Config_Curve['curve_name']]
#         # 把 除最后一列 的所有信息转换为数字格式
#         # curve_org.iloc[:, 0:-1] = curve_org.iloc[:, 0:-1].apply(pd.to_numeric, errors='coerce')
#         # print('Data Frame Choice describe:\n{}'.format(curve_org.describe()))
#
#         curve_depth = curve_org.iloc[:, 0].values.reshape((-1, 1))
#         # curve_org = curve_org[:, 1:]
#
#         # 计算 常规九条测井数据的分辨率
#         LEV_normal = (curve_depth[-1, 0] - curve_depth[0, 0]) / curve_depth.shape[0]
#
#         if 'depth' in Charter_Config_Curve['curve_name'][0].lower():
#             print('self.curve_org shape :{}, logging_data resolution:{:.2f}, Depth is from {} to {}'.format(
#                 curve_org.shape, LEV_normal, curve_depth[0, 0], curve_depth[-1, 0]))
#
#         return curve_org
#     elif len(target_file) > 1:
#         print('Error file charter:{},there is multi target file:{}'.format(Charter_Config_Curve['file_name'], target_file))
#         exit(0)