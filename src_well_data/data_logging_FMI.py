import os
import re

import cv2
import numpy as np
import pandas as pd
from src_data_process.cal_data_glcm_texture import cal_images_texture
import logging
from typing import Optional, List, Dict, Tuple
from enum import Enum

from src_fmi.fmi_fractal_dimension_extended_calculate import cal_fmis_fractal_dimension_extended, \
    trans_NMR_as_Ciflog_file_type, trans_fde_image_to_NMR_type
from src_plot.TEMP_8 import WellLogVisualizer

# 完整显示describe的全部信息不省略
#set_option函数可以解决数据显示不全问题，比如自动换行、显示……这种，本题不设置就会报错
pd.set_option('display.float_format', lambda x:'%.4f'%x) # 小数点后面保留3位小数，诸如此类，按需修改吧
pd.set_option('display.max_columns', None)# 显示所有的列，而不是以……显示
pd.set_option('display.max_rows', None)# 显示所有的行，而不是以……显示
pd.set_option('display.width', None) # 不自动换行显示


class FMIException(Exception):
    """
    FMI数据异常类
    用于处理电成像数据相关的特定异常情况
    """
    pass

class FileFormat(Enum):
    """
    文件格式枚举类
    定义支持的FMI数据文件格式
    """
    CSV = '.csv'
    TEXT = '.txt'
    IMAGE = '.image'
    UNKNOWN = 'unknown'

def ele_stripes_delete(Pic: np.ndarray, shape_target: Tuple[int, int] = (100, 8),
                       delete_pix: float = 0) -> np.ndarray:
    """
    空白条带删除函数 - 采用多退少补原则处理FMI图像中的无效像素

    算法原理：
    1. 对于图像中的每一行，查找有效像素（大于等于delete_pix的像素）
    2. 根据有效像素数量与目标宽度的关系进行处理：
       - 相等：直接使用有效像素
       - 不足：用0填充
       - 过多：调整尺寸以适应目标宽度

    Args:
        Pic: 原始FMI图像数据，二维numpy数组
        shape_target: 目标图像形状 (高度, 宽度)，高度必须与原始图像一致
        delete_pix: 像素删除阈值，小于此值的像素被视为无效

    Returns:
        处理后的图像数据，形状为shape_target

    Raises:
        ValueError: 当目标高度与原始图像高度不匹配时
    """
    # 创建目标形状的空数组
    pic_new = np.zeros(shape_target, dtype=np.float64)

    # 验证高度一致性
    if shape_target[0] != Pic.shape[0]:
        error_msg = f"形状错误: 原始形状{Pic.shape}与目标形状{shape_target}的高度不匹配"
        raise ValueError(error_msg)

    valid_count_all = 0
    # 逐行处理图像
    for i in range(pic_new.shape[0]):
        # 查找当前行中大于等于阈值的像素索引
        valid_indices = np.where(Pic[i, :] != delete_pix)[0]
        valid_count = len(valid_indices)
        valid_count_all += valid_count

        # 根据有效像素数量进行不同处理
        if valid_count == shape_target[1]:
            # 情况1: 有效像素数量正好等于目标宽度
            row_pixels = Pic[i, valid_indices]
        elif valid_count == 0:
            # 情况2: 没有有效像素，整行填充0
            row_pixels = np.zeros(shape_target[1])
        else:
            # 情况3: 有效像素数量不等于目标宽度，调整尺寸
            row_pixels = Pic[i, valid_indices]
            # 使用numpy的resize函数调整到目标宽度
            row_pixels = np.resize(row_pixels, shape_target[1])

        # 将处理后的行数据赋值到新图像
        pic_new[i, :] = row_pixels

    return pic_new, valid_count_all


class DataFMI:
    """
    FMI电成像数据管理核心类

    功能概述：
    1. 支持CSV和TXT格式的FMI数据读取
    2. 提供空白条带删除功能，优化图像质量
    3. 计算电成像数据的纹理特征（GLCM特征）
    4. 管理深度数据和成像数据的对应关系

    电成像数据特点：
    - 数据量大：每个深度点对应多个电极测量值
    - 深度连续性：数据按深度严格排序
    - 图像特性：可以视为沿深度方向的二维图像序列

    设计原则：
    - 数据封装：内部数据状态受保护，通过方法访问
    - 异常安全：完善的错误处理和数据验证
    - 配置灵活：纹理特征计算参数可配置
    """

    def __init__(self, path_fmi: str = '', well_name: str = '', fmi_charter: str = ''):
        """
        初始化FMI数据对象

        Args:
            path_fmi: FMI数据文件路径，支持.csv和.txt格式
            well_name: 井名标识，用于数据标识和日志记录
            fmi_charter: FMI仪器标识，用于特征命名区分

        Attributes:
            _table_2: 保留属性，用于与其他数据格式兼容
            _well_name: 井名标识
            _data_fmi: 存储原始FMI成像数据（二维numpy数组）
            _resolution: FMI数据分辨率（深度采样间隔）
            _data_depth: 存储深度数据（一维numpy数组）
            path_fmi: 数据文件路径
            fmi_charter: FMI仪器标识
            _logger: 日志记录器实例
            _is_data_loaded: 数据加载状态标志
        """
        # 数据存储属性
        self._table_2: pd.DataFrame = pd.DataFrame()  # 保留属性，用于兼容性
        self._data_fmi: np.ndarray = np.array([])  # FMI成像数据体
        self._data_depth: np.ndarray = np.array([])  # 深度数据

        # 配置参数
        self._resolution: float = 0.0025  # 默认分辨率
        self._well_name: str = well_name
        self.path_fmi: str = path_fmi
        if len(fmi_charter) == 0:
            if path_fmi.upper().__contains__('DYNA'):
                fmi_charter = 'DYNA'
            elif path_fmi.upper().__contains__('STAT'):
                fmi_charter = 'STAT'
            else:
                fmi_charter = 'OTHERS'
        self.fmi_charter: str = fmi_charter
        if self._well_name == '':
            self._well_name = self.path_fmi.split('\\')[-2]

        # 状态标志
        self._is_data_loaded: bool = False

        # 初始化日志系统
        self._logger = self.__setup_logger()

        # 检查文件是否存在
        if path_fmi and not os.path.isfile(path_fmi):
            self._logger.error(f"文件不存在或无法访问: {path_fmi}")

    def __setup_logger(self) -> logging.Logger:
        """
        设置并配置日志记录器

        Returns:
            配置好的logging.Logger实例
        """
        logger = logging.getLogger(f"DataFMI_{self._well_name}")

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        return logger

    def __detect_file_format(self, file_path: str) -> FileFormat:
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
        elif file_path.endswith('.png') or file_path.endswith('.jpg') or file_path.endswith('.bmp'):
            return FileFormat.IMAGE
        else:
            return FileFormat.UNKNOWN

    def read_data(self, file_path: str = '') -> None:
        """
        读取FMI电成像数据文件

        Args:
            file_path: 数据文件路径，为空时使用对象初始化路径

        Raises:
            FMIException: 文件读取失败或格式不支持时抛出

        Workflow:
            1. 确定文件路径并检查存在性
            2. 根据文件格式调用相应的读取方法
            3. 提取深度数据和成像数据
            4. 更新数据加载状态
        """
        # 确定文件路径
        file_path = file_path or self.path_fmi

        if not file_path:
            raise FMIException("未提供文件路径")

        if not os.path.isfile(file_path):
            raise FMIException(f"文件不存在: {file_path}")
        try:
            # 检测文件格式并读取数据
            file_format = self.__detect_file_format(file_path)
            self._logger.info(f"检测到文件格式: {file_format.value}")

            if file_format == FileFormat.CSV:
                self.__read_csv_file(file_path)
            elif file_format == FileFormat.TEXT:
                self.__read_text_file(file_path)
            elif file_format == FileFormat.IMAGE:
                self.__read_img_data(file_path)
            else:
                raise FMIException(f"不支持的文件格式: {file_path}")

            # 验证数据完整性
            if self._data_fmi.size == 0 or self._data_depth.size == 0:
                raise FMIException("读取到的数据为空")

            # 更新加载状态
            self._is_data_loaded = True
            self._logger.info(f"成功加载FMI数据，形状: {self._data_fmi.shape}")

        except Exception as e:
            self._logger.error(f"读取FMI数据失败: {e}")
            raise

    def __read_csv_file(self, file_path: str) -> None:
        """
        读取CSV格式的FMI数据文件
        Args:
            file_path: CSV文件路径
        Note:
            CSV格式假设第一列为深度，其余列为电极测量值
        """
        try:
            # 读取CSV文件，第一列作为索引（通常是深度）
            df = pd.read_csv(file_path, index_col=0)

            # 提取数据：第一列之后的所有列为FMI数据
            self._data_fmi = df.values
            # 索引列作为深度数据
            self._data_depth = df.index.values

            self._logger.debug(f"CSV文件读取成功，数据形状: {self._data_fmi.shape}")

        except Exception as e:
            raise FMIException(f"CSV文件读取失败: {e}")

    def __remove_path_image_extension(self, filename):
        """
        如果是图像文件，则移除文件的扩展名，来计算深度数据，如果是txt文件，就不需要这一步操作了
        :param filename:
        :return:
        """
        name, ext = os.path.splitext(filename)
        # 检查是否是图片扩展名
        if ext.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
            return name
        else:
            return filename  # 如果不是图片扩展名，返回原文件名

    def __read_img_data(self, file_path: str) -> None:
        """
        读取CSV格式的FMI数据文件
        Args:
            file_path: CSV文件路径
        Note:
            CSV格式假设第一列为深度，其余列为电极测量值
        """
        try:
            if file_path.endswith('.png'):
                self._data_fmi = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            elif file_path.endswith('.jpg'):
                self._data_fmi = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            elif file_path.endswith('.bmp'):
                # self._data_fmi = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                self._data_fmi = cv2.imread(file_path, cv2.IMREAD_COLOR_RGB)
            else:
                raise FMIException(f"图像文件读取失败{file_path}")

            file_name = self.__remove_path_image_extension(file_path)

            depth_str_list = re.split(r'[/\\_-]', file_name)
            depth_list = []
            for depth_str in depth_str_list:
                try:
                    depth_list.append(float(depth_str))
                except Exception as e:
                    continue

            if len(depth_list) >= 2:
                path_min = min(depth_list[-2:])
                path_max = max(depth_list[-2:])
                self._data_depth = np.linspace(path_min, path_max, self._data_fmi.shape[0])
            else:
                self._logger.warning(f"image have no depth information, reset it as default :{0} - {100} m")
                self._data_depth = np.linspace(0, 100, self._data_fmi.shape[0])

        except Exception as e:
            raise FMIException(f"图像文件读取失败: {e}")

    def __read_text_file(self, file_path: str) -> None:
        """
        读取TXT格式的FMI数据文件（通常为LAS格式）
        Args:
            file_path: TXT文件路径
        Note:
            TXT格式通常有文件头，需要跳过前几行
        """
        try:
            # 跳过前8行（LAS文件头），使用制表符分隔
            data_file = np.loadtxt(file_path, delimiter=None, skiprows=8)

            # 第一列为深度，其余列为FMI数据
            self._data_depth = data_file[:, 0]
            self._data_fmi = data_file[:, 1:]

            self._logger.debug(f"TXT文件读取成功，数据形状: {self._data_fmi.shape}")

        except Exception as e:
            raise FMIException(f"TXT文件读取失败: {e}")

    def get_data(self, depth: Optional[List[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取FMI成像数据和对应的深度数据
        Args:
            depth: 指定深度范围，为None时返回全部数据
        Returns:
            Tuple包含FMI数据数组和深度数据数组
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
            return self._data_fmi, self._data_depth
        else:
            # 根据深度范围筛选数据（待实现）
            depth_min = min(depth)
            depth_max = max(depth)
            idx = (self._data_depth >= depth_min) & (self._data_depth <= depth_max)
            return self._data_fmi[idx], self._data_depth[idx]

    def ele_stripes_delete(self, delete_pix: float = 0, width_ratio: float = 0.8) -> None:
        """
        删除FMI数据中的空白条带（无效像素）

        Args:
            delete_pix: 像素删除阈值，小于此值的像素被视为无效
            width_ratio: 宽度调整比例，用于确定目标图像的宽度

        Algorithm:
            1. 计算目标图像尺寸（高度不变，宽度按比例调整）
            2. 调用ele_stripes_delete函数处理每一行数据
            3. 更新内部FMI数据
        """
        if not self._is_data_loaded:
            self.read_data()

        # 计算目标尺寸：高度不变，宽度按比例调整
        original_height, original_width = self._data_fmi.shape
        target_width = int(original_width * width_ratio)
        target_shape = (original_height, target_width)

        self._logger.info(f"开始空白条带删除，目标形状: {target_shape}")

        try:
            image_value_count = np.size(self._data_fmi)
            # 应用空白条带删除算法
            self._data_fmi, valid_count = ele_stripes_delete(
                self._data_fmi,
                shape_target=target_shape,
                delete_pix=delete_pix
            )
            drop_ratio = 1 - valid_count/image_value_count

            self._logger.info("空白条带删除完成,空白带删除率：{}".format(drop_ratio))

        except Exception as e:
            self._logger.error(f"空白条带删除失败: {e}")
            raise

    def __get_fmi_texture_path(self):
        """
        初始化texture的保存路径
        :return: str path_texture
        """
        # 'path_fmi': r'F:\logging_workspace\桃镇1H\桃镇1H_DYNA_FULL_TEST.txt',
        if self.path_fmi.endswith((".txt")):
            return self.path_fmi.replace('.txt', '_texture_logging.csv')
        elif self.path_fmi.endswith((".csv")):
            return self.path_fmi.replace('.csv', '_texture_logging.csv')
        elif self.path_fmi.endswith((".jpg")):
            return self.path_fmi.replace('.jpg', '_texture_logging.csv')
        elif self.path_fmi.endswith((".png")):
            return self.path_fmi.replace('.png', '_texture_logging.csv')
        elif self.path_fmi.endswith((".bmp")):
            return self.path_fmi.replace('.bmp', '_texture_logging.csv')
        else:
            raise ValueError('path_fmi not end with ".txt" or ".csv"')

    def get_texture(self, texture_config: Optional[Dict] = None, fmi_texture_path: str = '') -> pd.DataFrame:
        """
        计算FMI数据的纹理特征（GLCM特征）
        Args:
            texture_config: 纹理计算配置字典
            fmi_texture_path: 纹理特征保存路径，为空时不保存

        Returns:
            包含深度和纹理特征的DataFrame

        Texture Features:
            - CON: 对比度 (Contrast)
            - DIS: 相异度 (Dissimilarity)
            - HOM: 同质性 (Homogeneity)
            - ENG: 能量 (Energy)
            - COR: 相关性 (Correlation)
            - ASM: 角二阶矩 (Angular Second Moment)
            - ENT: 熵 (Entropy)
        每种特征计算多个统计量：MEAN(均值), SUB(差值), X(水平方向), Y(垂直方向)
        """
        if not self._is_data_loaded:
            self.read_data()

        # 默认配置
        texture_config_default = {
            'level': 16,  # 灰度级别
            'distance': [2, 4],  # 像素距离
            # 'angles': [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],  # 角度方向
            'angles': [0, np.pi / 2],  # 角度方向
            'windows_length': 200,  # 窗口长度
            'windows_step': 100  # 滑动步长
        }
        texture_config_default.update(texture_config)
        texture_config = texture_config_default

        self._logger.info(f"开始纹理特征计算，配置: {texture_config}")

        # 生成纹理特征列名
        texture_headers = self.__generate_texture_headers()

        # 生成文件适配的csv文件路径，进行纹理信息的保存
        if len(fmi_texture_path) == 0:
            fmi_texture_path = self.__get_fmi_texture_path()

        # 看一下有没有这个文件，有的话，直接进行读取，没有的话，再进行计算
        if os.path.exists(fmi_texture_path):
            if fmi_texture_path.endswith('.csv'):
                self._logger.info('已经存在了纹理文件{}，进行直接的读取'.format(fmi_texture_path.split('/')[-1].split('\\')[-1]))
                texture_df = pd.read_csv(fmi_texture_path)
                return texture_df
        try:
            # 计算纹理特征
            texture_result = cal_images_texture(
                imgs=[self._data_fmi],
                depth=self._data_depth,
                windows=texture_config['windows_length'],
                step=texture_config['windows_step'],
                texture_config=texture_config,
                path_texture_saved=fmi_texture_path,
                texture_headers=texture_headers
            )

            self._logger.info(f"纹理特征计算完成，生成{len(texture_result)}个特征点")
            return texture_result

        except Exception as e:
            self._logger.error(f"纹理特征计算失败: {e}")
            raise

    def __generate_texture_headers(self) -> List[str]:
        """
        生成纹理特征列名列表

        Returns:
            纹理特征列名列表，包含各种统计量和方向的组合
        """
        # 基础特征名称
        base_features = ['CON', 'DIS', 'HOM', 'ENG', 'COR', 'ASM', 'ENT']
        # 方向中坠
        direction_suffixes = ['MEAN', 'SUB', 'X', 'Y']

        headers = []

        # 生成组合列名
        for direction in direction_suffixes:
            for feature in base_features:
                headers.append(f'{feature}_{direction}_{self.fmi_charter}')

        return headers

    def get_summary(self) -> Dict[str, any]:
        """
        获取FMI数据摘要信息

        Returns:
            包含各类统计信息的字典
        """
        summary = {
            'well_name': self._well_name,
            'fmi_charter': self.fmi_charter,
            'file_path': self.path_fmi,
            'is_loaded': self._is_data_loaded,
            'resolution': self._resolution,
            'path_fde': self.__get_fmi_fde_path(),
            'path_texture': self.__get_fmi_texture_path(),
        }

        if self._is_data_loaded:
            summary.update({
                'data_shape': self._data_fmi.shape,
                'depth_range': (np.min(self._data_depth), np.max(self._data_depth)),
                'depth_shape': self._data_depth.shape,
                'data_type': str(self._data_fmi.dtype),
                'path_fde': self.__get_fmi_fde_path(),
                'path_texture': self.__get_fmi_texture_path(),
            })

        return summary

    def __get_fmi_fde_path(self):
        # 'path_fmi': r'F:\logging_workspace\桃镇1H\桃镇1H_DYNA_FULL_TEST.txt',
        # return self.path_fmi.replace('.txt', '_fde_NMR.txt')
        return self.__remove_path_image_extension(self.path_fmi) + '_fde_NMR.txt'

    def get_fmi_fde(self, config_fde={}, path_fde='') -> np.ndarray:
        """
        获取FMI数据的fde分形谱

        Returns:
            分形谱的矩阵数据
        """
        if not self._is_data_loaded:
            self.read_data()

        if path_fde == '':
            path_fde = self.__get_fmi_fde_path()
            if os.path.exists(path_fde):
                if path_fde.endswith('.txt'):
                    image_fde_fmi = np.loadtxt(path_fde)
                elif path_fde.endswith('.csv'):
                    image_fde_fmi = pd.read_csv(path_fde)
                    image_fde_fmi = image_fde_fmi.values
                else:
                    self._logger.info('不支持的fde数据读取格式：{}'.format(path_fde))
                return image_fde_fmi
            else:
                self._logger.info('未存在该文件对应的fde格式，重新计算其fde谱')

        config_fde_default = {'windows_length': 150, 'windows_step': 50, 'processing_method': 'original'}
        if config_fde:
            config_fde_default.update(config_fde)

        fmi_result_list, fmi_multi_fde_list = cal_fmis_fractal_dimension_extended(
            fmi_dict={
                'depth': self._data_depth,  # 深度数据
                'fmis': [self._data_fmi.astype(np.uint8)],  # 电成像数据
            },
            windows_length=config_fde_default['windows_length'],            # 窗口长度：平衡纵向分辨率和统计可靠性
            windows_step=config_fde_default['windows_step'],                # 滑动步长：控制计算密度
            processing_method=config_fde_default['processing_method'],      # 图像预处理的方式：自适应二值化突出岩性边界 adaptive_binary  original
        )

        depth_array, image_fde = trans_NMR_as_Ciflog_file_type(fmi_multi_fde_list[0])
        alpha_f = np.hstack((depth_array.reshape((-1, 1)), image_fde.astype(np.float32)))
        np.savetxt(path_fde, alpha_f, delimiter='\t', comments='', fmt='%.4f')
        return alpha_f

    def fmi_preprocess(self, config_preprocess={}):
        image_new = self._data_fmi.copy()
        print(image_new.shape, image_new.dtype)

        pix_steps = config_preprocess.get('pix_steps', 10)
        pix_windows = config_preprocess.get('pix_windows', 250)
        way_preprocess = config_preprocess.get('way_preprocess', 'gabor')

        if way_preprocess.lower() == 'gabor':
            pass
        elif way_preprocess.lower() == 'gama':
            pass
        elif way_preprocess.lower() == 'wavelet':
            pass
        elif way_preprocess.lower() == 'filter':
            # 三大保边滤波器:导向滤波、双边滤波、最小二乘滤波  ######### 在去除噪声的同时，能最大限度保持边缘不被平滑
            # 各向同性滤波器:均值滤波、高斯滤波
            pass

        return image_new

def user_specific_test():
    """
    用户特定测试 - 使用用户提供的文件路径
    """
    print("\n" + "=" * 60)
    print("用户特定测试")
    print("=" * 60)

    # 用户提供的测试用例
    test_case = {
        # 'path_fmi': r'F:\logging_workspace\桃镇1H\桃镇1H_STAT_FULL.txt',
        # 'path_fmi': r'F:\logging_workspace\禄探\禄探_STAT.txt',
        'path_fmi': r'C:\Users\purem\Desktop\logging_workspace\禄探\禄探_STAT.txt',
        'fmi_charter': 'STAT'
    }

    print(f"测试文件: {test_case['path_fmi']}")
    print(f"仪器: {test_case['fmi_charter']}")
    print("-" * 50)

    try:
        # 创建FMI处理器实例
        test_FMI = DataFMI(
            path_fmi=test_case['path_fmi'],
            fmi_charter=test_case['fmi_charter']
        )

        # 执行用户要求的操作序列
        print(">>> 执行空白条带删除...")
        test_FMI.ele_stripes_delete()

        print(">>> 获取数据...")
        fmi_data, depth_data = test_FMI.get_data()

        print(f"FMI数据形状: {fmi_data.shape}")
        print(f"深度数据形状: {depth_data.shape}")

        if depth_data.size > 0:
            print(f"深度范围: {depth_data.min():.3f} - {depth_data.max():.3f}")

        # 显示数据摘要
        print("\n>>> 数据摘要:")
        summary = test_FMI.get_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")

        FMI_TEXTURE = test_FMI.get_texture(texture_config = {
                'level': 16,  # 灰度级别
                'distance': [2, 4],  # 像素距离
                # 'angles': [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],  # 角度方向
                'angles': [0, np.pi / 2],  # 角度方向
                'windows_length': 80,  # 窗口长度
                'windows_step': 10  # 滑动步长
            }
        )
        print(FMI_TEXTURE.describe())

        # FMI_FDE = test_FMI.get_fmi_fde()
        # print(FMI_FDE.shape)
        #
        # FMI_FDE_DICT = trans_fde_image_to_NMR_type(FMI_FDE)
        #
        # visualizer = WellLogVisualizer()
        # # 执行可视化
        # visualizer.visualize(
        #     logging_dict=None,
        #     fmi_dict={  # FMI图像数据
        #         'depth': test_FMI._data_depth,
        #         'image_data': [test_FMI._data_fmi] ,
        #         'title': ['FMI动态', 'FMI静态', 'DYNA_PRO', 'STAT_PRO']
        #     },
        #     NMR_dict=[FMI_FDE_DICT],
        #     NMR_Config={'X_LOG': [False, False], 'NMR_TITLE': ['α-fα-DYNA', 'α-fα-STAT'],
        #                 'X_LIMIT': [[0, 6.4], [0, 6.4]], 'Y_scaling_factor': 4},
        #     # depth_limit_config=[320, 380],                      # 深度限制
        #     figsize=(12, 10)  # 图形尺寸
        # )
        #
        # # 显示性能统计
        # stats = visualizer.get_performance_stats()
        # print("性能统计:", stats)

    except FileNotFoundError:
        print(f"文件不存在: {test_case['path_fmi']}")
        print("跳过该测试用例...")
    except Exception as e:
        print(f"测试失败: {e}")
        print("错误详情:", str(e))



if __name__ == '__main__':
    # """
    # 主程序入口
    # 执行顺序：
    # 1. 综合测试（使用模拟数据）
    # 2. 用户特定测试（使用用户提供的文件路径）
    # """
    #
    # # 设置日志级别
    # logging.basicConfig(level=logging.INFO)
    #
    # # 执行用户特定测试
    # user_specific_test()
    #
    # print("\n" + "=" * 60)
    # print("所有测试完成！")
    # print("=" * 60)

    data_fmi_test = DataFMI(path_fmi=r'C:\Users\Maple\Desktop\1-15\Step1\3728.2-3729.2.bmp')
    data_fmi_test.read_data()
    print(data_fmi_test.get_summary())

