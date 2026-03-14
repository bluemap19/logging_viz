import numpy as np
import pywt
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional, List
import warnings

warnings.filterwarnings('ignore')


class ElectricalImagingDWTProcessor:
    """
    电成像测井数据的小波变换处理类

    参数:
    ----------
    data : np.ndarray
        电成像测井数据，形状为(10000, 256)的图像数据
    wavelet : str, 可选
        小波基函数，默认为'db4' (Daubechies 4)
        其他常用选项: 'haar', 'db1'-'db20', 'sym2'-'sym20', 'coif1'-'coif5'
    """

    def __init__(self, data: np.ndarray, wavelet: str = 'db4'):
        """
        初始化函数
        """
        self.data = data.astype(np.float32)
        self.wavelet = wavelet
        self.original_shape = data.shape

        # 常用的图像处理小波
        self.common_wavelets = {
            'haar': 'Haar小波',
            'db1': 'Daubechies 1',
            'db4': 'Daubechies 4 (默认)',
            'db8': 'Daubechies 8',
            'sym4': 'Symlets 4',
            'coif1': 'Coiflets 1',
            'bior2.2': 'Biorthogonal 2.2',
        }

    def decompose(self,
                  level: int = 3,
                  mode: str = 'symmetric',
                  normalize: bool = True) -> dict:
        """
        执行多层小波分解

        参数:
        ----------
        level : int, 可选
            分解层数，默认为3
        mode : str, 可选
            边界处理模式，默认为'symmetric'
            可选值: 'zero', 'constant', 'symmetric', 'periodic', 'smooth'
        normalize : bool, 可选
            是否对输入数据进行归一化，默认为True

        返回:
        ----------
        dict: 包含小波系数的字典
            'approximation': 近似系数 (低频)
            'details': 细节系数列表 (高频) [水平, 垂直, 对角线]
        """
        # 数据预处理
        if normalize:
            data_norm = (self.data - np.mean(self.data)) / (np.std(self.data) + 1e-8)
        else:
            data_norm = self.data

        # 执行多层小波分解
        coeffs = pywt.wavedec2(data_norm,
                               wavelet=self.wavelet,
                               level=level,
                               mode=mode)

        # 组织系数
        coefficients = {
            'approximation': coeffs[0],  # 低频近似系数
            'details': coeffs[1:],  # 高频细节系数列表
            'level': level,
            'shape': self.original_shape
        }

        return coefficients

    def reconstruct(self,
                    coefficients: dict,
                    mode: str = 'symmetric') -> np.ndarray:
        """
        从小波系数重建图像

        参数:
        ----------
        coefficients : dict
            小波系数字典，包含'approximation'和'details'
        mode : str, 可选
            边界处理模式，默认为'symmetric'

        返回:
        ----------
        np.ndarray: 重建的图像数据
        """
        # 重新组织系数列表
        coeff_list = [coefficients['approximation']] + coefficients['details']

        # 重建图像
        reconstructed = pywt.waverec2(coeff_list,
                                      wavelet=self.wavelet,
                                      mode=mode)

        # 调整形状（由于边界处理可能导致形状略有变化）
        if reconstructed.shape != self.original_shape:
            reconstructed = reconstructed[:self.original_shape[0], :self.original_shape[1]]

        return reconstructed

    def extract_frequency_components(self,
                                     level: int = 3,
                                     freq_type: str = 'high',
                                     component: str = 'all',
                                     threshold: float = None) -> Tuple[np.ndarray, dict]:
        """
        提取指定频率成分

        参数:
        ----------
        level : int, 可选
            分解层数，默认为3
        freq_type : str, 可选
            频率类型:
                'high' - 高频成分
                'low' - 低频成分
                'both' - 同时提取
        component : str, 可选
            高频成分类型 (仅当freq_type='high'时有效):
                'horizontal' - 水平方向
                'vertical' - 垂直方向
                'diagonal' - 对角线方向
                'all' - 所有方向
        threshold : float, 可选
            阈值参数，用于软/硬阈值去噪

        返回:
        ----------
        Tuple[np.ndarray, dict]:
            处理后的图像，以及包含提取信息的字典
        """
        # 执行分解
        coeffs = self.decompose(level=level, normalize=True)

        if freq_type == 'low':
            # 只保留低频成分
            filtered_coeffs = {
                'approximation': coeffs['approximation'],
                'details': [(np.zeros_like(d[0]),
                             np.zeros_like(d[1]),
                             np.zeros_like(d[2])) for d in coeffs['details']]
            }

        elif freq_type == 'high':
            # 只保留高频成分
            filtered_coeffs = {
                'approximation': np.zeros_like(coeffs['approximation']),
                'details': coeffs['details']
            }

            # 如果指定了具体的高频方向
            if component != 'all':
                filtered_details = []
                for detail in coeffs['details']:
                    h, v, d = detail
                    if component == 'horizontal':
                        filtered_details.append((h, np.zeros_like(v), np.zeros_like(d)))
                    elif component == 'vertical':
                        filtered_details.append((np.zeros_like(h), v, np.zeros_like(d)))
                    elif component == 'diagonal':
                        filtered_details.append((np.zeros_like(h), np.zeros_like(v), d))
                filtered_coeffs['details'] = filtered_details

        else:  # 'both'
            # 保留所有成分
            filtered_coeffs = coeffs

        # 应用阈值去噪
        if threshold is not None and threshold > 0:
            filtered_coeffs = self._apply_threshold(filtered_coeffs, threshold)

        # 重建图像
        reconstructed = self.reconstruct(filtered_coeffs)

        # 信息统计
        info = {
            'frequency_type': freq_type,
            'component': component,
            'wavelet_level': level,
            'threshold_applied': threshold is not None,
            'energy_ratio': self._calculate_energy_ratio(coeffs, filtered_coeffs)
        }

        return reconstructed, info

    def _apply_threshold(self, coefficients: dict, threshold: float) -> dict:
        """
        应用阈值处理
        """
        thresholded_coeffs = coefficients.copy()

        # 对细节系数应用软阈值
        for i, (h, v, d) in enumerate(thresholded_coeffs['details']):
            # 软阈值处理
            h_thresh = pywt.threshold(h, threshold, mode='soft')
            v_thresh = pywt.threshold(v, threshold, mode='soft')
            d_thresh = pywt.threshold(d, threshold, mode='soft')

            thresholded_coeffs['details'][i] = (h_thresh, v_thresh, d_thresh)

        return thresholded_coeffs

    def _calculate_energy_ratio(self,
                                original_coeffs: dict,
                                filtered_coeffs: dict) -> float:
        """
        计算保留的能量比例
        """

        def calc_energy(coeffs):
            energy = np.sum(coeffs['approximation'] ** 2)
            for h, v, d in coeffs['details']:
                energy += np.sum(h ** 2) + np.sum(v ** 2) + np.sum(d ** 2)
            return energy

        orig_energy = calc_energy(original_coeffs)
        filt_energy = calc_energy(filtered_coeffs)

        return filt_energy / (orig_energy + 1e-8)

    def multi_scale_analysis(self,
                             levels: List[int] = [1, 2, 3],
                             freq_type: str = 'high') -> dict:
        """
        多尺度分析

        参数:
        ----------
        levels : List[int]
            要分析的尺度列表
        freq_type : str
            频率类型

        返回:
        ----------
        dict: 多尺度分析结果
        """
        results = {}

        for level in levels:
            reconstructed, info = self.extract_frequency_components(
                level=level,
                freq_type=freq_type
            )
            results[level] = {
                'reconstructed': reconstructed,
                'info': info
            }

        return results

    def visualize_results(self,
                          original: np.ndarray,
                          processed: np.ndarray,
                          info: dict,
                          figsize: Tuple[int, int] = (15, 8)):
        """
        可视化处理结果
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # 原始图像
        im1 = axes[0].imshow(original, aspect='auto', cmap='gray')
        axes[0].set_title('原始电成像数据')
        axes[0].set_xlabel('道号')
        axes[0].set_ylabel('深度')
        plt.colorbar(im1, ax=axes[0], shrink=0.6)

        # 处理后的图像
        im2 = axes[1].imshow(processed, aspect='auto', cmap='gray')
        title = f"处理结果: {info['frequency_type']}频"
        if info['frequency_type'] == 'high':
            title += f" ({info['component']})"
        axes[1].set_title(title)
        axes[1].set_xlabel('道号')
        axes[1].set_ylabel('深度')
        plt.colorbar(im2, ax=axes[1], shrink=0.6)

        # 差异图像
        diff = processed - original
        im3 = axes[2].imshow(diff, aspect='auto', cmap='RdBu_r')
        axes[2].set_title('差异图像')
        axes[2].set_xlabel('道号')
        axes[2].set_ylabel('深度')
        plt.colorbar(im3, ax=axes[2], shrink=0.6)

        plt.suptitle(f"小波变换分析 (小波: {self.wavelet}, 层数: {info['wavelet_level']})")
        plt.tight_layout()
        plt.show()

        # 打印处理信息
        print(f"处理信息:")
        print(f"- 频率类型: {info['frequency_type']}")
        print(f"- 方向成分: {info['component']}")
        print(f"- 小波层数: {info['wavelet_level']}")
        print(f"- 能量保留比例: {info['energy_ratio']:.2%}")
        print(f"- 阈值处理: {info['threshold_applied']}")


# 使用示例函数
def process_electrical_imaging_dwt(data: np.ndarray,
                                   wavelet: str = 'db4',
                                   level: int = 3,
                                   freq_type: str = 'high',
                                   component: str = 'all',
                                   threshold: float = None,
                                   visualize: bool = True) -> dict:
    """
    电成像测井数据小波变换处理主函数

    参数:
    ----------
    data : np.ndarray
        输入数据，形状为(10000, 256)
    wavelet : str, 可选
        小波基函数
    level : int, 可选
        分解层数
    freq_type : str, 可选
        频率类型: 'high', 'low', 'both'
    component : str, 可选
        高频方向: 'all', 'horizontal', 'vertical', 'diagonal'
    threshold : float, 可选
        阈值参数
    visualize : bool, 可选
        是否可视化结果

    返回:
    ----------
    dict: 处理结果
    """
    # 创建处理器实例
    processor = ElectricalImagingDWTProcessor(data, wavelet=wavelet)

    # 执行小波变换处理
    processed_data, info = processor.extract_frequency_components(
        level=level,
        freq_type=freq_type,
        component=component,
        threshold=threshold
    )

    # 可视化结果
    if visualize:
        processor.visualize_results(data, processed_data, info)

    # 返回结果
    result = {
        'processed_data': processed_data,
        'info': info,
        'original_data': data,
        'wavelet': wavelet
    }

    return result

# 示例使用代码
if __name__ == "__main__":
    # 生成示例数据 (实际使用时替换为真实数据)
    sample_data = np.random.randn(10000, 256)

    # 方法1: 简单使用
    result = process_electrical_imaging_dwt(
        data=sample_data,
        wavelet='db4',
        level=5,
        freq_type='high',  # 提取高频成分
        component='all',
        threshold=0.1,  # 阈值去噪
        visualize=True
    )

    # 方法2: 使用处理器类
    processor = ElectricalImagingDWTProcessor(sample_data, wavelet='db4')

    # 提取低频成分 (用于背景分析)
    low_freq_data, low_info = processor.extract_frequency_components(
        level=4,
        freq_type='low'
    )
    processor.visualize_results(sample_data, low_freq_data, low_info)

    # 提取特定方向的高频成分
    horizontal_data, horiz_info = processor.extract_frequency_components(
        level=3,
        freq_type='high',
        component='horizontal',
        threshold=0.05
    )
    processor.visualize_results(sample_data, horizontal_data, horiz_info)

    # 多尺度分析
    multi_scale_results = processor.multi_scale_analysis(
        levels=[1, 2, 3, 4],
        freq_type='high'
    )
