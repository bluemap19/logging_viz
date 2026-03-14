import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'
from typing import Optional, Tuple, Union

from src_fmi.fmi_data_read import get_random_ele_data


class OTSU_Segmenter:
    """
    OTSU阈值分割器

    基于大津法(Otsu's method)实现的自动阈值分割算法，
    通过最大化类间方差找到图像的最佳分割阈值。

    属性：
        source_img: 原始输入图像
        threshold: 计算得到的最佳阈值
        between_class_variance: 最大类间方差
        histogram: 图像的灰度直方图
        histogram_prob: 归一化的灰度直方图(概率分布)
        foreground_mean: 前景(目标)的平均灰度
        background_mean: 背景的平均灰度
    """

    def __init__(self,):
        """
        初始化OTSU分割器

        参数：
            image: 输入图像，可以是灰度图(2D)或彩色图(3D)。
                   如果为彩色图，会自动转换为灰度图。
        """
        self.source_img = np.array([])
        self.threshold = 0
        self.between_class_variance = 0.0
        self.foreground_mean = 0.0
        self.background_mean = 0.0
        self.histogram = None
        self.histogram_prob = None

    def load_image(self, image: np.ndarray) -> None:
        """
        加载新图像

        参数：
            image: 新的输入图像
        """

        if not isinstance(image, np.ndarray):
            raise TypeError(f"输入必须是numpy数组，当前类型: {type(image)}")

        if len(image.shape) not in [2, 3]:
            raise ValueError(f"图像必须是2D(灰度)或3D(彩色)，当前维度: {len(image.shape)}")

        self.source_img = image
        # 重置之前的计算结果
        self._reset_computations()

    def _reset_computations(self) -> None:
        """重置所有计算属性"""
        self.threshold = 0
        self.between_class_variance = 0.0
        self.foreground_mean = 0.0
        self.background_mean = 0.0
        self.histogram = None
        self.histogram_prob = None

    def _to_grayscale(self, image: np.ndarray=None) -> np.ndarray:
        """
        将图像转换为灰度图
        参数：
            image: 输入图像
        返回：
            灰度图像
        """
        if image is None:
            image = self.source_img

        if len(image.shape) == 3:
            # 彩色图像转换为灰度
            if image.shape[2] == 3:  # BGR格式
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                # 如果是其他通道数，取第一个通道
                return image[:, :, 0]
        return image

    def compute_threshold(self, gray_levels: int = 256) -> int:
        """
        计算OTSU最佳阈值
        参数：
            gray_levels: 灰度级数，默认256
        返回：
            计算得到的最佳阈值
        异常：
            ValueError: 如果没有加载图像
        """
        if self.source_img is None:
            raise ValueError("未加载图像，请先使用load_image()方法加载图像")

        # 转换为灰度图
        gray_img = self._to_grayscale(self.source_img)
        # 计算灰度直方图
        self._compute_histogram(gray_img, gray_levels)
        # 计算最佳阈值
        self.threshold = self._find_optimal_threshold(gray_levels)

        return self.threshold

    def _compute_histogram(self, gray_img: np.ndarray, gray_levels: int) -> None:
        """
        计算图像的灰度直方图和概率分布
        参数：
            gray_img: 灰度图像
            gray_levels: 灰度级数
        """
        # 展平图像并转换为整型
        flat_image = gray_img.ravel().astype(np.uint8)

        # 计算直方图
        self.histogram = np.zeros(gray_levels, dtype=np.float64)
        for pixel_value in flat_image:
            self.histogram[pixel_value] += 1

        # 计算归一化直方图(概率分布)
        total_pixels = len(flat_image)
        self.histogram_prob = self.histogram / total_pixels

    def _find_optimal_threshold(self, gray_levels: int) -> int:
        """
        寻找最大化类间方差的阈值
        参数：
            gray_levels: 灰度级数
        返回：
            最佳阈值
        """
        max_variance = 0.0
        optimal_threshold = 0

        # 遍历所有可能的阈值
        for threshold in range(1, gray_levels):
            # 分割像素为背景(0~threshold-1)和前景(threshold~gray_levels-1)
            background_probs = self.histogram_prob[:threshold]
            foreground_probs = self.histogram_prob[threshold:]

            # 计算背景和前景的概率权重
            w_background = np.sum(background_probs)
            w_foreground = 1.0 - w_background

            # 如果任一类权重为0，跳过当前阈值
            if w_background == 0 or w_foreground == 0:
                continue

            # 计算背景平均灰度
            bg_indices = np.arange(threshold)
            self.background_mean = np.sum(background_probs * bg_indices) / w_background

            # 计算前景平均灰度
            fg_indices = np.arange(threshold, gray_levels)
            self.foreground_mean = np.sum(foreground_probs * fg_indices) / w_foreground

            # 计算类间方差
            variance = w_background * w_foreground * (self.background_mean - self.foreground_mean) ** 2

            # 更新最大方差和对应阈值
            if variance > max_variance:
                max_variance = variance
                optimal_threshold = threshold

        self.between_class_variance = max_variance
        return optimal_threshold

    def segment(self, image: Optional[np.ndarray] = None, max_value: int = 255) -> Tuple[int, np.ndarray]:
        """
        对图像进行OTSU阈值分割
        参数：
            image: 要分割的图像。如果为None，使用已加载的图像
            max_value: 二值化时的最大值，默认255(白色)
        返回：
            tuple: (阈值, 分割后的二值图像)
        异常：
            ValueError: 如果没有图像可用于分割
        """
        # 确定要处理的图像
        if image is not None:
            self.load_image(image)
        else:
            raise ValueError("没有可用于分割的图像，请提供图像或先加载图像")

        # 应用阈值分割
        self.source_img = self._to_grayscale(self.source_img)
        self.compute_threshold()
        _, binary_img = cv2.threshold(self.source_img, self.threshold, max_value, cv2.THRESH_BINARY)

        return self.threshold, binary_img

    def get_statistics(self) -> dict:
        """
        获取分割统计信息

        返回：
            包含各种统计信息的字典
        """
        if self.threshold == 0:
            raise ValueError("请先计算阈值(调用compute_threshold或segment方法)")

        return {
            'threshold': self.threshold,
            'between_class_variance': self.between_class_variance,
            'foreground_mean': self.foreground_mean,
            'background_mean': self.background_mean,
            'separation_measure': abs(self.foreground_mean - self.background_mean)
        }

    def visualize(self, show_histogram: bool = True) -> None:
        """
        可视化原始图像、直方图和分割结果

        参数：
            show_histogram: 是否显示直方图
            show_result: 是否显示分割结果
        """

        if self.source_img is None:
            raise ValueError("没有图像可显示，请先加载图像")

        if self.threshold == 0:
            self.compute_threshold()

        # 准备显示
        gray_img = self._to_grayscale(self.source_img)
        _, binary_img = self.segment(gray_img)

        # 创建图形
        fig, axes = plt.subplots(1, 3 if show_histogram else 2,
                                 figsize=(15, 5) if show_histogram else (10, 5))

        if not isinstance(axes, np.ndarray):
            axes = [axes]

        # 显示原始图像
        ax_idx = 0
        axes[ax_idx].imshow(gray_img, cmap='hot')
        axes[ax_idx].set_title(f'原始图像\n尺寸: {gray_img.shape}')
        axes[ax_idx].axis('off')

        # 显示直方图
        if show_histogram:
            ax_idx += 1
            axes[ax_idx].bar(range(len(self.histogram)), self.histogram,
                             width=1.0, color='blue', alpha=0.7)
            axes[ax_idx].axvline(x=self.threshold, color='red',
                                 linestyle='--', linewidth=2,
                                 label=f'阈值={self.threshold}')
            axes[ax_idx].set_title('灰度直方图')
            axes[ax_idx].set_xlabel('灰度值')
            axes[ax_idx].set_ylabel('像素数量')
            axes[ax_idx].legend()
            axes[ax_idx].grid(True, alpha=0.3)

        # 显示分割结果
        ax_idx += 1
        axes[ax_idx].imshow(binary_img, cmap='gray')

        # 添加统计信息
        stats = self.get_statistics()
        stats_text = (f"阈值: {stats['threshold']}\n"
                      f"类间方差: {stats['between_class_variance']:.2f}\n"
                      f"前景均值: {stats['foreground_mean']:.1f}\n"
                      f"背景均值: {stats['background_mean']:.1f}")

        axes[ax_idx].text(0.02, 0.98, stats_text,
                          transform=axes[ax_idx].transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[ax_idx].set_title('OTSU分割结果')
        axes[ax_idx].axis('off')

        plt.tight_layout()
        plt.show()


# ============================================================================
# 测试函数
# ============================================================================

def test_otsu_segmenter():
    """
    测试OTSU分割器的功能
    """
    print("=" * 50)
    print("OTSU分割器测试")
    print("=" * 50)

    # 方法1: 生成测试图像
    print("\n1. 测试生成图像分割...")

    # 创建一个简单的测试图像：左侧暗，右侧亮
    img_dyna, img_stat, data_depth = get_random_ele_data()

    # 创建分割器并分割
    otsu = OTSU_Segmenter()
    threshold, result = otsu.segment(img_stat)

    print(f"  计算得到的阈值: {threshold}")
    print(f"  类间方差: {otsu.between_class_variance:.2f}")
    print(f"  前景平均灰度: {otsu.foreground_mean:.1f}")
    print(f"  背景平均灰度: {otsu.background_mean:.1f}")

    # 可视化结果
    otsu.visualize()



# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("OTSU图像分割器演示")
    print("-" * 30)

    # 运行完整测试
    test_otsu_segmenter()









