# -*- coding: utf-8 -*-
"""
FMI/EMI 电成像测井割理分割算法实现
Coal Cleat Segmentation for FMI/EMI Images

作者：Cuka (OpenClaw Agent)
日期：2026-03-15
项目：coal_fmi_seam
环境：Anaconda PY38
"""
import cv2
import numpy as np
import os
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pywt
from datetime import datetime
import json
import matplotlib.pyplot as plt


class FMISegmentation:
    """FMI 图像割理分割类"""
    
    def __init__(self, method='tophat_otsu', params=None):
        """
        初始化分割器
        
        Parameters:
        -----------
        method : str
            分割方法：'tophat_otsu', 'adaptive', 'kmeans', 'gmm', 'canny', 'watershed', 'wavelet'
        params : dict
            方法参数
        """
        self.method = method
        self.params = params or {}
        self.results = {}
    
    def preprocess(self, image, method='gaussian'):
        """预处理：去噪"""
        if method == 'gaussian':
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == 'bilateral':
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == 'median':
            return cv2.medianBlur(image, 5)
        else:
            return image
    
    def convert_to_grayscale(self, image):
        """转换为灰度图像"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        return gray
    
    def segment_tophat_otsu(self, image, kernel_size=15):
        """
        顶帽变换 + Otsu 阈值（推荐用于亮色割理）
        
        Parameters:
        -----------
        image : ndarray
            输入图像（灰度）
        kernel_size : int
            结构元素大小（默认 15，适用于电成像割理特征）
        
        Returns:
        --------
        mask : ndarray
            二值分割掩膜
        """
        # 顶帽变换增强亮色特征
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        
        # Otsu 阈值
        ret, thresh = cv2.threshold(
            tophat, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        self.results['otsu_threshold'] = ret
        
        # # 后处理：去除小噪声
        # kernel_post = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_post, iterations=1)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_post, iterations=1)
        
        return thresh
    
    def segment_otsu(self, image):
        """Otsu 阈值分割"""
        ret, thresh = cv2.threshold(
            image, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        self.results['otsu_threshold'] = ret
        return thresh
    
    def segment_adaptive(self, image, block_size=11, C=10):
        """自适应阈值分割"""
        thresh = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size, C
        )
        return thresh
    
    def segment_kmeans(self, image, K=3):
        """K-means 聚类分割"""
        pixels = image.reshape((-1, 1)).astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        compactness, labels, centers = cv2.kmeans(
            pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # 选择最高灰度中心为割理（亮色）
        center_values = centers.flatten()
        cleat_label = np.argmax(center_values)
        
        mask_flat = (labels.flatten() == cleat_label).astype(np.uint8) * 255
        mask = mask_flat.reshape(image.shape)
        
        self.results['kmeans_centers'] = center_values.tolist()
        self.results['kmeans_compactness'] = compactness
        
        return mask
    
    def segment_gmm(self, image, n_components=3):
        """高斯混合模型分割"""
        pixels = image.reshape((-1, 1)).astype(np.float32)
        
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(pixels)
        
        labels = gmm.predict(pixels)
        
        # 选择最高均值组件为割理（亮色）
        means = gmm.means_.flatten()
        cleat_component = np.argmax(means)
        
        mask_flat = (labels == cleat_component).astype(np.uint8) * 255
        mask = mask_flat.reshape(image.shape)
        
        self.results['gmm_means'] = means.tolist()
        self.results['gmm_weights'] = gmm.weights_.tolist()
        
        return mask
    
    def segment_canny(self, image, threshold1=50, threshold2=150):
        """Canny 边缘检测"""
        edges = cv2.Canny(image, threshold1, threshold2)
        # 膨胀边缘使其可见
        kernel = np.ones((3,3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        return edges_dilated
    
    def segment_watershed(self, image):
        """分水岭分割"""
        # 二值化
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 距离变换
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        
        # 确定前景
        ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # 确定背景
        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(thresh, kernel, iterations=3)
        
        # 未知区域
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 创建标记
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # 分水岭
        markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
        
        # 创建掩膜
        mask = np.zeros_like(markers, dtype=np.uint8)
        mask[markers > 1] = 255
        
        return mask
    
    def segment_wavelet(self, image, wavelet='db4', level=3, threshold_factor=0.5):
        """小波变换分割"""
        coeffs = pywt.wavedec2(image, wavelet, level=level)
        
        sigma = np.median(np.abs(coeffs[-1][0])) / 0.6745
        threshold = threshold_factor * sigma
        
        new_coeffs = [coeffs[0]]
        for i in range(1, len(coeffs)):
            new_coeffs.append(tuple(
                pywt.threshold(c, value=threshold, mode='soft') for c in coeffs[i]
            ))
        
        reconstructed = pywt.waverec2(new_coeffs, wavelet)
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        ret, mask = cv2.threshold(
            reconstructed, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        return mask
    
    def postprocess(self, mask, kernel_size=3, operations=['open', 'close']):
        """后处理：形态学操作"""
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        processed = mask.copy()

        if (operations is not None) and (type(operations) is list):
            for op in operations:
                if op == 'open':
                    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
                elif op == 'close':
                    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
                elif op == 'dilate':
                    processed = cv2.dilate(processed, kernel, iterations=1)
                elif op == 'erode':
                    processed = cv2.erode(processed, kernel, iterations=1)
                else:
                    raise ValueError('Operation not supported')
        else:
            pass
        
        return processed
    
    def segment(self, image, enhance_method=None, seg_method='', post_method=['open', 'close']):
        """主分割流程"""
        gray = self.convert_to_grayscale(image)
        preprocessed = self.preprocess(gray, method=enhance_method)

        if preprocessed.dtype != np.uint8:
            preprocessed = preprocessed.astype(np.uint8)

        if len(seg_method) > 0:
            self.method = seg_method
        
        if self.method == 'tophat_otsu':
            mask = self.segment_tophat_otsu(preprocessed, kernel_size=self.params.get('kernel_size', 7))
        elif self.method == 'otsu':
            mask = self.segment_otsu(preprocessed)
        elif self.method == 'adaptive':
            mask = self.segment_adaptive(preprocessed, block_size=self.params.get('block_size', 11), C=self.params.get('C', 10))
        elif self.method == 'kmeans':
            mask = self.segment_kmeans(preprocessed, K=self.params.get('K', 3))
        elif self.method == 'gmm':
            mask = self.segment_gmm(preprocessed, n_components=self.params.get('n_components', 3))
        elif self.method == 'canny':
            mask = self.segment_canny(preprocessed, threshold1=self.params.get('threshold1', 50), threshold2=self.params.get('threshold2', 150))
        elif self.method == 'watershed':
            mask = self.segment_watershed(preprocessed)
        elif self.method == 'wavelet':
            mask = self.segment_wavelet(preprocessed, wavelet=self.params.get('wavelet', 'db4'), level=self.params.get('level', 3))
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        mask = self.postprocess(mask, kernel_size=3, operations=post_method)
        
        return mask


class SegmentationMetrics:
    """分割评估指标（无监督）"""
    
    def __init__(self, image, mask):
        self.image = image
        # 确保掩膜与图像尺寸一致
        if mask.shape != image.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        self.mask = mask.astype(bool)
        self.foreground = image[self.mask]
        self.background = image[~self.mask]
    
    def contrast_ratio(self):
        """对比度比"""
        if len(self.background) == 0:
            return 0.0
        mean_fg = np.mean(self.foreground) if len(self.foreground) > 0 else 0
        mean_bg = np.mean(self.background)
        return abs(mean_fg - mean_bg) / (mean_bg + 1e-6)
    
    def foreground_ratio(self):
        """前景占比"""
        return np.sum(self.mask) / self.mask.size
    
    def edge_density(self):
        """边缘密度"""
        edges = cv2.Canny((self.mask * 255).astype(np.uint8), 50, 150)
        return np.sum(edges > 0) / edges.size
    
    def compactness(self):
        """紧凑度"""
        contours, _ = cv2.findContours((self.mask * 255).astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 0.0
        
        total_compactness = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter * perimeter)
                total_compactness += compactness
        
        return total_compactness / len(contours)
    
    def cleat_density(self, image_width_px=1000):
        """割理密度估算（条/m）"""
        contours, _ = cv2.findContours((self.mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 过滤小轮廓
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
        num_cleats = len(valid_contours)
        # 假设图像宽度对应 1m
        density = num_cleats / (image_width_px / 1000)
        return density
    
    def all_metrics(self):
        """返回所有指标"""
        return {
            'contrast_ratio': self.contrast_ratio(),
            'foreground_ratio': self.foreground_ratio(),
            'edge_density': self.edge_density(),
            'compactness': self.compactness(),
            'cleat_density': self.cleat_density()
        }


def process_image(image_path, methods=None):
    """处理单张图像，测试多种方法"""
    if methods is None:
        methods = ['tophat_otsu', 'otsu', 'adaptive', 'kmeans', 'gmm', 'wavelet']
    
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ 无法读取图像：{image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_name = Path(image_path).stem
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    results = {
        'image': image_name,
        'timestamp': datetime.now().isoformat(),
        'methods': {}
    }
    
    # 创建可视化图
    n_cols = 3
    n_rows = (len(methods) + 5) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6*n_rows))
    axes = axes.flatten()
    
    # 显示原始图像
    axes[0].imshow(image_rgb)
    axes[0].set_title(f'Original Image\n{image_name}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 显示灰度图
    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title('Grayscale Image', fontsize=12)
    axes[1].axis('off')
    
    # 显示直方图
    axes[2].hist(gray.flatten(), bins=256, color='gray', alpha=0.7)
    axes[2].set_title('Intensity Histogram', fontsize=12)
    axes[2].set_xlabel('Gray Level')
    axes[2].set_ylabel('Frequency')
    axes[2].set_ylim(0, gray.size//50)  # 设置y轴上限为2000
    
    # 测试各种方法
    idx = 3
    method_configs = [
        ('tophat_otsu', 'TopHat + Otsu', None),
        ('otsu', 'Otsu Threshold', None),
        ('adaptive', 'Adaptive Threshold', None),
        ('kmeans', 'K-means (K=3)', None),
        ('gmm', 'GMM (n=3)', None),
        ('wavelet', 'Wavelet (db4)', None),
    ]
    
    for method, name, enhance in method_configs:
        if method not in methods:
            continue
        
        # 为 tophat_otsu 使用优化后的 kernel_size=15
        params = {'K': 3, 'n_components': 3}
        if method == 'tophat_otsu':
            params['kernel_size'] = 15
        else:
            params['kernel_size'] = 7
        
        seg = FMISegmentation(method=method, params=params)
        mask = seg.segment(image_rgb, enhance_method=enhance)
        
        # 计算指标
        metrics_calc = SegmentationMetrics(gray, mask)
        metrics = metrics_calc.all_metrics()
        results['methods'][method] = {
            'metrics': metrics,
            'params': seg.results
        }
        
        # 可视化
        axes[idx].imshow(mask, cmap='gray')
        title = f'{name}\n'
        title += f'CR: {metrics["contrast_ratio"]:.2f} | '
        title += f'FR: {metrics["foreground_ratio"]:.2%} | '
        title += f'CD: {metrics["cleat_density"]:.1f}/m'
        axes[idx].set_title(title, fontsize=10)
        axes[idx].axis('off')

        # 为后六张图片添加边框 - 不添加的话，看不到很多结果的边界，与背景板重合了
        # 使用 add_patch 添加矩形边框
        from matplotlib.patches import Rectangle
        # 获取当前坐标轴
        ax = axes[idx]
        # 获取当前坐标轴的数据范围
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # 计算矩形的宽度和高度
        width = xlim[1] - xlim[0]
        height = ylim[1] - ylim[0]
        # 添加红色边框矩形
        rect = Rectangle((xlim[0], ylim[0]), width, height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        idx += 1
    
    # 隐藏未使用的子图
    for i in range(idx, n_rows * n_cols):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.close()
    
    return results



def main():
    """主函数"""
    project_dir = Path(r'C:\Users\Maple\Desktop\test_imags')
    images_dir = project_dir
    
    # 查找所有测试图像
    test_images = sorted(list(images_dir.glob('test*.*')))
    
    if len(test_images) == 0:
        print("X 未找到测试图像！")
        return
    
    print("=" * 70)
    print("FMI 电成像割理分割测试")
    print("=" * 70)
    print(f"找到 {len(test_images)} 张测试图像\n")
    
    # 处理每张图像
    all_results = []
    for image_path in test_images:
        results = process_image(
            image_path,
            methods=['tophat_otsu', 'otsu', 'adaptive', 'kmeans', 'gmm', 'wavelet']
        )
        if results:
            all_results.append(results)



if __name__ == '__main__':
    main()
