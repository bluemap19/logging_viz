from src_fmi.fmi_data_read import get_ele_data_from_path
import numpy as np
from scipy import interpolate
from scipy.ndimage import distance_transform_edt
import cv2

from src_fmi.fmi_data_save import fmi_data_save
from src_plot.well_logging_viz.data_manager import LoggingDataManager
from src_plot.well_logging_viz.data_visulization import WellLogVisualizer


def repair_image(input_image, mask, method='inpainting', telea_radius=5, ns_radius=2):
    """
    电成像图像修复统一接口，整合所有修复算法

    参数:
        input_image: 原始图像数据，形状为(H, W)，支持float32/float64类型
        mask: 掩膜数据，形状与input_image一致，非零值表示需要修复的区域
        method: 修复方法，可选值：
            - 'inpainting': 兼容旧接口，等价于'telea'
            - 'telea': 纯Telea算法，速度快，适合小缺损填充
            - 'ns': 纯Navier-Stokes算法，保边缘，适合纹理修复
            - 'hybrid': 【新增】两阶段级联修复：Telea初修+NS精修，兼顾速度与纹理连续性
            - 'interpolation': 径向基函数全局插值，仅适合极小缺损，大图像慎用
            - 'nearest': 最近邻填充，速度快，适合孤立点修复
            - 'column_based': 列方向线性插值，最适合测井数据（深度方向连续性优先）
            - 'row_based': 行方向线性插值，仅适合方位方向连续的场景
        telea_radius: Telea算法邻域半径，默认5，仅在method为telea/hybrid时生效
        ns_radius: NS算法邻域半径，默认2，仅在method为ns/hybrid时生效

    返回:
        repaired_image: 修复后的图像，数据类型与输入一致
    """
    # 保存原始数值范围，用于后处理逆归一化
    img_min, img_max = input_image.min(), input_image.max()
    input_image = input_image.astype(np.float32)
    mask = mask.astype(np.uint8)
    repaired_image = None

    # 生成OpenCV兼容的掩膜（255代表待修复区域）
    mask_cv = np.where(mask > 0, 255, 0).astype(np.uint8)

    if method in ['inpainting', 'telea']:
        # Telea算法：快速填充小缺损
        norm_img = (input_image - img_min) / (img_max - img_min + 1e-8) * 255
        norm_img = norm_img.astype(np.uint8)
        repaired_norm = cv2.inpaint(norm_img, mask_cv, telea_radius, cv2.INPAINT_TELEA)
        repaired_image = repaired_norm.astype(np.float32) / 255 * (img_max - img_min) + img_min

    elif method == 'ns':
        # NS算法：保边缘纹理修复
        norm_img = (input_image - img_min) / (img_max - img_min + 1e-8) * 255
        norm_img = norm_img.astype(np.uint8)
        repaired_norm = cv2.inpaint(norm_img, mask_cv, ns_radius, cv2.INPAINT_NS)
        repaired_image = repaired_norm.astype(np.float32) / 255 * (img_max - img_min) + img_min

    elif method == 'hybrid':
        # 【核心修改】两阶段级联修复：Telea初修 + NS精修
        # 第一阶段：Telea快速填充大范围缺损
        norm_img = (input_image - img_min) / (img_max - img_min + 1e-8) * 255
        norm_img = norm_img.astype(np.uint8)
        stage1 = cv2.inpaint(norm_img, mask_cv, telea_radius, cv2.INPAINT_TELEA)
        # 第二阶段：NS精修边缘，恢复纹理连续性
        repaired_norm = cv2.inpaint(stage1, mask_cv, ns_radius, cv2.INPAINT_NS)
        repaired_image = repaired_norm.astype(np.float32) / 255 * (img_max - img_min) + img_min

    elif method == 'interpolation':
        # 径向基函数全局插值（大图像慎用，易内存溢出）
        repaired_image = input_image.copy()
        valid_mask = mask == 0
        coords = np.array(np.nonzero(valid_mask)).T
        values = input_image[valid_mask]
        missing_coords = np.array(np.nonzero(mask)).T

        if len(missing_coords) > 0 and len(coords) > 0:
            rbf = interpolate.Rbf(coords[:, 1], coords[:, 0], values,
                                  function='linear', smooth=0)
            repaired_values = rbf(missing_coords[:, 1], missing_coords[:, 0])
            repaired_image[missing_coords[:, 0], missing_coords[:, 1]] = repaired_values

    elif method == 'nearest':
        # 最近邻填充
        repaired_image = input_image.copy()
        dist, indices = distance_transform_edt(mask, return_indices=True)
        repaired_image[mask > 0] = input_image[indices[0][mask > 0], indices[1][mask > 0]]

    elif method == 'column_based':
        # 列方向线性插值（测井数据首选）
        repaired_image = input_image.copy()
        for col in range(input_image.shape[1]):
            col_data = input_image[:, col]
            col_mask = mask[:, col]
            if np.any(col_mask):
                valid_idx = np.where(col_mask == 0)[0]
                invalid_idx = np.where(col_mask > 0)[0]
                if len(valid_idx) > 1 and len(invalid_idx) > 0:
                    repaired_col = col_data.copy()
                    repaired_col[invalid_idx] = np.interp(invalid_idx, valid_idx, col_data[valid_idx])
                    repaired_image[:, col] = repaired_col

    elif method == 'row_based':
        # 行方向线性插值（仅适合方位连续场景）
        repaired_image = input_image.copy()
        for row in range(input_image.shape[0]):
            row_data = input_image[row, :]
            row_mask = mask[row, :]
            if np.any(row_mask):
                valid_idx = np.where(row_mask == 0)[0]
                invalid_idx = np.where(row_mask > 0)[0]
                if len(valid_idx) > 1 and len(invalid_idx) > 0:
                    repaired_row = row_data.copy()
                    repaired_row[invalid_idx] = np.interp(invalid_idx, valid_idx, row_data[valid_idx])
                    repaired_image[row, :] = repaired_row

    else:
        raise ValueError(
            f"未知的修复方法: {method}，可选值为{'inpainting' / 'telea' / 'ns' / 'hybrid' / 'interpolation' / 'nearest' / 'column_based' / 'row_based'}")

    return repaired_image


if __name__ == '__main__':
    dyna_image, depth_dyna = get_ele_data_from_path(strname=r'Z:\logging_workspace\姬119H2\姬119H2_FMI_DYNA.txt')
    mask, depth_mask = get_ele_data_from_path(strname=r'Z:\logging_workspace\姬119H2\target_mask.txt')
    print(f"原始图像形状: {dyna_image.shape}")
    print(f"掩膜形状: {mask.shape}")
    print(f"需要修复的像素数量: {np.sum(mask > 0)}")

    # 方法1: Telea单算法修复
    repaired1 = repair_image(dyna_image, mask, method='telea', telea_radius=5)

    # 方法2: 【新增】两阶段级联修复（Hybrid）
    repaired2 = repair_image(dyna_image, mask, method='hybrid', telea_radius=5, ns_radius=2)

    # 方法3: 列方向插值（测井数据首选）
    repaired3 = repair_image(dyna_image, mask, method='column_based')

    # 可视化对比
    LDM = LoggingDataManager(
        fmi_data={'depth': depth_dyna, 'image_data': [
            255 - dyna_image,
            255 - mask,
            255 - repaired1,
            255 - repaired2,
            255 - repaired3
        ]},
    )
    well_viewer = WellLogVisualizer(
        LDM,
        config_fmi={
            'color_map': 'hot',
            'title_fmi': ['原始FMI', '缺损掩膜', 'Telea修复', 'Hybrid级联修复', '列插值修复']
        }
    )
    well_viewer.visualize()

    # # 保存结果
    # fmi_data_save(r'F:\logging_workspace\姬119H2\姬119H2_FMI_DYNA_TELEA.txt', repaired1, depth_dyna)
    # fmi_data_save(r'F:\logging_workspace\姬119H2\姬119H2_FMI_DYNA_HYBRID.txt', repaired2, depth_dyna)
    # fmi_data_save(r'F:\logging_workspace\姬119H2\姬119H2_FMI_DYNA_COLUMN.txt', repaired3, depth_dyna)