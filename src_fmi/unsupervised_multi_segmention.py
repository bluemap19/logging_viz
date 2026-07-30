import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def kmeans_color_segmentation(
    image: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    max_iter: int = 300,
    n_init: int = 10,
    return_type: str = 'center',
    black_thresh: int = 5,
    white_thresh: int = 250,
) -> tuple:
    """
    使用 K-Means 对 RGB 彩色图像进行多阈值（多区域）分割。
    原理：将每个像素的 (R,G,B) 视为 3 维特征点，用 K-Means 聚成 n_clusters 个簇，
         簇中心即为颜色空间的"决策边界"（多阈值），每个像素被赋予最近中心的颜色/标签。
    剔除全黑/全白像素后，对 RGB 图像做 K-Means 多阈值分割。
    参数:
        image        : (H,W,3) uint8 RGB
        n_clusters   : 有效区域内的聚类数
        black_thresh : 低于此值且三通道都满足 → 判为黑点
        white_thresh : 高于此值且三通道都满足 → 判为白点
        return_type  : 'center' 返回着色图 / 'label' 返回标签图
    返回:
        segmented    : 分割结果
        labels       : (H,W) 标签图，-1 表示黑/白无效点
        invalid_mask : (H,W) bool，True 为被剔除的黑/白点
        centers      : (K,3) 有效区域聚类中心 (uint8)
    """
    # -------- 输入校验 --------
    assert image.ndim == 3 and image.shape[2] == 3, "输入必须是 HxWx3 的 RGB 图像"
    assert n_clusters >= 2, "n_clusters 必须 >= 2"

    H, W, C = image.shape
    # ---------- 1. 检测全黑 / 全白 ----------
    is_black = np.all(image < black_thresh, axis=2)
    is_white = np.all(image > white_thresh, axis=2)
    invalid_mask = is_black | is_white
    n_invalid = invalid_mask.sum()
    print(f"检测到全黑点: {is_black.sum()} | 全白点: {is_white.sum()} | 合计剔除: {n_invalid} ({n_invalid / (H * W) * 100:.2f}%)")

    # ---------- 2. 仅对有效像素聚类,归一化到 [0, 1] ----------
    if image.dtype == np.uint8 or image.max() > 1.0:
        valid_pixels = image[~invalid_mask].reshape(-1, C).astype(np.float32) / 255.0
    else:
        valid_pixels = image[~invalid_mask].reshape(-1, C).astype(np.float32)

    if len(valid_pixels) == 0:
        raise ValueError("所有像素都被判为黑/白，无法聚类！请调整阈值。")

    # -------- K-Means 聚类 --------
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init
    )
    labels_valid = kmeans.fit_predict(valid_pixels)

    centers = (kmeans.cluster_centers_ * 255).astype(np.uint8)

    # ---------- 3. 重建标签图 ----------
    labels = np.full((H, W), -1, dtype=np.int32)
    labels[~invalid_mask] = labels_valid

    # ---------- 4. 重建分割图 ----------
    if return_type == 'center':
        seg = np.zeros_like(image)
        seg[~invalid_mask] = centers[labels_valid]
        # 无效像素：保持原图（或设为灰色 [128,128,128]，按需求改）
        seg[invalid_mask] = image[invalid_mask]
    elif return_type == 'label':
        # 标签图：无效点标为 n_clusters（单独一类）
        seg = labels.copy()
        seg[invalid_mask] = n_clusters
    else:
        raise ValueError("return_type 必须是 'center' 或 'label'")
    
    return seg, labels, invalid_mask, centers


def generate_test_image(seed: int = 0) -> np.ndarray:
    """生成 200x200 的合成 RGB 测试图：4 个颜色方块 + 高斯噪声"""
    rng = np.random.default_rng(seed)
    H, W = 200, 200
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # 四个区域
    img[20:95, 20:95] = [220, 30, 30]  # 红
    img[20:95, 105:180] = [30, 200, 30]  # 绿
    img[105:180, 20:95] = [30, 30, 220]  # 蓝
    img[105:180, 105:180] = [230, 230, 20]  # 黄

    # 加入噪声（模拟真实图像干扰）
    noise = rng.integers(-5, 5, size=img.shape)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    return img


# ========== 执行测试 ==========
if __name__ == "__main__":
    # 1. 生成测试数据
    test_img = generate_test_image()

    # 2. 调用分割接口（K=4，因为我们知道有4个区域）
    seg_img, labels, mask, centers = kmeans_color_segmentation(
        test_img, n_clusters=4, return_type='center'
    )

    # 3. 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))

    axes[0].imshow(test_img)
    axes[0].set_title("Original (with noise)")
    axes[0].axis('off')

    axes[1].imshow(seg_img)
    axes[1].set_title("K-Means Segmented (K=4)")
    axes[1].axis('off')

    axes[2].imshow(labels, cmap='tab10')
    axes[2].set_title("Label Map")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

    # 4. 打印聚类中心（即学到的"多阈值"代表色）
    print("聚类中心 (RGB):")
    for i, c in enumerate(centers):
        print(f"  类别 {i}: R={c[0]:3d}, G={c[1]:3d}, B={c[2]:3d}")