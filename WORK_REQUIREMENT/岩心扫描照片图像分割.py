import os
import cv2
import matplotlib.pyplot as plt
from src_fmi.unsupervised_multi_segmention import kmeans_color_segmentation


if __name__ == '__main__':
    path_core_image = r'C:\Users\Maple\Desktop\core_scan\900.jpg'

    if os.path.exists(path_core_image):
        print(path_core_image+' image file exits successfully')
    else:
        print(path_core_image+' image file does not exist')
        exit(0)

    image_core_grey = cv2.imread(path_core_image, cv2.IMREAD_GRAYSCALE)
    image_core_rgb = cv2.imread(path_core_image, cv2.IMREAD_COLOR_RGB)
    print(image_core_grey.shape)     # (866, 866)
    print(image_core_rgb.shape)     # (866, 866, 3)

    # 双边滤波
    # d=9 邻域直径, sigmaColor=75 颜色空间标准差, sigmaSpace=75 坐标空间标准差
    image_core_rgb = cv2.bilateralFilter(
        image_core_rgb, d=7, sigmaColor=60, sigmaSpace=60
    )

    # ---- 调用：剔除黑白点 + K-Means 分割（K=4）----
    seg_img, labels, invalid_mask, centers = kmeans_color_segmentation(
        image_core_rgb,
        n_clusters=4,
        return_type='center',
        black_thresh=5,
        white_thresh=240,
    )

    print("聚类中心 (RGB):")
    for i, c in enumerate(centers):
        print(f"  类 {i}: R={c[0]:3d} G={c[1]:3d} B={c[2]:3d}")

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(image_core_rgb)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis('off')

    # 掩码可视化（红色=被剔除的黑/白点）
    mask_vis = image_core_rgb.copy()
    mask_vis[invalid_mask] = [255, 0, 0]
    axes[0, 1].imshow(mask_vis)
    axes[0, 1].set_title("Invalid Pixels (Red = Black/White)")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(seg_img)
    axes[1, 0].set_title("Segmented (Black/White kept original)")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(labels, cmap='tab10')
    axes[1, 1].set_title("Label Map (-1 = Invalid)")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.show()