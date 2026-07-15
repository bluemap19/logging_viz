# fmi_data_save.py
import numpy as np
import cv2
import pandas as pd
import os


def fmi_data_save(
    save_path: str,
    img_data: np.ndarray,
    depth_data: np.ndarray,
    header_lines=None
):
    """
    电成像测井数据保存接口（与 get_ele_data_from_path 对应）

    参数:
    - save_path: 保存路径，支持 .png / .txt / .csv
    - img_data: 电成像数据 (N × M)
    - depth_data: 深度数据 (N,) 或 (N, 1)
    - header_lines: TXT 文件头（仅 TXT 有效）
    """

    if img_data is None or depth_data is None:
        raise ValueError("img_data 或 depth_data 为空")

    depth_data = np.asarray(depth_data).reshape(-1)

    if img_data.shape[0] != depth_data.shape[0]:
        raise ValueError("深度点数与电成像行数不一致")

    # # ==========================
    # # 1. 保存为图像
    # # ==========================
    # if save_path.lower().endswith(('.png', '.jpg')):
    #     # 归一化到 0–255
    #     img_norm = img_data.astype(np.float32)
    #     img_norm -= img_norm.min()
    #     img_norm /= (img_norm.max() + 1e-8)
    #     img_uint8 = (img_norm * 255).astype(np.uint8)
    #
    #     # 文件名中加入深度信息
    #     start_dep = depth_data[0]
    #     end_dep = depth_data[-1]
    #     base = save_path.split('.')[0]
    #     ext = save_path.split('.')[-1]
    #     save_path = f"{base}_{start_dep:.4f}_{end_dep:.4f}.{ext}"
    #
    #     cv2.imwrite(save_path, img_uint8)
    #     print(f"[SAVE] 图像保存成功: {save_path}")

    # ==========================
    # 1. 保存为图像（✅ 终极稳定版）
    # ==========================
    if save_path.lower().endswith(('.png', '.jpg')):
        # 归一化
        img_norm = img_data.astype(np.float32)
        img_min, img_max = img_norm.min(), img_norm.max()

        if img_max - img_min < 1e-8:
            raise ValueError("图像数据全为常数，无法归一化")

        img_norm = (img_norm - img_min) / (img_max - img_min)
        img_uint8 = (img_norm * 255).astype(np.uint8)

        # 深度信息
        start_dep = depth_data[0]
        end_dep = depth_data[-1]

        base, ext = os.path.splitext(save_path)
        save_path_new = f"{base}_{start_dep:.4f}_{end_dep:.4f}{ext}"

        # ✅ 确保目录存在
        os.makedirs(os.path.dirname(save_path_new), exist_ok=True)

        # ✅ 方法一：OpenCV（优先）
        try:
            # 使用 imencode + tofile（支持中文）
            success = cv2.imencode(ext, img_uint8)[1].tofile(save_path_new)
            if success:
                print(f"[SAVE] OpenCV 保存成功: {save_path_new}")
            else:
                raise IOError("OpenCV imencode 失败")
        except Exception as e:
            print(f"[WARN] OpenCV 保存失败: {e}")

            # ✅ 方法二：PIL（兜底，100% 成功）
            try:
                from PIL import Image
                pil_img = Image.fromarray(img_uint8)
                pil_img.save(save_path_new)
                print(f"[SAVE] PIL 保存成功: {save_path_new}")
            except Exception as e2:
                raise IOError(f"图像保存彻底失败: {save_path_new}, 错误: {e2}")



    # ==========================
    # 2. 保存为 TXT
    # ==========================
    elif save_path.lower().endswith('.txt'):
        with open(save_path, 'w', encoding='GBK') as f:
            if header_lines:
                for line in header_lines:
                    f.write(line + '\n')
            else:
                f.write("# FMI ELECTRIC IMAGE DATA\n")
                f.write(f"# DEPTH START: {depth_data[0]:.6f}\n")
                f.write(f"# DEPTH END  : {depth_data[-1]:.6f}\n")
                f.write("# COLUMNS: DEPTH + IMAGE DATA\n")
                f.write("# UNIT: NORMALIZED\n")
                f.write("# FORMAT: TAB DELIMITED\n")
                f.write("# VERSION: 1.0\n")
                f.write("#\n")

            for i in range(len(depth_data)):
                line = f"{depth_data[i]:.6f}"
                for v in img_data[i]:
                    line += f"\t{v:.6f}"
                f.write(line + "\n")

        print(f"[SAVE] TXT 保存成功: {save_path}")

    # ==========================
    # 3. 保存为 CSV
    # ==========================
    elif save_path.lower().endswith('.csv'):
        df = pd.DataFrame(img_data)
        df.insert(0, 'DEPTH', depth_data)
        df.to_csv(save_path, index=False)
        print(f"[SAVE] CSV 保存成功: {save_path}")
    else:
        raise ValueError("不支持的文件格式")

def generate_random_fmi_data(
    num_depths=500,
    num_pads=250,
    depth_start=3600.0,
    depth_step=0.125
):
    """
    生成随机电成像数据（模拟真实 FMI）
    """
    depth = np.linspace(
        depth_start,
        depth_start + num_depths * depth_step,
        num_depths,
        endpoint=False
    )

    # 模拟地层 + 噪声
    img = np.random.randn(num_depths, num_pads)
    img += np.sin(np.linspace(0, 10, num_depths)).reshape(-1, 1)
    img += 0.1 * np.random.randn(*img.shape)

    return img, depth


if __name__ == "__main__":

    # 生成随机数据
    img_data, depth_data = generate_random_fmi_data()

    # 保存为 PNG
    fmi_data_save(
        save_path=r"F:\logging_workspace\塬22\test_fmi.png",
        img_data=img_data,
        depth_data=depth_data
    )

    # 保存为 TXT
    fmi_data_save(
        save_path=r"F:\logging_workspace\塬22\test_fmi.txt",
        img_data=img_data,
        depth_data=depth_data
    )

    # 保存为 CSV
    fmi_data_save(
        save_path=r"F:\logging_workspace\塬22\test_fmi.csv",
        img_data=img_data,
        depth_data=depth_data
    )