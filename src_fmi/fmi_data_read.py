import numpy as np
import pandas as pd
import cv2

def get_ele_data_from_path(strname=r'D:\Data\target\107S\YS107_FMI_BorEID_FA.txt', depth=[-1.0, -1.0]):
    """
    电成像测井资料通用读取接口

    功能: 从文件路径读取电成像数据，支持多种格式（PNG/JPG图像、TXT文本、CSV表格）
          自动提取深度信息，支持按深度范围截取数据

    参数:
    - strname: 文件路径，支持 .png/.jpg/.txt/.csv 格式
    - depth: 深度范围 [起始深度, 结束深度]，负值表示不进行深度截取

    返回:
    - img_data: 电成像数据矩阵 (深度点数 × 电极数)
    - depth_data: 对应的深度数据 (深度点数 × 1)

    原理说明:
    1. 根据文件扩展名判断数据类型
    2. 图像文件: 从文件名解析深度信息，生成等间隔深度序列
    3. 文本文件: 读取格式化数据，第一列为深度，其余为电成像值
    4. 支持按指定深度范围截取数据子集
    """

    # 初始化返回变量
    img_data = np.array([])  # 电成像数据矩阵
    depth_data = np.array([])  # 深度数据向量

    # ============================================================================
    # 1. 处理图像文件 (.png, .jpg)
    # ============================================================================
    if strname.endswith('.png') or strname.endswith('.jpg'):
        # 读取灰度图像
        img_data = np.array(cv2.imread(strname, cv2.IMREAD_GRAYSCALE))

        # 数据检查: 验证图像是否成功加载
        if img_data.shape[0] == 0:
            print(f'图像数据为空: {strname}')
            exit(0)  # 严重错误，直接退出程序

        # 创建深度数据占位符 (与图像行数相同)
        depth_data = np.arange(0, img_data.shape[0], dtype=np.float32).reshape(-1, 1)

        # 尝试从文件名解析深度信息
        # 文件名格式示例: wellname_startDepth_endDepth_suffix.png
        if len(strname.split('_')) < 4:
            print(f'文件 {strname} 不包含深度信息，使用默认深度序列')
            return img_data, depth_data

        try:
            # 提取文件名（去除路径）
            filename = strname.split('\\')[-1].split('/')[-1]
            # 解析深度信息: 假设格式为 ..._startDepth_endDepth_...
            parts = filename.split('_')
            startdep = float(parts[2])  # 起始深度
            enddep = float(parts[3])  # 结束深度

            # 计算深度步长并生成深度序列
            Step = (enddep - startdep) / img_data.shape[0]
            for i in range(img_data.shape[0]):
                depth_data[i] = i * Step + startdep

        except Exception as e:
            # 深度解析失败，使用默认深度序列 (0, 1, 2, ...)
            print(f'文件 {strname} 深度信息解析失败: {e}，使用默认深度序列')
            depth_data = np.arange(0, img_data.shape[0]).reshape(-1, 1)

    # ============================================================================
    # 2. 处理文本文件 (.txt) - 电成像标准数据格式
    # ============================================================================
    elif strname.endswith('.txt'):
        # 读取文本数据，跳过前8行头文件，使用制表符分隔
        AllData = np.loadtxt(strname, delimiter='\t', skiprows=8, encoding='GBK')

        # 数据检查: 验证数据是否成功加载
        if AllData.shape[0] == 0:
            print(f'文本数据为空: {strname}')
            exit(0)  # 严重错误，直接退出程序

        # 文本文件格式: 第一列为深度，其余列为电成像测量值
        img_data = AllData[:, 1:]  # 电成像数据 (排除深度列)
        depth_data = AllData[:, 0].reshape((AllData.shape[0], 1))  # 深度数据 (第一列)

    # ============================================================================
    # 3. 处理CSV文件 (.csv) - 待实现功能
    # ============================================================================
    elif strname.endswith('.csv'):
        print('CSV数据格式待实现')
        try:
            # 读取CSV文件，使用pandas处理
            df = pd.read_csv(strname)

            # 数据检查: 验证CSV文件是否成功加载
            if df.empty:
                print(f'CSV数据为空: {strname}')
                exit(0)
            # 验证列数足够（至少2列：深度列+至少1个数据列）
            if df.shape[1] < 2:
                print(f'CSV文件列数不足: {strname}，需要至少2列（深度列+数据列）')
                exit(0)

            # 提取第一列作为深度数据
            depth_data = df.iloc[:, 0].values.reshape(-1, 1)
            # 提取其余列作为电成像数据
            img_data = df.iloc[:, 1:].values
            # 数据验证: 检查深度数据是否单调递增
            depth_diff = np.diff(depth_data.flatten())

            if np.any(depth_diff <= 0):
                print(f'警告: CSV文件深度数据非单调递增，可能存在异常: {strname}')
            # 数据验证: 检查电成像数据范围
            if np.any(np.isnan(img_data)):
                print(f'警告: CSV文件包含NaN值: {strname}')
            if np.any(np.isinf(img_data)):
                print(f'警告: CSV文件包含无穷大值: {strname}')
            print(f'成功加载CSV文件: {strname}，数据形状: {img_data.shape}')
        except Exception as e:
            print(f'读取CSV文件失败: {strname}，错误: {e}')
            exit(0)

    # ============================================================================
    # 4. 深度范围截取功能
    # ============================================================================
    # 检查是否需要按深度范围截取数据
    if depth[0] < 0 and depth[1] < 0:
        # 深度参数为负值，不进行截取，返回全部数据
        pass
    else:
        # 计算深度步长 (用于深度匹配)
        Step = (depth_data[-1, 0] - depth_data[0, 0]) / depth_data.shape[0]

        # 初始化截取索引
        start_index = 0
        end_index = 0

        # 处理起始深度参数
        if depth[0] <= 0:
            start_index = 0  # 起始深度参数无效，从0开始
        # 处理结束深度参数
        if depth[1] <= 0:
            end_index = img_data.shape[0] - 1  # 结束深度参数无效，取到最后

        # 查找最接近指定深度的数据点索引
        for i in range(depth_data.shape[0]):
            # 使用半步长容差匹配深度值
            if abs(depth_data[i] - depth[0]) <= Step / 2 + 0.0001:
                start_index = i
            if abs(depth_data[i] - depth[1]) <= Step / 2 + 0.0001:
                end_index = i

        # 返回截取后的数据子集
        return img_data[start_index:end_index, :], depth_data[start_index:end_index, :]

    # 返回完整数据 (未截取)
    return img_data, depth_data



def get_random_ele_data():
    # 500*250 长图像
    path_test1 = r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\guan17-11_204_3682.0025_3683.2525_dyna.png'
    path_test2 = r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\guan17-11_204_3682.0025_3683.2525_stat.png'

    # 正常图像 250*250
    # path_test1 = r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\guan17-11_95_3627.5025_3628.1275_dyna.png'
    # path_test2 = r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\guan17-11_95_3627.5025_3628.1275_stat.png'
    # path_test1 = r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\guan17-11_141_3650.5025_3651.1275_dyna.png'
    # path_test2 = r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\guan17-11_141_3650.5025_3651.1275_dyna.png'
    # path_test1 = r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\guan17-11_189_3674.5025_3675.1275_dyna.png'
    # path_test2 = r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\guan17-11_189_3674.5025_3675.1275_stat.png'
    # path_test1 = r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\guan17-11_195_3677.5025_3678.1275_dyna.png'
    # path_test2 = r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\guan17-11_195_3677.5025_3678.1275_stat.png'
    # path_test1 = r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\LG7-12_423_5191.7520_5192.3770_dyna.png'
    # path_test2 = r'F:\DeepLData\target_stage1_small_big_mix\FMI_IMAGE\ZG_FMI_SPLIT\LG7-12_423_5191.7520_5192.3770_stat.png'

    # # 桃镇1H 14960*256的图像
    # path_test1 = r'F:\桌面\算法测试-长庆数据收集\logging_CSV\桃镇1H\桃镇1H_DYNA_FULL_TEST.txt'
    # path_test2 = r'F:\桌面\算法测试-长庆数据收集\logging_CSV\桃镇1H\桃镇1H_STAT_FULL_TEST.txt'

    # # 桃镇1H 全部图像
    # path_test1 = r'F:\logging_workspace\桃镇1H\桃镇1H_DYNA_FULL.txt'
    # path_test2 = r'F:\logging_workspace\桃镇1H\桃镇1H_STAT_FULL.txt'

    data_img_dyna, data_depth = get_ele_data_from_path(path_test1)
    data_img_stat, data_depth = get_ele_data_from_path(path_test2)
    return data_img_dyna, data_img_stat, data_depth

if __name__ == '__main__':
    img_dyna, img_stat, data_depth = get_random_ele_data()
    print(img_dyna.shape, img_stat.shape, data_depth.shape)