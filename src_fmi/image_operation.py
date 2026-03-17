import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import math


# import seaborn as sns
# sns.set()
# 图片的一些操作
# 1.show_Pic(pic_list, pic_order='12', pic_str=[], save_pic=False, path_save='')
# 展示图片，无返回
# 2.WindowsDataZoomer(SinglePicWindows, ExtremeRatio=0.02)
# 数据缩放，把电阻的数据域映射到图像的数据域，返回原图片数组大的图片数组[m,n] int
# 3.GetPicContours(PicContours, threshold = 4000)
# 对图片进行分割，threshold代表了目标区域需要保留的最小面积大小
# 返回的 contours_Conform, contours_Drop, contours_All 代表了目标轮廓信息list，被丢掉的轮廓信息list，总的轮廓信息list
# 轮廓信息包括，轮廓面积数值，轮廓描述（即是轮廓的存放），轮廓的质心[x, y]




# 定义一个随机增加 随机的膨胀、腐蚀、开闭操作
def pic_open_close_random(pic):
    # # 噪声去除
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))            # 矩形
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))           # 交叉形
    x_k_size = np.random.randint(1, 2)
    y_k_size = np.random.randint(1, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*x_k_size+1, 2*y_k_size+1))  # 椭圆形
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 椭圆形

    # # cv2.MORPH_CLOSE 闭运算(close)，先膨胀后腐蚀的过程。闭运算可以用来排除小黑洞。
    # # cv2.MORPH_OPEN  开运算(open) ,先腐蚀后膨胀的过程。开运算可以用来消除小黑点，在纤细点处分离物体、平滑较大物体的边界的 同时并不明显改变其面积。
    # # iterations – 操作次数，默认为1
    pic = cv2.morphologyEx(pic, cv2.MORPH_CLOSE, kernel, iterations=1)
    pic = cv2.morphologyEx(pic, cv2.MORPH_OPEN, kernel, iterations=1)

    # for i in range(pic.shape[0]):
    #     for j in range(pic.shape[1]):
    #         if pic[i][j] > 1:
    #             pic[i][j] = 255

    return pic


def get_pic_distribute(pic=np.random.randint(1,256,(8, 8)), dist_length=9, min_V=0, max_V=256):
    # pic_mean = np.mean(pic)
    # pic_s2 = np.var(pic)

    if len(pic.shape)==2:
        step = (max_V-min_V)/dist_length
        pic_dist = np.zeros(dist_length)
        for i in range(pic.shape[0]):
            for j in range(pic.shape[1]):
                index_t = math.floor((pic[i][j]-min_V)/step)
                pic_dist[index_t] += 1

        pic_dist = pic_dist/pic.size
        # return pic_dist, np.array([pic_mean, pic_s2])
        return pic_dist
    else:
        print('wrong pic shape:{}'.format(pic.shape))
        exit(0)


def show_Pic(pic_list, pic_order=None, pic_str=[], path_save='', title='title', figure=(16, 9), show=True):
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
    # 设置后端为Agg避免PyCharm后端问题
    # import matplotlib
    # print(matplotlib.matplotlib_fname())
    # matplotlib.use('Qt5Agg')  # 使用非交互式后端
    from matplotlib import pyplot as plt
    plt.rcParams['font.family'] = 'SimHei'

    if pic_order is None:
        num_pic = len(pic_list)
        # 使用字典简化配置
        size_config = {
            4: ('22', (9, 9)),
            6: ('23', (12, 9)),
            8: ('24', (14, 9)),
            9: ('33', (9, 9)),
            10: ('25', (18, 9)),
            12: ('34', (12, 9)),
            14: ('27', (18, 6)),
            15: ('35', (15, 9))
        }

        if num_pic in size_config:
            pic_order, figure = size_config[num_pic]
        else:
            pic_order = f'1{num_pic}'
            figure = (num_pic, 1)

    if len(pic_order) != 2:
        print(f'pic order error: {pic_order}')
        return

    # 计算图像总数
    rows, cols = map(int, pic_order)
    num = rows * cols

    if num != len(pic_list):
        print(f'pic order num is not equal to pic_list num: {len(pic_list)} vs {pic_order}')
        return

    # 自动生成标题
    pic_str += [f'Image {i + 1}' for i in range(len(pic_list) - len(pic_str))]

    # 预处理图像
    processed_pics = []
    for pic in pic_list:
        # 归一化处理
        if np.max(pic) < 4.01:
            pic = 255 * pic
        # 确保数据类型正确
        pic = np.clip(pic, 0, 255).astype(np.uint8)

        # 通道顺序调整
        if len(pic.shape) == 3 and pic.shape[0] == 3:
            pic = np.transpose(pic, (1, 2, 0))

        processed_pics.append(pic)

    plt.close('all')
    fig, axes = plt.subplots(rows, cols, figsize=figure)
    fig.suptitle(title, fontsize=18)

    # 展平轴数组以便迭代
    axes = axes.flatten() if rows > 1 or cols > 1 else [axes]

    for i, (ax, pic, title_str) in enumerate(zip(axes, processed_pics, pic_str)):
        ax.set_title(title_str)
        ax.axis('off')

        if len(pic.shape) == 3 and pic.shape[-1] == 3:
            ax.imshow(pic)
        else:
            ax.imshow(pic, cmap='hot')  # 使用热力图显示单通道图像

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 为标题留出空间

    if path_save:
        plt.savefig(path_save, bbox_inches='tight')

    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"显示图像时出错: {e}")
            print("尝试保存图像到临时文件...")
            temp_path = "temp_plot.png"
            plt.savefig(temp_path)
            print(f"图像已保存到: {temp_path}")

    plt.close()

# 数据缩放，把电阻的数据域映射到图像的数据域
def WindowsDataZoomer(SinglePicWindows, ExtremeRatio=0.02):
    """
    数据缩放，把电阻的数据域映射到图像的数据域
    通过计算5%的极大值、极小值来完成，会修改原本的数组，数组依旧是小数
    修改原数据
    :param SinglePicWindows:2d np.array
    :return:no change original data
    """
    ExtremePointNum = max(int(SinglePicWindows.size*ExtremeRatio), 1)
    sorted_pic = np.sort(SinglePicWindows.reshape(1, -1)[0])
    bigTop = np.mean(sorted_pic[-ExtremePointNum:])
    smallTop = np.mean(sorted_pic[:ExtremePointNum])
    if bigTop - smallTop < 0.000001:
        print("Error........bigTop == smallTop")
        exit(0)
    Step = 256 / (bigTop - smallTop)
    SinglePicWindows_new = (SinglePicWindows - smallTop) * Step
    SinglePicWindows_new = np.clip(SinglePicWindows_new, 0, 255)
    return SinglePicWindows_new, Step, smallTop


def GetPicContours(PicContours, threshold = 4000):
    # findContours函数第二个参数表示轮廓的检索模式
    # cv2.RETR_EXTERNAL 表示只检测外轮廓
    # cv2.RETR_LIST     检测的轮廓不建立等级关系
    # cv2.RETR_CCOMP    建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
    # cv2.RETR_TREE     建立一个等级树结构的轮廓。
    # 第三个参数method为轮廓的近似办法
    # cv2.CHAIN_APPROX_NONE     存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1 - x2），abs（y2 - y1）） == 1
    # cv2.CHAIN_APPROX_SIMPLE   压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    # cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh - Chinl chain近似算法
    contours, hierarchy = cv2.findContours(PicContours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )

    contours_Conform = [[], [], []]       # 存储符合要求的轮廓 顺序为 # 面积，轮廓，质心
    contours_Drop = [[], [], []]          # 存储不符合要求的轮廓
    contours_All = [[], [], []]           # 存储所有轮廓
    for i in range(len(contours)):
        # contour_S 为轮廓面积
        contour_S = cv2.contourArea(contours[i])
        M = cv2.moments(contours[i])
        # mc为质心
        mc = [int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])]

        if contour_S > threshold:         # 筛选出面积大于4000的轮廓
            # print('第%d个轮廓面积：' % i + str(temp))
            contours_Conform[0].append(contour_S)
            contours_Conform[1].append(contours[i])
            contours_Conform[2].append(mc)
        else:                           # 剩下的为不合格的轮廓
            # print('第%d个轮廓面积：'%i + str(temp))
            contours_Drop[0].append(contour_S)
            contours_Drop[1].append(contours[i])
            contours_Drop[2].append(mc)

        # 记录全部的轮廓信息
        contours_All[0].append(contour_S)
        contours_All[1].append(contours[i])
        contours_All[2].append(mc)
    return contours_Conform, contours_Drop, contours_All




# 图像对比度计算
def contrast(img1):
    m, n = img1.shape
    # 图片矩阵向外扩展一个像素
    img1_ext = cv2.copyMakeBorder(img1,1,1,1,1,cv2.BORDER_REPLICATE) / 1.0   # 除以1.0的目的是uint8转为float型，便于后续计算
    rows_ext,cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1,rows_ext-1):
        for j in range(1,cols_ext-1):
            b += ((img1_ext[i,j]-img1_ext[i,j+1])**2 + (img1_ext[i,j]-img1_ext[i,j-1])**2 +
                    (img1_ext[i,j]-img1_ext[i+1,j])**2 + (img1_ext[i,j]-img1_ext[i-1,j])**2)

    cg = b/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4)
    # cg = b/(m*n)
    # print(cg)
    return cg

#计算峰值信噪比
def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# 计算图像信息熵
def comentropy(img):
    # img = cv2.imread('20201210_3.bmp',0)
    # img = np.zeros([16,16]).astype(np.uint8)
    img = np.array(img).astype(np.uint8)
    m, n = img.shape

    hist_cv = cv2.calcHist([img], [0], None, [256], [0, 256])  # [0,256]的范围是0~255.返回值是每个灰度值出现的次数

    P = hist_cv / (m * n)  # 概率
    E = np.sum([p * np.log2(1 / p) for p in P if p>0])

    return E





# 常规方法的图像修复
def pic_repair_normal(pic, windows_l=5):
    PicDataWhiteStripe = np.zeros_like(pic)

    # 手动空白带提取
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if pic[i][j] <= 0.09:
                PicDataWhiteStripe[i][j] = 255
            else:
                PicDataWhiteStripe[i][j] = 0
    # 空白带提取
    # ret, PicDataWhiteStripe = cv2.threshold(pic, 0, 1, cv2.THRESH_BINARY_INV)

    PicDataWhiteStripe = np.uint8(PicDataWhiteStripe)
    pic = np.uint8(pic)

    # TELEA 图像修复
    PIC_Repair_dst_TELEA = cv2.inpaint(pic, PicDataWhiteStripe, windows_l, cv2.INPAINT_TELEA)
    # NS 图像修复
    PIC_Repair_dst_NS = cv2.inpaint(pic, PicDataWhiteStripe, windows_l, cv2.INPAINT_NS)

    return PIC_Repair_dst_TELEA, PIC_Repair_dst_NS, PicDataWhiteStripe
# pic_new = pic_scale_simple(pic_shape=[0.5, 0.5])
# print(pic_new)



def cal_pic_generate_effect(pic_org, pic_repair):
    # print(pic_org.shape, pic_repair.shape)
    # 计算PSNR：
    PSNR = peak_signal_noise_ratio(pic_org, pic_repair)
    # 计算SSIM
    SSIM = structural_similarity(pic_org, pic_repair)
    # 计算MSE 、 RMSE、 MAE、r2
    mse = np.sum((pic_org - pic_repair) ** 2) / pic_org.size
    rmse = math.sqrt(mse)
    mae = np.sum(np.absolute(pic_org - pic_repair)) / pic_org.size
    r2 = 1 - mse / np.var(pic_org)  # 均方误差/方差

    Entropy_org = comentropy(pic_org)
    Entropy_vice = comentropy(pic_repair)

    Con_org = contrast(pic_org)
    Con_vice = contrast(pic_repair)

    return PSNR, SSIM, mse, rmse, mae, r2, Entropy_org, Entropy_vice, Con_org, Con_vice


# if __name__ == '__main__':
#     folder_path = r'D:\GitHubProj\Logging_Interpretation\test\texture_set'
#     path_list = traverseFolder(folder_path)
#     distance = [1, 2]
#     angles = [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4]
#     level = 16
#     pic_all_list = []
#     feature_all_list = []
#     str_list_all = []
#
#     print(path_list)
#     for path in path_list:
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (128, 128))
#
#         feature, glcm_matric, _, _ = get_glcm_sixFeature(img, level=level, distance=distance, angles=angles)
#         pic_all_list.append(img)
#         pic_all_list.append(glcm_matric[:, :, 0, 0])
#         feature_all_list.append(feature[:, 0, 0])
#
#         pic_list_glcm_matric = [img]
#         pic_str = ['原始图像']
#         for i in range(glcm_matric.shape[2]):
#             for j in range(glcm_matric.shape[3]):
#                 pic_list_glcm_matric.append(glcm_matric[:, :, i, j])
#                 pic_str.append('distance_{}_angle_{:.2f}Π'.format(distance[i], angles[j] / np.pi))
#         # pic_order = '{}{}'.format(len(distance), len(angles))
#         pic_order = '33'
#
#         show_Pic(pic_list_glcm_matric, pic_order=pic_order, path_save=False, title=path.split('/')[-1].split('.')[0],
#                  pic_str=pic_str, figure=(10, 9))
#         # print(feature.shape, glcm_matric.shape)
#         print(path.split('/')[-1].split('.')[0], feature.shape, feature.ravel()[:6])
#         str_list_all.append(path.split('/')[-1].split('.')[0])
#         str_list_all.append('GLCM_Map')
#
#     show_Pic(pic_all_list, pic_order='46', pic_str=str_list_all)
#     # 设置打印位数为2 设置是否使用科学计数法
#     np.set_printoptions(precision=2, suppress=True)
#     print(np.array(feature_all_list))





def image_similarity(pic1: np.ndarray, pic2: np.ndarray) -> dict:
    from skimage.metrics import structural_similarity as ssim
    from sklearn.metrics import mean_squared_error
    from scipy.stats import entropy
    """
    多维度图像相似度对比接口
    参数：
        pic1, pic2: 单通道灰度图像数组 (H, W) 值域[0,255]
    返回：
        包含7种相似度指标的字典
    """
    # 输入验证
    if pic1.shape != pic2.shape:
        pic2 = cv2.resize(pic2, (pic1.shape[1], pic1.shape[0]))
    # assert pic1.shape == (100, 100) and pic2.shape == (100, 100), "输入必须为100x100图像"
    assert pic1.dtype == np.uint8 and pic2.dtype == np.uint8, "图像必须为uint8类型"

    # 初始化结果字典
    result = {
        "MSE": 0.0,
        "PSNR": 0.0,
        "SSIM": 0.0,
        "Histogram_Bhattacharyya": 0.0,
        "Histogram_ChiSquare": 0.0,
        "Histogram_KLD": 0.0,
        "FeatureMatching_MatchCount": 0.0,
        "FeatureMatching_MatchScore": 0.0,
    }

    # 1. 均方误差(MSE)和峰值信噪比(PSNR)
    result["MSE"] = mean_squared_error(pic1, pic2)
    if result["MSE"] == 0:  # 完全相同图像
        result["PSNR"] = 100.0
    else:
        result["PSNR"] = 20 * np.log10(255.0 / np.sqrt(result["MSE"]))

    # 2. 结构相似性指数(SSIM)
    result["SSIM"] = ssim(pic1, pic2, data_range=255)

    # 3. 直方图相似度
    hist1 = cv2.calcHist([pic1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([pic2], [0], None, [256], [0, 256])

    # 归一化直方图
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # 巴氏距离 (值越小越相似)
    result['Histogram_Bhattacharyya'] = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    # 卡方距离 (值越小越相似)
    result["Histogram_ChiSquare"] = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

    # KL散度 (非对称性度量)
    eps = 1e-10  # 避免零除
    result["Histogram_KLD"] = entropy(hist1 + eps, hist2 + eps)

    # 4. 局部特征匹配
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(pic1, None)
    kp2, des2 = orb.detectAndCompute(pic2, None)

    if des1 is not None and des2 is not None:
        # 暴力匹配器
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # 计算匹配分数
        result["FeatureMatching_MatchCount"] = len(matches)
        if len(matches) > 0:
            distances = [m.distance for m in matches]
            result["FeatureMatching_MatchCount"] = np.mean(distances)
    else:
        result["FeatureMatching_MatchScore"] = 0.0

    return result


# def test_image_similarity_cal():
#     # 生成测试图像
#     img1 = np.random.randint(0, 255, (100,100), dtype=np.uint8)
#     img2 = np.random.randint(0, 255, (100,100), dtype=np.uint8)
#
#     # 调用接口
#     result = image_similarity(img1, img2)
#
#     # 输出结果示例
#     print(f"""
#     MSE: {result['MSE']:.2f}
#     PSNR: {result['PSNR']:.2f} dB
#     SSIM: {result['SSIM']:.4f}
#     直方图相似度:
#       - 巴氏距离: {result['Histogram']['Bhattacharyya']:.4f}
#       - 卡方距离: {result['Histogram']['ChiSquare']:.2f}
#       - KL散度: {result['Histogram']['KLD']:.4f}
#     特征匹配:
#       - 匹配点数量: {result['FeatureMatching']['MatchCount']}
#       - 平均距离: {result['FeatureMatching']['MatchScore']:.2f}
#     """)
