import copy
import math
import random
import numpy as np
import cv2
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.feature import graycomatrix, graycoprops
import math
from src_file_op.dir_operation import get_all_file_paths
from src_fmi.fmi_data_read import get_random_ele_data, get_ele_data_from_path



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
# 4.GetBinaryPic(ProcessingPic)
# 有点问题，别用这个函数
# 5.pic_enhence(input, windows_shape = 7, ratio_top = 0.33, ratio_migration = 5/6)
# 图片的增强函数，用的是局部梯度偏移
# 6.pic_enhence_random(input, windows_shape=3, ratio_top=0.2, ratio_migration=0.6, random_times=3)
# 图片的增强函数，用的是随机局部梯度偏移
# 7.pic_scale(input, windows_shape=3, center_ratio=0.5, x_size=100.0, y_size=100.0, ratio_top=0.1)
# 图片缩放函数，用的是局部梯度增强缩放
# 8.test_pic_enhance_effect()
# 测试图片的增强效果
# 9.adjust_gamma(image, gamma=1.0)
# 图片的 伽马增强
# 10.save_img_data(dep, data, path='')
# 保存图片
# 11.pic_smooth_effect_compare()
# 图片的增强效果对比函数，对比上面的随机局部梯度偏移、伽马增强、直方图均衡增强效果


# 最简单的图像二值化，将图像进行简单的二值化，截止值为图像最值的一半
def binary_pic(pic):
    # max_v = np.max(pic)
    # for i in range(pic.shape[0]):
    #     for j in range(pic.shape[1]):
    #         if pic[i][j] > (max_v / 2):
    #             pic[i][j] = max_v
    #         else:
    #             pic[i][j] = 0
    retVal, img = cv2.threshold(pic, 0, 255, cv2.THRESH_OTSU)

    return img


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

def WindowsDataZoomer_PicList(pic_list, ExtremeRatio=0.02, USE_EXTRE=False, Max_V=-1, Min_V=-1):
    pic_list_numpy = np.array(pic_list)
    ExtremePointNum = int(pic_list_numpy.size * ExtremeRatio)
    bigTop = np.max(pic_list_numpy)
    smallTop = np.min(pic_list_numpy)
    pic_list_N = copy.deepcopy(pic_list)

    if USE_EXTRE:
        bigTop = np.mean(np.sort(pic_list_numpy.reshape(1, -1)[0])[-ExtremePointNum:])
        smallTop = np.mean(np.sort(pic_list_numpy.reshape(1, -1)[0])[:ExtremePointNum])

    if Max_V > 0:
        bigTop = Max_V
        smallTop = Min_V

    if bigTop - smallTop < 0.001:
        print("Error........bigTop == smallTop")
        exit(0)
    Step = 256 / (bigTop - smallTop)

    for n in range(len(pic_list)):
        for j in range(pic_list[n].shape[0]):
            for k in range(pic_list[n].shape[1]):
                pic_list_N[n][j][k] = (pic_list[n][j][k] - smallTop) * Step
                if pic_list_N[n][j][k] < 0:
                    pic_list_N[n][j][k] = 0
                elif pic_list_N[n][j][k] > 255:
                    pic_list_N[n][j][k] = 255

    return pic_list_N, Step, smallTop


# 数据缩放，把电阻的数据域映射到图像的数据域
def WindowsDataZoomer(SinglePicWindows, ExtremeRatio=0.02):
    """
    数据缩放，把电阻的数据域映射到图像的数据域
    通过计算5%的极大值、极小值来完成，会修改原本的数组，数组依旧是小数
    修改原数据
    :param SinglePicWindows:2d np.array
    :return:no change original data
    """
    # print('Windows Data Zoomer......')
    # tem = np.argsort(SinglePicWindows.reshape(1, -1)[0], axis=-1, kind='quicksort', order=None)
    # print(np.sort(SinglePicWindows.reshape(1, -1)[0]))
    ExtremePointNum = int(SinglePicWindows.size*ExtremeRatio)
    # print('缩放的最大最小值的窗口大小：%s'%ExtremePointNum)
    # bigTop = np.mean(np.sort(SinglePicWindows.reshape(1, -1)[0])[-ExtremePointNum:])
    bigTop = np.max(SinglePicWindows)
    # print('大的一段%.5f'%bigTop)
    # smallTop = np.mean(np.sort(SinglePicWindows.reshape(1, -1)[0])[:ExtremePointNum])
    smallTop = np.min(SinglePicWindows)
    # print('小的一段%.5f'%smallTop)
    if bigTop - smallTop < 0.000001:
        print("Error........bigTop == smallTop")
        exit(0)
    Step = 256 / (bigTop - smallTop)
    # print('缩放的倍数：%.5f'%Step)
    # print(SinglePicWindows[:5, :5])
    # print('缩放前子图平均数：%.5f'%(np.mean(SinglePicWindows)))
    SinglePicWindows_new = np.copy(SinglePicWindows)
    for j in range(SinglePicWindows.shape[0]):
        for k in range(SinglePicWindows.shape[1]):
            SinglePicWindows_new[j][k] = (SinglePicWindows[j][k] - smallTop) * Step
            if SinglePicWindows_new[j][k] < 0:
                SinglePicWindows_new[j][k] = 0
            elif SinglePicWindows_new[j][k] > 255:
                SinglePicWindows_new[j][k] = 255
    # print(SinglePicWindows[:5, :5])
    # print('缩放后子图平均数：%.5f'%(np.mean(SinglePicWindows)))
    # SinglePicWindows = np.array(SinglePicWindows, dtype=np.int)
    # SinglePicWindows = np.array(SinglePicWindows, dtype=np.float)
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


def GetBinaryPic(ProcessingPic):
    Blur_Average = cv2.blur(ProcessingPic, (7, 5))
    Blur_Gauss = cv2.GaussianBlur(ProcessingPic, (7, 5), 0)
    Blur_Median = cv2.medianBlur(ProcessingPic, 5)

    ProcessingPic = Blur_Gauss
    firstLevel = 40
    ret, img_binary_Level1 = cv2.threshold(ProcessingPic, firstLevel, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level2 = cv2.threshold(ProcessingPic, firstLevel + 10, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level3 = cv2.threshold(ProcessingPic, firstLevel + 20, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level4 = cv2.threshold(ProcessingPic, firstLevel + 30, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level5 = cv2.threshold(ProcessingPic, firstLevel + 40, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level6 = cv2.threshold(ProcessingPic, firstLevel + 50, 255, cv2.THRESH_BINARY)

    ProcessingPic = img_binary_Level3
    Kernel_Rect = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))  # 生成形状为矩形5x5的卷积核
    Kernel_Ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 3))  # 椭圆形9x9
    kernel = np.ones((5, 5), np.uint8)
    targetKernel = Kernel_Ellipse
    Pic_erosion = cv2.erode(ProcessingPic, targetKernel, iterations=1)
    Pic_dilation = cv2.dilate(ProcessingPic, targetKernel, iterations=1)
    Pic_opening = cv2.morphologyEx(ProcessingPic, cv2.MORPH_OPEN, targetKernel)
    Pic_closing = cv2.morphologyEx(ProcessingPic, cv2.MORPH_CLOSE, targetKernel)
    Pic_opening_closing = cv2.morphologyEx(Pic_opening, cv2.MORPH_CLOSE, targetKernel)
    Pic_closing_opening = cv2.morphologyEx(Pic_closing, cv2.MORPH_CLOSE, targetKernel)
    ProcessingPic = Pic_opening_closing
    t, Pic_To_Count_Contours = cv2.threshold(ProcessingPic, 0, 255, cv2.THRESH_BINARY_INV)  # 通过阀值将其反色为白图 二值化图像反转

    return Pic_To_Count_Contours


def process_pix(index_x, index_y, input, windows_shape, max_pixel, ratio_top, ratio_migration):
    # 寻找窗口的index
    start_index_x = max(index_x-windows_shape//2, 0)
    end_index_x = min(index_x+windows_shape//2 + 1, input.shape[0])
    start_index_y = max(index_y-windows_shape//2, 0)
    end_index_y = min(index_y+windows_shape//2 + 1, input.shape[1])

    # 根据窗口index 获得窗口的 数据
    data_windows = copy.deepcopy(input[start_index_x:end_index_x, start_index_y:end_index_y]).ravel()

    value = input[index_x][index_y]

    # 根据窗口周边数据情况，计算像素移动方向， 正的为 增大，负的为 减小
    direction = -1
    if (np.sum(data_windows)-value) > (max_pixel/2) * (windows_shape*windows_shape-1):
        direction = 1
    # direction = ((np.sum(data_windows)-value)//(windows_shape*windows_shape-1))-(max_pixel//2)

    # ordered_list = sorted(data_windows)
    # small_top = np.mean(ordered_list[:int(len(ordered_list)*ratio_top)])
    # big_top = np.mean(ordered_list[-int(len(ordered_list)*ratio_top):])
    # print(small_top, big_top)
    small_top = np.min(data_windows)
    big_top = np.max(data_windows)

    if direction < 0:
        return (value - (value - small_top)*ratio_migration)
    else:
        return (value + (big_top - value)*ratio_migration)



def pic_enhence(input, windows_shape = 7, ratio_top = 0.33, ratio_migration = 5/6):
    max_pixel = np.max(input)
    data_new = copy.deepcopy(input)
    if (windows_shape%2) != 1:
        print('windows shape error...........')
        exit()

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            data_new[i][j] = process_pix(i, j, input, windows_shape, max_pixel, ratio_top, ratio_migration)

    return data_new

# 洗牌算法随机一个数组
def shuffle(lis):
    for i in range(len(lis) - 1, 0, -1):
        p = random.randrange(0, i + 1)
        lis[i], lis[p] = lis[p], lis[i]
    return lis



# 图像的随机偏移图像增强
def pic_enhence_random(input, windows_shape=3, ratio_top=0.2, ratio_migration=0.6, random_times=3):
    if ((windows_shape % 2) != 1) | (windows_shape < 0):
        print('windows shape error...........')
        exit()
    if len(input.shape) >= 3:
        print('转换成灰度图再运行')
        exit()

    max_pixel = np.max(input)
    data_new = copy.deepcopy(input)
    all_times = input.shape[0] * input.shape[1]

    a = list(range(all_times))
    r = shuffle(a)
    # print(r)

    # for i in range(all_times):
    #     x = random.randint(0, input.shape[0]-1)
    #     y = random.randint(0, input.shape[1]-1)
    #
    #     data_new[x][y] = process_pix(x, y, input, windows_shape, max_pixel, ratio_top, ratio_migration)

    for j in range(random_times):
        for i in r:
            # print(i)
            x = i // input.shape[1]
            y = i % input.shape[1]
            # print(i, x, y)

            data_new[x][y] = process_pix(x, y, input, windows_shape, max_pixel, ratio_top, ratio_migration)

    return data_new

# 图像缩放
def pic_scale(input, windows_shape=3, center_ratio=0.5, x_size=100.0, y_size=100.0, ratio_top=0.1):
    if x_size <= 1.0:
        x_size = int(x_size * input.shape[0])
    else:
        x_size = int(x_size)
    if y_size <= 1.0:
        y_size = int(y_size * input.shape[1])
    else:
        y_size = int(y_size)

    pic_new = np.zeros((x_size, y_size)).astype('uint8')

    if x_size>input.shape[0] | y_size>input.shape[1]:
        print('size error pic processing..............')
        exit()

    if windows_shape%2 != 1:
        print('windows shape error, must be single......')
        exit()


    for i in range(x_size):
        for j in range(y_size):
            index_x = int(i/x_size * (input.shape[0] - 1))
            index_y = int(j/y_size * (input.shape[1] - 1))

            # 寻找窗口的index
            start_index_x = max(index_x - windows_shape // 2, 0)
            end_index_x = min(index_x + windows_shape // 2 + 1, input.shape[0])
            start_index_y = max(index_y - windows_shape // 2, 0)
            end_index_y = min(index_y + windows_shape // 2 + 1, input.shape[1])

            # 根据窗口index 获得窗口的 数据
            data_windows = copy.deepcopy(input[start_index_x:end_index_x, start_index_y:end_index_y]).ravel()

            value = input[index_x][index_y]

            windows_mean = np.mean(data_windows)

            ordered_list = sorted(data_windows)
            small_top = np.mean(ordered_list[:int(len(ordered_list) * ratio_top)])
            big_top = np.mean(ordered_list[-int(len(ordered_list) * ratio_top):])

            if windows_mean > int(center_ratio * 256):
                pic_new[i][j] = int(max(value, big_top))
            elif windows_mean < int(center_ratio * 256):
                pic_new[i][j] = int(min(value, small_top))

    return pic_new


# 图像缩放
def pic_scale_normal(input, shape=(196, 196)):
    if len(input.shape) == 2:
        if (shape[0] < input.shape[0]) | (shape[1] < input.shape[1]):
            print('pic scale fun error:shape {}&{}'.format(shape,input.shape))
            exit(0)

        pic_new = np.zeros(shape).astype('uint8')

        for i in range(shape[0]):
            for j in range(shape[1]):
                index_x = int(i/shape[0] * (input.shape[0] - 1))
                index_y = int(j/shape[1] * (input.shape[1] - 1))

                pic_new[i][j] = input[index_x, index_y]

        return pic_new
    elif len(input.shape) == 3:
        img_tar = []
        for i in range(input.shape[0]):
            img_tar.append(cv2.resize(input[i], shape))
        return np.array(img_tar)
    else:
        print('error shape:{}'.format(input.shape))
        exit(0)
# def get_pixel_normal(pic_t=np.random.random((9, 17))):
#     index_1 = -1
#     index_2 = -1
#     for j in range(pic_t.shape[1]):
#         if (pic_t[0][j] < 0) :
#             index_2 = j
#             if index_1 == -1:
#                 index_1 = j
#
#     # print(index_1, index_2)
#     a = pic_t[:, :index_1]
#     b = pic_t[:, index_2+1:]
#
#     # print(a, b)
#
#     if a.shape[1] > b.shape[1]:
#         return np.mean(a)
#     else:
#         return np.mean(b)


def test_pic_random_enhance_effect():
    data_img, data_depth = get_random_ele_data()
    print(data_img.shape, data_depth.shape)

    processing_pic = data_img[0:600, :]
    pic_EH = pic_enhence_random(processing_pic, windows_shape=3, ratio_top=0.1, ratio_migration=0.3, random_times=1)
    pic_equalizeHist = cv2.equalizeHist(processing_pic)  # 直方图均衡化

    show_Pic([processing_pic, pic_EH], save_pic=False, pic_order='12', pic_str=['pic_org', 'pic_enhance'])

    # hist_o = cv2.calcHist([np.uint8(processing_pic)], [0], None, [256], [0, 256])
    # hist_EH = cv2.calcHist([np.uint8(pic_EH)], [0], None, [256], [0, 256])
    # plt.subplot(2, 2, 1)
    # plt.plot(hist_o/processing_pic.size, label="原图灰度直方图", linestyle="--", color='g')
    # plt.legend()
    # plt.subplot(2, 2, 2)
    # plt.plot(hist_EH/pic_EH.size, label="增强后灰度直方图", linestyle="--", color='r')
    # plt.legend()
    # # plt.show()
    #
    # plt.subplot(2, 2, 3)
    # plt.imshow(processing_pic)
    # plt.subplot(2, 2, 4)
    # plt.imshow(pic_EH)
    # plt.savefig("pic_enhance_effect.jpg")
    # plt.show()


# a = cv2.imread('1_2.png')
# print(a.shape)
# show_Pic([a , a, a, a], pic_order='22', save_pic=True, path_save='121212.png')



def metrological_performance():
    img1 = cv2.imread('messi5.jpg')
    e1 = cv2.getTickCount()
    for i in range(5, 49, 2):
        img1 = cv2.medianBlur(img1, i)
    e2 = cv2.getTickCount()
    t = (e2 - e1) / cv2.getTickFrequency()
    # print(t)


def save_img_data(dep, data, path=''):
    dep = np.reshape(dep, (-1, 1))
    data = np.hstack((dep, data))

    np.savetxt(path, data, fmt='%.4f', delimiter='\t', comments='',
               header='WELLNAME={}\nSTDEP\t= {}\nENDEP\t= {}\nLEV\t= {:.4f}\nUNIT\t= meter\nCURNAMES= {}\n#DEPTH\t{}'.format(
                   'Temp_well', dep[0, 0], dep[-1, 0], dep[1, 0]-dep[0, 0], 'Img_data', 'Img_data'))

# 线性变换的原理是对所有像素值乘上一个扩张因子 factor
# 像素值大的变得越大，像素值小的变得越小，从而达到图像增强的效果，这里利用 Numpy 的数组进行操作；
def line_trans_img(img,coffient):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    out = 2*img
    #像素截断；；；
    out[out>255] = 255
    out = np.around(out)
    return out


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

    cg = b/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4) #对应上面48的计算公式
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



# from skimage.metrics import structural_similarity
# from skimage.metrics import peak_signal_noise_ratio

def pic_smooth_effect_compare():

    # import cv2 as cv
    # import numpy as np
    from matplotlib import pyplot as plt
    # %matplotlib inline
    # img = cv.imread('opencv-logo-white.png')
    os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
    plt.rcParams['font.family'] = 'SimHei'
    data_img_dyna, data_img_stat, data_depth = get_random_ele_data()
    print('data_image shape:{}'.format(data_img_dyna.shape))

    data_img = cv2.resize(data_img_dyna,(256, 256))

    # index_1, index_2 = 1400, 1800
    # img = np.uint8(data_img[index_1:index_2, :])
    # dep_img = data_depth[index_1:index_2, :]

    img = np.uint8(data_img)
    ret, img = cv2.threshold(img, 200+np.random.randint(0, 20)-10, 255, cv2.THRESH_BINARY_INV)
    print(img.shape)
    avg_blur = cv2.blur(img, (5, 5))
    guass_blur = cv2.GaussianBlur(img, (5, 5), 0)
    median_blur = cv2.medianBlur(img, 5)
    pic_bilateral_filter = cv2.bilateralFilter(img, 9, 75, 75)

    windows_shape = [3, 5, 7, 9]
    ratio_mig = [0.4, 0.6, 0.6, 0.6]
    random_times = [1, 1, 1, 1]

    pic_EH_3 = pic_enhence_random(img, windows_shape=windows_shape[0], ratio_migration=ratio_mig[0], random_times=random_times[0])
    # pic_EH_5 = pic_enhence_random(img, windows_shape=windows_shape[1], ratio_migration=ratio_mig[1], random_times=random_times[1])
    # pic_EH_7 = pic_enhence_random(img, windows_shape=windows_shape[2], ratio_migration=ratio_mig[2], random_times=random_times[2])
    # pic_EH_9 = pic_enhence_random(img, windows_shape=windows_shape[3], ratio_migration=ratio_mig[3], random_times=random_times[3])

    # 对比不同参数 随机偏移 的图像增强效果
    # show_Pic([img, pic_EH_3, pic_EH_5, pic_EH_7], pic_order='22',
    #          pic_str=['原始电成像图像', '像素值偏移增图像效果:n=5', '像素值偏移增图像效果:n=7', '像素值偏移增图像效果:n=9'])

    # 直方图均衡化
    pic_equalizeHist = cv2.equalizeHist(img)

    # 对图像进行局部直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10, 10))  # 对图像进行分割，10*10
    pic_local_equalizeHist = clahe.apply(img)  # 进行直方图均衡化

    # gama 伽马变换
    imgGrayNorm = img / 255
    gamma = 0.8
    pic_gamma_transf = (np.power(imgGrayNorm, gamma) * 256).astype(np.uint8)

    # print(contrast(img), contrast(pic_EH_3), contrast(pic_gamma_transf), contrast(pic_equalizeHist))
    # print(psnr(img, img), psnr(img, pic_EH_3), psnr(img, pic_gamma_transf), psnr(img, pic_equalizeHist))
    # print(comentropy(img), comentropy(pic_EH_3), comentropy(pic_gamma_transf), comentropy(pic_equalizeHist))

    # num_sp = 20
    # pixel_per_window = img.shape[0]//num_sp
    # E1 = []
    # E2 = []
    # E3 = []
    # E4 = []
    # for i in range(num_sp):
    #     for j in range(num_sp):
    #         pic_temp = img[i*pixel_per_window:(i+1)*pixel_per_window]
    #         E1.append(comentropy(pic_temp))
    #         pic_temp = pic_EH_3[i*pixel_per_window:(i+1)*pixel_per_window]
    #         E2.append(comentropy(pic_temp))
    #         pic_temp = pic_gamma_transf[i*pixel_per_window:(i+1)*pixel_per_window]
    #         E3.append(comentropy(pic_temp))
    #         pic_temp = pic_equalizeHist[i*pixel_per_window:(i+1)*pixel_per_window]
    #         E4.append(comentropy(pic_temp))
    #
    # print(np.mean(np.array(E1)), np.mean(np.array(E2)), np.mean(np.array(E3)), np.mean(np.array(E4)))

    # print(comentropy(img), comentropy(pic_EH_3), comentropy(pic_gamma_transf), comentropy(pic_equalizeHist))

    pic_equalizeHist = pic_open_close_random(pic_equalizeHist)
    pic_local_equalizeHist = pic_open_close_random(pic_local_equalizeHist)
    pic_bilateral_filter = pic_open_close_random(pic_bilateral_filter)
    pic_EH_3 = pic_open_close_random(pic_EH_3)
    pic_gamma_transf = pic_open_close_random(pic_gamma_transf)

    # cv2.imwrite('pic_equalizeHist.png', traverse_pic(pic_equalizeHist))
    # # cv2.imwrite('pic_local_equalizeHist.png', traverse_pic(pic_local_equalizeHist))
    # # cv2.imwrite('pic_bilateral_filter.png', traverse_pic(pic_bilateral_filter))
    # cv2.imwrite('pic_EH_3.png', traverse_pic(pic_EH_3))
    # # cv2.imwrite('pic_gamma_transf.png', traverse_pic(pic_gamma_transf))

    show_Pic([256-data_img, 256-pic_equalizeHist, 256-pic_local_equalizeHist,
              256-pic_gamma_transf, 256-pic_bilateral_filter, 256-pic_EH_3], pic_order='23',
             pic_str=['原始图像', '直方图均衡', '局部直方图均衡', '伽马变换', '双边滤波', '随机偏移增强'])


    # cv2.calcHist(images, channels, mask, histSize, ranges, hist, accumulate)
    # mask: 掩模图像。要统计整幅图像的直方图就把它设为 None。但是如 果你想统计图像某一部分的直方图的话，你就需要制作一个掩模图像，并 使用它。（后边有例子）
    # histSize：BIN 的数目。也应该用中括号括起来，例如：[256]。 5. ranges: 像素值范围，通常为 [0，256]
    # hist：是一个 256x1 的数组作为返回值，每一个值代表了与次灰度值对应的像素点数目。
    # accumulate：是一个布尔值，用来表示直方图是否叠加。
    # hist_org = cv2.calcHist([img], [0], None, [256], [0, 256])/img.size
    # hist_equalize = cv2.calcHist([pic_equalizeHist], [0], None, [256], [0, 256])/img.size
    # hist_local_equalize = cv2.calcHist([pic_local_equalizeHist], [0], None, [256], [0, 256])/img.size
    # hist_gama = cv2.calcHist([pic_gamma_transf], [0], None, [256], [0, 256])/img.size
    # hist_bil_blur = cv2.calcHist([pic_bilateral_filter], [0], None, [256], [0, 256])/img.size
    # hist_random_shift = cv2.calcHist([pic_EH_3], [0], None, [256], [0, 256])/img.size


    # # Draw Plot
    # # cut：参数表示绘制的时候，切除带宽往数轴极限数值的多少(默认为3)
    # # cumulative ：是否绘制累积分布，默认为False
    # # fill：若为True，则在kde曲线下面的区域中进行阴影处理，color控制曲线及阴影的颜色
    # # vertical：表示以X轴进行绘制还是以Y轴进行绘制
    # # label="原始成像"
    # # plt.figure(figsize=(10, 8), dpi=80)
    # # sns.kdeplot(img.ravel(), cut=0, fill=True, color="#01a2d9", alpha=.7).set(xlabel="Percentage", ylabel="Pixel distribution")
    # # sns.kdeplot(pic_equalizeHist.ravel(), fill=True, color="#dc2624", label="Cyl=5", alpha=.7)
    # # sns.kdeplot(pic_local_equalizeHist.ravel(), fill=True, color="#C89F91", label="Cyl=6", alpha=.7)
    # # sns.kdeplot(pic_gamma_transf.ravel(), fill=True, color="#649E7D", label="Cyl=8", alpha=.7)
    # # sns.kdeplot(pic_bilateral_filter.ravel(), fill=True, color="#649E7D", label="Cyl=8", alpha=.7)
    # # sns.kdeplot(pic_EH_3.ravel(), fill=True, color="#649E7D", label="Cyl=8", alpha=.7)
    #
    # sns.set_style(style="white")
    # sns.despine(top=True, right=True, left=False, bottom=False)
    # # sns.distplot(img.ravel(), bins=20).set(xlabel="Percentage", ylabel="Pixel distribution")
    # # sns.distplot(pic_EH_3.ravel(), bins=20).set(xlabel="Percentage", ylabel="Pixel distribution")
    # # sns.distplot(pic_gamma_transf.ravel(), bins=20).set(xlabel="Percentage", ylabel="Pixel distribution")
    # sns.distplot(pic_equalizeHist.ravel(), bins=20).set(xlabel="Percentage", ylabel="Pixel distribution")
    #
    # # sns.histplot(img.ravel(), color='#01a2d9').set(xlabel="Percentage", ylabel="Pixel distribution")
    # # sns.displot(img.ravel(), color='#01a2d9').set(xlabel="Percentage", ylabel="Pixel distribution")
    #
    # # plt.title('Density Plot of City Mileage by n_Cylinders', fontsize=18)
    # # plt.legend('legend')
    # plt.xlim([0, 256])
    # plt.ylim([0, 0.01])
    # plt.show()

    # # plt.subplot(2, 2, 1)
    # plt.subplot(1, 1, 1)
    # # plt.plot(hist_o, label="原始电成像图像灰度直方图", linestyle="--", color='g')
    # plt.plot(hist_org, linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.show()

    # # plt.subplot(2, 2, 2)
    # plt.subplot(1, 1, 1)
    # # plt.plot(hist_EH, label="随即迁移增强后的灰度直方图", linestyle="--", color='r')
    # plt.plot(hist_EH, linestyle="--", color='r')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.show()

    # # plt.subplot(2, 2, 3)
    # plt.subplot(1, 1, 1)
    # # plt.plot(hist_gamma, label="伽马变换增强后的灰度直方图", linestyle="--", color='g')
    # plt.plot(hist_gamma, linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.show()

    # # plt.subplot(2, 2, 4)
    # plt.subplot(1, 1, 1)
    # # plt.plot(hist_equalize_hist, label="直方图均衡增强后的灰度直方图", linestyle="--", color='g')
    # plt.plot(hist_equalize_hist, linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.savefig("pic_enhance_effect.jpg")
    # plt.show()



    # hist_o = cv2.calcHist([np.uint8(img)], [0], None, [256], [0, 256])/img.size
    # hist_EH = cv2.calcHist([np.uint8(pic_EH_3)], [0], None, [256], [0, 256])/img.size
    # hist_gamma = cv2.calcHist([np.uint8(pic_gamma_transf)], [0], None, [256], [0, 256])/img.size
    # # hist_gamma_1 = cv2.calcHist([np.uint8(pic_gamma_transf_1)], [0], None, [256], [0, 256])/img.size
    # hist_equalize_hist = cv2.calcHist([np.uint8(pic_equalizeHist)], [0], None, [256], [0, 256])/img.size

    # plt.subplot(2, 2, 1)
    # plt.plot(hist_o, label="原始电成像图像灰度直方图", linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.subplot(2, 2, 2)
    # plt.plot(hist_EH, label="随机迁移增强后的灰度直方图", linestyle="--", color='r')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.subplot(2, 2, 3)
    # plt.plot(hist_gamma, label="伽马变换增强后的灰度直方图", linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # plt.subplot(2, 2, 4)
    # plt.plot(hist_equalize_hist, label="直方图均衡增强后的灰度直方图", linestyle="--", color='g')
    # plt.xlabel("像素值")
    # plt.ylabel("频率分布")
    # plt.legend()
    # # plt.savefig("pic_enhance_effect.jpg")
    # plt.show()

# pic_smooth_effect_compare()




# 对二维数据 进行 简单的数据缩放
def pic_scale_simple(pic=np.array([]), pic_shape=[0,0]):
    if len(pic.shape) >= 3:
        print('only process two dim pic& pic shape is:{}'.format(pic.shape))
        exit(0)

    if pic_shape[0] <= 0:
        print('shape error...:{}'.format(pic_shape))
        exit(0)
    elif (pic_shape[0]>1) & (pic_shape[1]>1):
        x_size = pic_shape[0]
        y_size = pic_shape[1]
    elif (pic_shape[0] <= 1) & (pic_shape[0] > 0) & (pic_shape[1] <= 1) & (pic_shape[1] > 0):
        x_size = int(pic_shape[0] * pic.shape[0])
        y_size = int(pic_shape[1] * pic.shape[1])
    elif (pic_shape[0] > pic.shape[0]) | (pic_shape[1] > pic.shape[1]):
        print('target pic shape is {},org shape is {}'.format(pic_shape, pic.shape))
        exit(0)
    else:
        print('pic shape error:{}'.format(pic_shape))
        exit(0)

    pic_new = np.zeros((x_size, y_size))
    for i in range(x_size):
        for j in range(y_size):
            index_x = int(i/x_size*pic.shape[0])
            index_y = int(j/y_size*pic.shape[1])
            pic_new[i][j] = pic[index_x][index_y]

    return pic_new

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


def pic_seg_by_kai_bi():
    path_in = r'C:\Users\Administrator\Desktop\paper_f\unsupervised_segmentation\fracture\LN11-4_367_5444.3994_5445.0244_dyna.png'
    # path_in = r'D:\Data\target_stage3_small_p\train\2\LG701-H1_358_5380.5020_5381.1695_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG7-4_301_5259.5002_5260.1627_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG7-4_132_5171.0002_5171.6427_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG701_205_5224.5000_5225.1325_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG7-4_129_5169.5002_5170.1252_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG7-4_116_5162.0002_5162.6252_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG701_189_5215.5000_5216.1725_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG701_194_5218.0000_5218.6275_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG701-H1_104_5248.0020_5248.6370_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG7-4_107_5157.5002_5158.1277_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG701-H1_93_5242.5020_5243.1645_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG7-4_301_5259.5002_5260.1627_dyna.png'
    path_in = r'D:\Data\target_stage3_small_p\train\1\LG701_126_5183.0000_5183.6600_dyna.png'
    # path_in = r'D:\Data\target_stage3_small_p\train\2\LG701-H1_241_5320.0020_5320.6595_dyna.png'
    # path_in = r'D:\Data\target_stage3_small_p\train\2\LG701-H1_250_5324.5020_5325.1645_dyna.png'
    # path_in = r'D:\Data\target_stage3_small_p\train\2\LG701-H1_249_5324.0020_5324.6745_dyna.png'
    # path_in = r'D:\Data\target_stage3_small_p\train\2\LG701-H1_148_5271.5020_5272.1295_dyna.png'
    # path_in = r'D:\Data\target_stage3_small_p\train\2\LG7-4_336_5278.0002_5278.6477_dyna.png'

    pic = cv2.imread(path_in, cv2.IMREAD_GRAYSCALE)
    print(pic.shape)

    ProcessingPic = copy.deepcopy(pic)

    Blur_Average = cv2.blur(ProcessingPic, (7, 5))
    Blur_Gauss = cv2.GaussianBlur(ProcessingPic, (7, 5), 0)
    Blur_Median = cv2.medianBlur(ProcessingPic, 5)

    firstLevel = 40
    ret, img_binary_Level1 = cv2.threshold(ProcessingPic, firstLevel, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level2 = cv2.threshold(ProcessingPic, firstLevel + 10, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level3 = cv2.threshold(ProcessingPic, firstLevel + 20, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level4 = cv2.threshold(ProcessingPic, firstLevel + 30, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level5 = cv2.threshold(ProcessingPic, firstLevel + 40, 255, cv2.THRESH_BINARY)
    ret, img_binary_Level6 = cv2.threshold(ProcessingPic, firstLevel + 130, 255, cv2.THRESH_BINARY)
    ProcessingPic = img_binary_Level6

    # cv2.THRESH_BINARY：二值阈值处理，只有大于阈值的像素值为最大值，其他像素值为最小值。
    # cv2.THRESH_BINARY_INV：反二值阈值处理，只有小于阈值的像素值为最大值，其他像素值为最小值。
    # cv2.THRESH_TRUNC：截断阈值处理，大于阈值的像素值被赋值为阈值，小于阈值的像素值保持原值不变。
    # cv2.THRESH_TOZERO：置零阈值处理，只有大于阈值的像素值被置为0，其他像素值保持原值不变。
    # cv2.THRESH_TOZERO_INV：反置零阈值处理，只有小于阈值的像素值被置为0，其他像素值保持原值不变。

    Kernel_Rect = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # 生成形状为矩形5x5的卷积核
    Kernel_Rect2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    Kernel_Rect3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    Kernel_Ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 3))  # 椭圆形9x9
    kernel = np.ones((5, 5), np.uint8)
    targetKernel = Kernel_Rect3


    Pic_erosion = cv2.erode(ProcessingPic, targetKernel, iterations=1)
    Pic_dilation = cv2.dilate(ProcessingPic, targetKernel, iterations=1)
    Pic_opening = cv2.morphologyEx(ProcessingPic, cv2.MORPH_OPEN, targetKernel)
    Pic_closing = cv2.morphologyEx(ProcessingPic, cv2.MORPH_CLOSE, targetKernel)
    Pic_opening_closing = cv2.morphologyEx(Pic_opening, cv2.MORPH_CLOSE, targetKernel)
    Pic_closing_opening = cv2.morphologyEx(Pic_closing, cv2.MORPH_OPEN, targetKernel)

    ProcessingPic = Pic_opening_closing

    contours_Conform, contours_Drop, contours_All = GetPicContours(ProcessingPic, threshold=500)
    img_white = np.zeros((ProcessingPic.shape[0], ProcessingPic.shape[1]), np.uint8)
    img_white2 = np.zeros((ProcessingPic.shape[0], ProcessingPic.shape[1]), np.uint8)
    # img_white = np.zeros_like(ProcessingPic).astype(np.uint8).fill(0)
    # img_white = copy.deepcopy(ProcessingPic).astype(np.uint8).fill(0)
    print(ProcessingPic.shape[0], ProcessingPic.shape[1])
    # print(img_white)
    cv2.drawContours(img_white, contours_Conform[1], -1, 255, thickness=-1)
    # print(img_mask)

    # show_Pic([pic, ProcessingPic, img_white, img_white2], pic_order='14', pic_str=[], save_pic=False)

    cv2.imwrite(path_in.replace('dyna', 'mask2'), img_white)
    cv2.imwrite(path_in.replace('dyna', 'mask'), ProcessingPic)


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
# pic_seg_by_kai_bi()


import numpy as np



def cal_glcm_features(glcm_map, features):
    """
    手动实现GLCM特征计算函数

    参数:
    glcm_map: np.array类型灰度共生矩阵，四维张量，形状为(level, level, distances_len, angles_len)
    features: list, 包含需要计算的纹理特征名称

    返回:
    np.array 特征矩阵，形状为(len(features), distances_len, angles_len)
    """
    # 获取矩阵的形状参数
    level = glcm_map.shape[0]
    num_distances = glcm_map.shape[2]
    num_angles = glcm_map.shape[3]
    num_features = len(features)

    # 初始化结果矩阵
    feature_matrix = np.zeros((num_features, num_distances, num_angles))

    # 创建i和j的网格坐标（用于特征计算）
    i, j = np.meshgrid(np.arange(level), np.arange(level), indexing='ij')

    # 遍历所有距离和角度组合
    for d_idx in range(num_distances):
        for a_idx in range(num_angles):
            # 提取当前GLCM矩阵
            P = glcm_map[:, :, d_idx, a_idx]

            # 确保矩阵归一化（元素和为1）
            total = np.sum(P)
            if total > 0:
                P = P / total
            else:
                # 处理全零矩阵（避免除以零）
                feature_matrix[:, d_idx, a_idx] = np.nan
                continue

            # 初始化特征值字典
            feature_vals = {}

            # 计算基本统计量（在多个特征中复用）
            # 行和：p_i = Σ_j P(i,j)
            p_i = np.sum(P, axis=1)
            # 列和：p_j = Σ_i P(i,j)
            p_j = np.sum(P, axis=0)

            # 均值
            mu_i = np.sum(i * p_i)
            mu_j = np.sum(j * p_j)

            # 方差
            sigma_i = np.sqrt(np.sum((i - mu_i) ** 2 * p_i))
            sigma_j = np.sqrt(np.sum((j - mu_j) ** 2 * p_j))

            # 计算请求的特征
            for feature_name in features:
                if feature_name == 'contrast':
                    # 对比度: Σ_iΣ_j |i-j|² P(i,j)
                    feature_vals['contrast'] = np.sum((i - j) ** 2 * P)

                elif feature_name == 'dissimilarity':
                    # 相异性: Σ_iΣ_j |i-j| P(i,j)
                    feature_vals['dissimilarity'] = np.sum(np.abs(i - j) * P)

                elif feature_name == 'homogeneity':
                    # 同质性: Σ_iΣ_j P(i,j) / (1 + |i-j|)
                    feature_vals['homogeneity'] = np.sum(P / (1 + np.abs(i - j)))

                elif feature_name == 'energy' or feature_name == 'ASM':
                    # 能量/角二阶矩: Σ_iΣ_j P(i,j)²
                    asm_value = np.sum(P ** 2)
                    feature_vals['energy'] = np.sqrt(asm_value)
                    feature_vals['ASM'] = asm_value

                elif feature_name == 'correlation':
                    # 相关性: [Σ_iΣ_j (i-mu_i)(j-mu_j) P(i,j)] / (sigma_i * sigma_j)
                    if sigma_i > 0 and sigma_j > 0:
                        correlation = np.sum((i - mu_i) * (j - mu_j) * P) / (sigma_i * sigma_j)
                    else:
                        correlation = 0  # 处理常值图像
                    feature_vals['correlation'] = correlation

                elif feature_name == 'entropy':
                    # 熵: -Σ_iΣ_j P(i,j) log(P(i,j))
                    entropy = 0
                    # 仅处理非零概率元素
                    non_zero_mask = P > 0
                    non_zero_probs = P[non_zero_mask]
                    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
                    feature_vals['entropy'] = entropy

                else:
                    # 处理未知特征
                    feature_vals[feature_name] = np.nan

            # 将计算的特征值填充到结果矩阵中
            for feat_idx, feature_name in enumerate(features):
                feature_matrix[feat_idx, d_idx, a_idx] = feature_vals.get(feature_name, np.nan)

    return feature_matrix





# 计算GLCM矩阵的熵值，输入的是GLCM矩阵，16*16*M*N或者是32*32*M*N、64*64*M*N等
def glcm_entropy(glcm):
    """
    计算多维GLCM矩阵的熵值矩阵
    参数:
        glcm: 灰度共生矩阵，形状为 [L, L, M, N] (L=16,32,64等; M=距离数量; N=角度数量)
    返回:
        entropy_matrix: 二维矩阵 [M, N]，每个元素表示对应位置GLCM的熵值
    """
    if len(glcm.shape) == 2:
        glcm = glcm.reshape((glcm.shape[0], glcm.shape[1], 1, 1))
    elif len(glcm.shape) == 3:
        glcm = glcm.reshape((glcm.shape[0], glcm.shape[1], glcm.shape[2], 1))

    # 保存原始形状以便后续恢复
    original_shape = glcm.shape
    L = original_shape[0]  # 灰度级

    # 将多维GLCM重塑为二维矩阵 (L*L, M*N)
    glcm_2d = glcm.reshape(L * L, -1)
    num_glcms = glcm_2d.shape[1]  # GLCM切片数量

    # 计算每个GLCM切片的归一化因子
    totals = np.sum(glcm_2d, axis=0)
    # 避免除零错误（全零矩阵处理）
    totals[totals == 0] = 1.0

    # 归一化所有GLCM切片
    glcm_norm = glcm_2d / totals

    # 计算熵值（向量化操作）
    entropy_vals = np.zeros(num_glcms)
    for i in range(num_glcms):
        non_zero_probs = glcm_norm[:, i][glcm_norm[:, i] > 0]
        if non_zero_probs.size > 0:
            entropy_vals[i] = -np.sum(non_zero_probs * np.log2(non_zero_probs + 1e-10))

    # 恢复为原始维度结构 [M, N]
    return entropy_vals.reshape(original_shape[2:])

def get_glcm_Features(IMG_gray, level=16, distance=[1, 2], angles=[0, np.pi / 4, np.pi / 2, np.pi * 3 / 4],
                      feature_descrip=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']):  # 传入灰度图像
    """
    :param IMG_gray: 原始图像，一般为灰度图像
    :param level: 颜色等级，将图像灰度化为多少个颜色level的简单图像，方便进行灰度统计，这个决定了，GLCM矩阵的形状长度
    :param distance: 距离列表，构建glcm矩阵时，使用了那些距离参数
    :param angles: 角度列表，构建glcm矩阵时，使用的角度参数
    :param feature_descrip:都是哪些特征进行纹理信息的提取，目前支持：
    feature_descrip = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    :return: 返回四个矩阵，分别是，原始GLCM矩阵特诊参数，原始GLCM矩阵，平均GLCM矩阵特征参数，平均GLCM特征矩阵
    其中平均GLCM特征矩阵一般为七个，分别是，总的GLCM特征，两个不同distance特征矩阵（四个角度下的均值），四个不同angle特征矩阵（两个距离下的均值）
    """
    img = copy.deepcopy(IMG_gray).astype(np.int32)
    # 灰度压缩（确保处理一致）
    img = np.floor_divide(img, 256 / level).astype(np.uint8)
    img = np.clip(img, 0, level - 1)

    # 得到共生矩阵，参数：图像矩阵，距离，方向，灰度级别，是否对称，是否标准化
    glcm = graycomatrix(img,
                        distances=distance,
                        angles=angles,
                        levels=level,
                        symmetric=False,
                        normed=True)      # 100x100的灰度图像  ---> 16*16*2*4 不同 level * level * distance * angle  的灰度共生矩阵
    # print(glcm.shape)                 ####### 16*16*2*4 即：level * level * distance * angle

    # 总的glcm平均矩阵
    glcm_mean = [glcm.reshape((level, level, -1)).mean(axis=2)]         # 16*16*1
    # 图像  不同距离 下的glcm灰度共生矩阵glcm
    for i in range(glcm.shape[2]):                                      # 16*16*2
        glcm_mean.append(glcm[:, :, i, :].mean(axis=2))
    # 图像 不同方向 上的图形glcm灰度矩阵
    for i in range(glcm.shape[3]):                                      # 16*16*4
        glcm_mean.append(glcm[:, :, :, i].mean(axis=2))

    glcm_mean = np.array(glcm_mean)                                     # ---> 7*16*16
    glcm_mean = glcm_mean.transpose(1, 2, 0)                            # 转换形状7*16*16--->16*16*7
    glcm_mean = np.expand_dims(glcm_mean, axis=-1)                      # 转换形状16*16*7--->16*16*7*1

    # 得到共生矩阵的特征统计值，官方文档
    # http://tonysyu.github.io/scikit-image/api/skimage.feature.html#skimage.feature.greycoprops
    features = []
    # features = cal_glcm_features(glcm_mean, feature_descrip)

    ###### 这个返回的是： 总的平均矩阵特征参数、总的平均GLCM矩阵、六个平均矩阵（分别是2个距离上的，4个角度上的）
    for prop in feature_descrip:
        temp = graycoprops(glcm_mean, prop)
        # print(temp.ravel())
        features.append(temp)
    features.append(glcm_entropy(glcm_mean))

    # feature_mean ---> (7, 1, 1)
    # glcm_mean ---> (16, 16)
    # feature  --->  (7, 3(len(distance)+len(angles)), 1)
    # print(np.array(feature).shape, glcm_mean.shape)
    # glcm_mean  ---> (16, 16, 4(len(distance)+len(angles)+1), 1)

    return np.array(features)[:, 0, 0], glcm_mean[:, :, 0, 0], np.array(features), glcm_mean


def get_glcm_sub(IMG_gray, level=16, distance=[1,2]):  # 传入灰度图像
    """
    :param IMG_gray: 原始图像，一般为灰度图像
    :param level: 颜色等级，将图像灰度化为多少个颜色level的简单图像，方便进行灰度统计，这个决定了，GLCM矩阵的形状长度
    :param distance: 距离列表，构建glcm矩阵时，使用了那些距离参数
    :return: 返回灰度矩阵的X-Y纹理差结果，是一个float类型数值
    """
    texture_x, glcm_map_x, feature_all_x, glcm_map_x = get_glcm_Features(IMG_gray, level=level, distance=distance, angles=[0])
    texture_y, glcm_map_y, feature_all_y, glcm_map_y = get_glcm_Features(IMG_gray, level=level, distance=distance, angles=[np.pi / 2])

    # print(texture_x.shape, texture_y.shape, glcm_map_x.shape, glcm_map_y.shape)
    texture_sub = texture_x - texture_y

    return texture_sub

def get_glcm_xy(IMG_gray, level=16, distance=[1,2]):  # 传入灰度图像
    """
    :param IMG_gray: 原始图像，一般为灰度图像
    :param level: 颜色等级，将图像灰度化为多少个颜色level的简单图像，方便进行灰度统计，这个决定了，GLCM矩阵的形状长度
    :param distance: 距离列表，构建glcm矩阵时，使用了那些距离参数
    :return: 返回灰度矩阵的X-Y纹理差结果，是一个float类型数值
    """
    texture_x, glcm_map_x, feature_all_x, glcm_map_all_x = get_glcm_Features(IMG_gray,
                                                                         level=level, distance=distance, angles=[0])
    texture_y, glcm_map_y, feature_all_y, glcm_map_all_y = get_glcm_Features(IMG_gray,
                                                                         level=level, distance=distance, angles=[np.pi / 2])

    return texture_x, texture_y, glcm_map_x, glcm_map_y


def ele_stripes_delete(Pic, shape_target=(100, 8), delete_pix = 0):
    """
    空白条带删除函数，采用拉伸原则，把所有的像素放到一起拉伸成为新的像素行
    :param Pic: 原始图像
    :param shape_target:目标图像形状大小，这个y方向一定保持一致，x方向一定小于原始输入图像
    :param delete_pix: 要删除的像素点大小
    :return: 返回删除指定像素点后的新图像
    """
    pic_new = np.zeros(shape_target, np.uint8)

    if shape_target[0] != Pic.shape[0]:
        print('shape error, org shape is :{}, target shape is :{}'.format(Pic.shape, shape_target))

    for i in range(pic_new.shape[0]):
        # 查找符合条件的像素点坐标
        index_temp = np.where(Pic[i, :].ravel() != delete_pix)
        # 符合条件的像素点个数
        len_fit = np.array(index_temp).shape[-1]

        # 刚好相等，不用修改
        if len_fit == shape_target[-1]:
            row_pix = Pic[i, index_temp]
        # 多了，j进行相应的形变
        else:
            row_pix = Pic[i, index_temp]
            row_pix = np.resize(row_pix, (shape_target[-1], )).astype(np.uint8)

        pic_new[i, :] = row_pix

    return pic_new




# if __name__ == '__main__':
#     pic_temp = np.random.random((100, 100))*256
#     glcm, map, glcm_average, map_average = get_glcm_sixFeature(pic_temp)
#     print('glcm matrix shape:{},{}'.format(glcm.shape, glcm_average.shape))
#     print('glcm map shape:{},{}'.format(map.shape, map_average.shape))


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



def show_pic_glcm_graymap_effect():
    folder_path = r'D:\GitHubProj\Logging_Interpretation\test\texture_set'
    path_list = get_all_file_paths(folder_path)
    distance = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4]
    level = 16
    pic_all_list = []
    feature_all_list = []
    str_list_all = []

    print(path_list)
    for path in path_list:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))

        _, _, feature, glcm_matric = get_glcm_Features(img, level=level, distance=distance, angles=angles)
        pic_all_list.append(img)
        pic_all_list.append(glcm_matric[:, :, 0, 0])
        feature_all_list.append(feature[:, 0, 0])

        pic_list_glcm_matric = [img, glcm_matric[:, :, 0, 0]]
        pic_str = ['原始图像', 'GLCM_AVERAGE']
        # 不同distance的图像 灰度矩阵添加
        for i in range(len(distance)):
            pic_list_glcm_matric.append(glcm_matric[:, :, i+1, 0])
            pic_str.append('GLCM_distance_{}'.format(distance[i]))

        # 不同angle的图像 灰度矩阵对添加
        for i in range(len(angles)):
            pic_list_glcm_matric.append(glcm_matric[:, :, len(distance)+i+1, 0])
            pic_str.append('GLCM_angle_{:.2f}Π'.format(angles[i] / np.pi))

        pic_order = '42'

        CHARTER = path.split('/')[-1].split('.')[0]
        show_Pic(pic_list_glcm_matric, pic_order=pic_order, save_pic=True,
                 path_save='C:\\Users\\amd\\Desktop\\PIC_TEXTURE\\{}.png'.format(CHARTER),
                 title=path.split('/')[-1].split('.')[0],
                 pic_str=pic_str, figure=(9, 16))
        # print(feature.shape, glcm_matric.shape)
        print(CHARTER, feature.shape, feature.ravel()[:6])
        str_list_all.append(CHARTER)
        str_list_all.append('GLCM_MATRIC')

    show_Pic(pic_all_list, pic_order='46', pic_str=str_list_all, save_pic=True,
             path_save='C:\\Users\\amd\\Desktop\\PIC_TEXTURE\\Abstract.png')
    # 设置打印位数为2 设置是否使用科学计数法
    np.set_printoptions(precision=2, suppress=True)
    print(np.array(feature_all_list))


def show_pic_glcm_graymap_effect_x_y():
    folder_path = r'D:\GitHubProj\Logging_Interpretation\test\texture_set'
    path_list = get_all_file_paths(folder_path)
    distance = [1, 2]
    angles = [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4]
    level = 16
    pic_all_list = []
    feature_all_list = []
    str_list_all = []

    print(path_list)
    for path in path_list:
        # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img, _ = get_ele_data_from_path(path)
        img = cv2.resize(img, (128, 128))

        _, _, feature_x, glcm_matric_x = get_glcm_Features(img, level=level, distance=distance, angles=[0])
        _, _, feature_y, glcm_matric_y = get_glcm_Features(img, level=level, distance=distance, angles=[np.pi / 2])
        _, _, feature_mean, glcm_matric_mean = get_glcm_Features(img, level=level, distance=distance, angles=angles)

        pic_all_list.append(img)
        pic_all_list.append(glcm_matric_mean[:, :, 0, 0])
        pic_all_list.append(glcm_matric_x[:, :, 0, 0])
        pic_all_list.append(glcm_matric_y[:, :, 0, 0])
        feature_all_list.append(feature_mean[:, 0, 0])

        pic_list_glcm_matric = [img, glcm_matric_mean[:, :, 0, 0],
                                glcm_matric_x[:, :, 0, 0], glcm_matric_y[:, :, 0, 0]]
        pic_str = ['原始图像', 'GLCM_AVERAGE', 'GLCM_X', 'GLCM_Y']
        pic_order = '22'
        CHARTER = path.split('/')[-1].split('.')[0]
        show_Pic(pic_list_glcm_matric, pic_order=pic_order, save_pic=True,
                 path_save='C:\\Users\\amd\\Desktop\\PIC_TEXTURE\\{}_angle.png'.format(CHARTER),
                 title=path.split('/')[-1].split('.')[0],
                 pic_str=pic_str, figure=(9, 9))
        # print(feature.shape, glcm_matric.shape)
        # print(CHARTER, feature.shape, feature.ravel()[:6])
        str_list_all.append(CHARTER)
        str_list_all.append('GLCM_MATRIC_MEAN')
        str_list_all.append('GLCM_MATRIC_X')
        str_list_all.append('GLCM_MATRIC_Y')

    print(len(pic_all_list))
    show_Pic(pic_all_list, pic_order='68', pic_str=str_list_all, save_pic=True, figure=(24, 12),
             path_save='C:\\Users\\amd\\Desktop\\PIC_TEXTURE\\Abstract_angle.png')
    # 设置打印位数为2 设置是否使用科学计数法
    np.set_printoptions(precision=2, suppress=True)
    print(np.array(feature_all_list))


import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy


def image_similarity(pic1: np.ndarray, pic2: np.ndarray) -> dict:
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
        "FeatureMatching_MatchScore": 0.0
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


# # 生成测试图像
# img1 = np.random.randint(0, 255, (100,100), dtype=np.uint8)
# img2 = np.random.randint(0, 255, (100,100), dtype=np.uint8)
#
# # 调用接口
# result = image_similarity(img1, img2)
#
# # 输出结果示例
# print(f"""
# MSE: {result['MSE']:.2f}
# PSNR: {result['PSNR']:.2f} dB
# SSIM: {result['SSIM']:.4f}
# 直方图相似度:
#   - 巴氏距离: {result['Histogram']['Bhattacharyya']:.4f}
#   - 卡方距离: {result['Histogram']['ChiSquare']:.2f}
#   - KL散度: {result['Histogram']['KLD']:.4f}
# 特征匹配:
#   - 匹配点数量: {result['FeatureMatching']['MatchCount']}
#   - 平均距离: {result['FeatureMatching']['MatchScore']:.2f}
# """)

# if __name__ == '__main__':
    # show_pic_glcm_graymap_effect_x_y()
