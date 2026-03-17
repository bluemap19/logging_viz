import copy
import random
import numpy as np
import cv2
import os
from src_fmi.fmi_data_read import get_random_ele_data
from src_fmi.image_operation import show_Pic, pic_open_close_random


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

    for j in range(random_times):
        for i in r:
            x = i // input.shape[1]
            y = i % input.shape[1]

            data_new[x][y] = process_pix(x, y, input, windows_shape, max_pixel, ratio_top, ratio_migration)

    return data_new



def test_pic_random_enhance_effect():
    img_stat, img_dyna, data_depth = get_random_ele_data()
    print(img_stat.shape, data_depth.shape)

    processing_pic = img_stat[0:600, :]
    pic_EH = pic_enhence_random(processing_pic, windows_shape=3, ratio_top=0.1, ratio_migration=0.3, random_times=1)
    pic_equalizeHist = cv2.equalizeHist(processing_pic)  # 直方图均衡化

    show_Pic([processing_pic, pic_EH], path_save=False, pic_order='12', pic_str=['pic_org', 'pic_enhance'])
# if __name__ == '__main__':
#     test_pic_random_enhance_effect()


def pic_smooth_effect_compare():
    from matplotlib import pyplot as plt
    os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
    plt.rcParams['font.family'] = 'SimHei'
    data_img_dyna, data_img_stat, data_depth = get_random_ele_data()
    print('data_image shape:{}'.format(data_img_dyna.shape))

    data_img = cv2.resize(data_img_dyna,(256, 256))

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