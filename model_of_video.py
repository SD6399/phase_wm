import math
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool
import statsmodels.api as sm
import cv2
from skimage import io


def avg(lst):
    return sum(lst) / len(lst)


def read_video(path, coord_x, coord_y):
    vidcap = cv2.VideoCapture(path)
    count = 0
    list00 = []
    dsp00 = []
    temp00 = []

    success = True
    while success:
        success, image = vidcap.read()
        if success:
            p00 = None
            if count != 0:
                temp00.append(image[coord_x, coord_y, 0] * p00)
            p00 = int(image[coord_x, coord_y, 0])
            list00.append(p00)
            dsp00.append(p00 * p00)

        print("записан кадр", count)

        if cv2.waitKey(10) == 27:
            break
        count += 1
    mog00 = avg(list00)

    avg00_2 = avg(dsp00)
    av_2_00 = avg(temp00)
    print("MO", mog00)
    print("MO^2", avg00_2, )
    print("Temporary", av_2_00, )
    print("Variance", avg00_2 - mog00 * mog00)
    print("ACF", av_2_00 - mog00 * mog00)

    return mog00, avg00_2 - mog00 * mog00, av_2_00 - mog00 * mog00


def bracket_1():
    p = 0.01
    alf = 0.0016
    betta = 0.01
    i, j = np.indices((1920, 1080))
    r = np.sqrt(i ** 2 + j ** 2)
    new_matr = (p * np.exp(-alf * r)) + ((1 - p) * np.exp(-betta * r))

    return new_matr


def bracket_2():
    p = 0.01
    alf = 0.0016
    betta = 0.01
    i = np.indices((3000, 1))
    new_matr = (p * math.e ** (-alf * i[0, :, 0]) + (1 - p) * math.e ** (-betta * i[0, :, 0]))
    new_matr = np.ravel(new_matr)

    return new_matr


def DFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)

    X = np.dot(e, x)

    return X


from scipy.signal import correlate2d
from scipy.signal.windows import hann


def ACF_image():
    image_array = io.imread("frames_orig_video/frame0.png")[:, :, 0]
    height, width = image_array.shape
    acf = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            acf += image_array[i, j] * image_array[i + x, j + y]

    # Normalize the autocorrelation function
    acf = acf / np.max(acf)

    return acf


# ACF(x, y) = ∑[I(i, j) * I(i + x, j + y)]
def ACF_by_periodogram(lst):
    lst_pix = np.array(lst)

    # fft = np.fft.fft2(image_array)
    fft = np.fft.fft(lst_pix)

    # fft[0]=0
    # fft[0][0]=0
    abs_fft = np.power(np.abs(fft), 2)
    ifft = np.fft.ifft(abs_fft)
    ifft = np.abs(ifft) / fft.size

    return ifft


def ACF_by_periodogram2(lst):
    lst_pix = np.array(lst)

    # fft = np.fft.fft2(image_array)
    fft = np.fft.fft2(lst_pix)

    # fft[0]=0
    # fft[0][0]=0
    abs_fft = np.power(np.abs(fft), 2)
    ifft = np.fft.ifft2(abs_fft)
    print(fft.size,ifft.size)
    ifft = np.abs(ifft) / fft.size

    return ifft


def calc_ACF2(p, betta, alf, x, y):
    # p = 0.001
    # alf = 0.0066
    # betta = 0.072
    # r = math.sqrt(coord_x * coord_x + coord_y * coord_y)

    # R = (p * math.e ** (-alf * r) + (1 - p) * math.e ** (-betta * r)) * \
    R = (p * math.e ** (-alf * math.sqrt(x ** 2 + y ** 2)) + (1 - p) * math.e ** (-betta * math.sqrt(x ** 2 + y ** 2)))

    return R


def calc_ACF(p, betta, alf, numb_frame):
    # p = 0.001
    # alf = 0.0066
    # betta = 0.072
    # r = math.sqrt(coord_x * coord_x + coord_y * coord_y)

    # R = (p * math.e ** (-alf * r) + (1 - p) * math.e ** (-betta * r)) * \
    R = (p * math.e ** (-alf * numb_frame) + (1 - p) * math.e ** (-betta * numb_frame))

    return R


if __name__ == '__main__':

    rand_list = np.random.choice(1000, 1000, replace=False)

    vidcap = cv2.VideoCapture("cut_RealBarca120.mp4")
    success = True

    pair_lst = [(rand_list[i - 1], rand_list[i]) for i in range(1, len(rand_list))]
    ext_list = [(rand_list[i], rand_list[i - 1]) for i in range(1, len(rand_list))]

    pair_lst.extend(ext_list)
    my_square = pair_lst

    image = io.imread("D:/pythonProject/phase_wm/embedding_BBC/result0.png")
    matr_time = np.array([])
    # permut = list(product([250,275,555,551, 300, 350, 375, 400, 425, 450, 475, 500, 550, 600, 650, 675, 700, 750], repeat=2))
    # permut = list(product(my_square, repeat=2))

    count = 0

    mean_by_mean = np.mean(image[:, :, 0])
    print(len(pair_lst), mean_by_mean)
    # ifft_matr = np.zeros(matr_time.shape)
    # for i in range(matr_time.shape[0]):
    # print(len(matr_time[i, :]))
    print(image.shape)
    ifft_matr = ACF_by_periodogram2(image[:, :, 0])
    # ifft_matr[i, :] = ifft_matr[i, :] - np.mean(matr_time[i, :]) ** 2
    # ifft_matr[i, :] /= np.max(ifft_matr[i, :])

    check_row = (ifft_matr[0, :1080] )
    check_row -= mean_by_mean*mean_by_mean
    check_row /= np.max(check_row)

    check_row = list(check_row)
    # for i in range(0,1000,300):
    #     plt.plot(ifft_matr[i,:]/np.max(ifft_matr[i,:]),label="string " + str(i))

    # plt.plot(check_row[:30], label="Average ACF")
    # # plt.plot(ifft1000, label="pixel[1000][1000]")
    # # plt.plot(ifft900, label="string 900")
    #
    # plt.xlabel("Number of frame")
    # plt.ylabel("ACF")

    all_mse = []
    params = []
    count = 0
    # for p in np.arange(0.69, 1, 0.1):
    #     for betta in np.arange(0.2,0.3, 0.01):
    #         for alfa in np.arange(0.002,0.1, 0.001):
    #             list_ACF = []
    #             for x in range(3000):
    #                 # for y in range(100):
    #                 list_ACF.append(calc_ACF(p, betta, alfa, x))
    #
    #             params.append((p,betta,alfa))
    #             all_mse.append(mean_squared_error(list(check_row[:100]), list(list_ACF[:100])))
    #             print(list_ACF[:30])
    #             print(count, ":  ",p,betta,alfa)
    #             print(mean_squared_error(list(check_row[:30]), list(list_ACF[:30])))
    #             count += 1

    # print(min(all_mse),all_mse.index(min(all_mse)))
    # need_params2= params[all_mse.index(min(all_mse))]
    # print(need_params2)
    need_params2 = (0.79, 0.26, 0.004)
    need_params = (0.5, 0.31, 0.009)

    list_ACF = []
    for x in range(3000):
        list_ACF.append(calc_ACF(need_params[0], need_params[1], need_params[2], x))

    list_ACF2 = []
    for x in range(3000):
        # for y in range(100):
        list_ACF2.append(calc_ACF(need_params2[0], need_params2[1], need_params2[2], x))

    print("LIsts of ACF")
    print(len(list_ACF))
    print(list(check_row))
    div_ACF_video = []
    div_ACF_model = []

    # union_list = [list_ACF[i]*list_ACF2[i] for i in range(100)]



    # for i in range(1, min(len(list_ACF), len(check_row))):
    #     div_ACF_video.append(check_row[i] / check_row[i - 1])
    #     div_ACF_model.append(list_ACF[i] / list_ACF[i - 1])

    from statistics import mean

    # print(mean(div_ACF_video[40:100]))
    # print(sum(div_ACF_video[30:100]) / len(div_ACF_video[30:100]), div_ACF_video)

    # plt.title("RealBarca")
    # plt.plot(list_ACF[:30], label="Model ACF")
    #
    # print("Shelf=", np.mean(div_ACF_video[20:60]))
    # print("Shelf of model=", np.mean(div_ACF_model[20:60]))
    # # plt.plot(acorr)
    # plt.legend()
    # plt.show()
    # pool = Pool()

    plt.title("ACF. RealBarca.")
    plt.xlabel("Count")
    plt.ylabel("ACF")
    print(div_ACF_video)
    print(div_ACF_model)
    print(list_ACF2[:300])
    # print(list_ACF[:3000])
    # print(union_list[:3000])

    from mpl_toolkits.mplot3d import Axes3D

    # Создание данных для кривых
    x = np.arange(0, 100, 1)  # Значения по оси x
    print(x)
    y1 = list_ACF[:100] # Значения для первой кривой
    y2 = list_ACF2[:100]  # Значения для второй кривой

    print(y1)
    print(y2)

    # Создание сетки для построения поверхности
    X, Y = np.meshgrid(x, x)

    Z = np.dot(np.reshape(y2, (100, 1)),np.reshape(y1, (1, 100)))  # Пример функции для создания поверхности из кривых
    print(Z.shape)

    tensor = np.stack((X, Y,Z))

    # Создание трехмерной фигуры
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Построение поверхности
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # Построение кривых
    # ax.plot(x, x, y1, color='red', label='Curve 1')
    # ax.plot(x, x, y2, color='blue', label='Curve 2')

    # Настройка осей и меток
    ax.set_xlabel('T')
    ax.set_ylabel('r')
    ax.set_zlabel('ACF')
    ax.legend()

    # Отображение графика
    plt.show()
