import math
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool
import statsmodels.api as sm
from PIL import Image
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

    ifft = np.abs(ifft) / fft.size

    return ifft


def signal_by_ACF(ACF):
    ACF *= ACF.size
    fft = np.fft.fft(ACF)
    sqrt_fft = np.sqrt(np.abs(fft))

    ifft = np.fft.ifft(sqrt_fft)
    ifft = np.abs(ifft)

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


def reconstruct_signal(acf):
    N = len(acf) // 2  # Длина исходного сигнала
    R = np.zeros((N, N))  # Матрица автокорреляционных коэффициентов

    # Заполнение матрицы автокорреляционных коэффициентов
    for i in range(N):
        for j in range(N):
            R[i, j] = acf[N - 1 + abs(i - j)]

    # Вычисление вектора автокорреляции
    r = acf[N:2 * N]

    # Вычисление коэффициентов регрессии
    a = np.linalg.inv(R) @ r

    # Восстановление сигнала
    reconstructed_signal = np.zeros(N)
    for i in range(N):
        for j in range(N - i):
            reconstructed_signal[i] += a[j] * reconstructed_signal[i + j]

    return reconstructed_signal


if __name__ == '__main__':

    np.random.seed(42)
    rand_list = np.random.choice(1000, 1000, replace=False)

    vidcap = cv2.VideoCapture("cut_RealBarca120.mp4")
    success = True

    pair_lst = [(rand_list[i - 1], rand_list[i]) for i in range(1, len(rand_list))]
    ext_list = [(rand_list[i], rand_list[i - 1]) for i in range(1, len(rand_list))]

    pair_lst.extend(ext_list)
    my_square = pair_lst
    print(pair_lst)

    image = io.imread("D:/pythonProject/phase_wm/embedding_BBC/result0.png")
    matr_time = np.zeros((len(pair_lst), 2048))
    # permut = list(product([250,275,555,551, 300, 350, 375, 400, 425, 450, 475, 500, 550, 600, 650, 675, 700, 750], repeat=2))
    # permut = list(product(my_square, repeat=2))

    count = 0

    mean_by_mean = np.mean(image[:, :, 0])
    print(len(pair_lst), mean_by_mean)
    # ifft_matr = np.zeros(matr_time.shape)
    # ifft_matr = np.zeros((len(my_square), 2048))
    # for i in range(matr_time.shape[0]):
    # success = True
    # while success and count < 2048:
    #
    #     success, image = vidcap.read()
    #     for i in range(len(pair_lst)):
    #         # print(i)
    #         matr_time[i, count] = image[pair_lst[i][0], pair_lst[i][1], 0]
    #     count += 1

    # mean_matr_time = np.mean(matr_time, axis=0)
    # mean_by_mean = np.mean(mean_matr_time)
    # # ifft_matr = np.zeros(matr_time.shape)
    # for i in range(matr_time.shape[0]):
    #     # print(len(matr_time[i, :]))
    #     ifft_matr[i, :] = ACF_by_periodogram(matr_time[i, :])
    #     # ifft_matr[i, :] = ifft_matr[i, :] - np.mean(matr_time[i, :]) ** 2
    #     # ifft_matr[i, :] /= np.max(ifft_matr[i, :])

    ifft_matr = ACF_by_periodogram2(image[:, :, 0])

    shift_matr = np.fft.fftshift(ifft_matr)
    spectrum = np.fft.fft2(ifft_matr)
    imag= np.imag(spectrum)

    display_4 = ifft_matr / np.max(ifft_matr) * 255
    img1 = Image.fromarray(display_4.astype('uint8'))
    img1.save(r"D:/pythonProject/phase_wm\ACF_2d_original" + ".png")

    # mean_ifft = np.mean(ifft_matr, axis=0)
    # mean_ifft -= mean_by_mean ** 2
    # mean_ifft /= np.max(mean_ifft)
    check_row = (ifft_matr[0, :])
    check_row -= np.mean(image) ** 2

    # while len(check_row)<2048:
    #     check_row.append(0)

    # check_row -= np.mean(image[0,:1080,0])*np.mean(image[0,:1080,0])
    # check_row /= np.max(check_row)
    # check_row = list(mean_ifft)

    # for i in range(0,1000,300):
    #     plt.plot(ifft_matr[i,:]/np.max(ifft_matr[i,:]),label="string " + str(i))

    # # plt.plot(ifft1000, label="pixel[1000][1000]")
    # # plt.plot(ifft900, label="string 900")
    #
    # plt.xlabel("Number of frame")
    # plt.ylabel("ACF")

    # all_mse = []
    # params = []
    # for p in np.arange(0.1, 0.98, 0.1):
    #     for betta in np.arange(0.01,0.21, 0.02):
    #         for alfa in np.arange(0.001, 0.02,  0.002):
    #             list_ACF = []
    #             print(p)
    #             for x in range(500):
    #                 # for y in range(100):
    #                 list_ACF.append(check_row[0]*calc_ACF(p, betta, alfa, x))
    #
    #             params.append((p,betta,alfa))
    #             all_mse.append(mean_squared_error(list(check_row[:500]), list(list_ACF[:500])))
    #             print(list_ACF[:30])
    #             print(count, ":  ",p,betta,alfa)
    #             print(mean_squared_error(list(check_row[:30]), list(list_ACF[:30])))
    #             count += 1
    #
    # print(min(all_mse), all_mse.index(min(all_mse)))
    # need_params2 = params[all_mse.index(min(all_mse))]
    # print(need_params2)
    # need_params2 = (0.6, 0.9, 0.05)
    need_params2 = (0.6, 0.03, 0.003)
    # need_params = (0.76, 0.1, 0.00005)

    # list_ACF = []
    # for x in range(3000):
    #     list_ACF.append(check_row[0]*calc_ACF(need_params[0], need_params[1], need_params[2], x))

    SIZE = 2048
    SIZE_HALF = int(SIZE/2)

    list_ACF2 = np.zeros((SIZE, SIZE))
    tmp_matr= np.zeros((SIZE_HALF, SIZE_HALF))
    for x in range(0, SIZE_HALF):
        for y in range(0, SIZE_HALF):
            tmp_matr[x][y] = (check_row[0] * calc_ACF2(need_params2[0], need_params2[1], need_params2[2], x, y))

    list_ACF2[SIZE_HALF:,SIZE_HALF:] = tmp_matr[:SIZE_HALF,:SIZE_HALF]
    # for x in range(0, 64):
    #     for y in range(0, 64):
    for x in range(SIZE_HALF):
        for y in range(SIZE_HALF):
            list_ACF2[SIZE_HALF-x,SIZE_HALF-y] = tmp_matr[x,y]
            list_ACF2[SIZE_HALF + x, SIZE_HALF - y] = tmp_matr[x, y]
            list_ACF2[SIZE_HALF - x, SIZE_HALF + y] = tmp_matr[x, y]

    # for i in range(SIZE):
    #     list_ACF2[0][i] = list_ACF2[i][0]= 64

    list_ACF2 = np.fft.fftshift(list_ACF2)
    spectrum_my = np.fft.fft2(list_ACF2)

    real_my = np.real(spectrum_my)
    imag_my = np.imag(spectrum_my)
    print("my imag",imag_my)
    display_4 = list_ACF2 / np.max(list_ACF2) * 255
    img1 = Image.fromarray(display_4.astype('uint8'))
    img1.save(r"D:/pythonProject/phase_wm\ACF_2d" + ".png")
    # list_ACF2 = np.array(list_ACF2).reshape((1, len(list_ACF2)))
    #
    # akf_2d = np.sqrt(np.outer(list_ACF2, list_ACF2))

    # print(mean_squared_error(list(check_row[:100]), list(list_ACF2[:100])))
    #
    # list_ACF_exp = []
    # for x in range(3000):
    #     list_ACF_exp.append(check_row[0]*calc_ACF(need_params22[0], need_params22[1], need_params22[2], x))

    # plt.plot(check_row[:SIZE], label="Average ACF")
    # plt.plot(list_ACF_exp[:100], label="Exper ACF")
    plt.plot(list_ACF2[0, :SIZE], label="Model ACF")

    plt.title("Spatial")
    plt.legend()
    plt.show()

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

    tau = np.arange(-2, 2, 0.01)
    akf = 0.5 * np.exp(-0.01 * tau) + 0.5 * np.exp(-0.23 * tau)  # Вычислите АКФ для каждого значения tau
    energy_spectrum = np.abs(np.fft.fft(akf)) ** 2

    # plt.show()
    # print(list_ACF[:3000])
    # print(union_list[:3000])

    """
    from mpl_toolkits.mplot3d import Axes3D

    # Создание данных для кривых
    x = np.arange(0, 100, 1)  # Значения по оси x

    y1 = list_ACF[:100] # Значения для первой кривой
    y2 = list_ACF2[:100]  # Значения для второй кривой

    # Создание сетки для построения поверхности
    X, Y = np.meshgrid(x, x)

    Z = np.dot(np.reshape(y2, (100, 1)),np.reshape(y1, (1, 100)))

    # var2_matr = np.zeros((2048,2048))
    # var2_matr[:ifft_matr.shape[0], :ifft_matr.shape[1]] = ifft_matr

    # img1 = Image.fromarray(var2_matr.astype('uint8'))
    # img1.save(r"D:/pythonProject/phase_wm\ACF_image_2d" + ".png")
    """
    for i in range(10):
        step1= list_ACF2 / np.max(list_ACF2) * 255
        img1 = Image.fromarray(step1.astype('uint8'))
        img1.save(r"D:/pythonProject/phase_wm\step1_" + str(i) + ".png")

        var2 = np.abs(np.fft.fft2(list_ACF2))
        # var2[0][0]=0
        step15 = var2 / np.max(var2) * 255
        img1 = Image.fromarray(step15.astype('uint8'))
        img1.save(r"D:/pythonProject/phase_wm\step15_" + str(i) + ".png")

        var2 = np.sqrt(var2)
        step2 = var2 / np.max(var2) * 255
        img1 = Image.fromarray(step2.astype('uint8'))
        img1.save(r"D:/pythonProject/phase_wm\step2_" + str(i) + ".png")

        np.random.seed(i)
        var1 = np.random.rand(SIZE, SIZE)
        # print(var1)

        var1 = np.fft.fft2(var1)
        step3 = var1 / np.max(var1) * 255
        img1 = Image.fromarray(var1.astype('uint8'))
        img1.save(r"D:/pythonProject/phase_wm\exper_noise" + str(i) + ".png")
        # var1 = np.fft.fftshift(var1)
        # var2 = np.fft.fftshift(var2)

        # exper = np.fft.ifft2(var1)

        un_var = var1*var2

        final_res = np.fft.ifft2(un_var)
        imag = np.imag(final_res)
        print(final_res)
        img1 = Image.fromarray(final_res.astype('uint8'))
        img1.save(r"D:/pythonProject/phase_wm\simtez_image" + str(i) + ".png")

    # Пример функции для создания поверхности из кривых
    tensor = np.stack((X, Y, Z))

    # Создание трехмерной фигуры
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Построение поверхности
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # Настройка осей и меток
    ax.set_xlabel('T')
    ax.set_ylabel('r')
    ax.set_zlabel('ACF')
    ax.legend()

    # Отображение графика
    # plt.show()
