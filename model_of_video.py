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

SIZE = 2048
SIZE_HALF = int(SIZE / 2)
mean_of_all = 72.6219124403117


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
    image_array = cv2.imread("frames_orig_video/frame0.png")[:, :, 0]
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
    # print("DFT [0][0]",fft[0][0])
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
    print("DFT[0][0] ", fft[0][0])
    # fft[0]=0
    # fft[0][0]=0
    abs_fft = np.power(np.abs(fft), 2)
    print("DFT abs[0][0] ", abs_fft[0][0])
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


def plot_ACF(img):
    mean_by_mean = np.mean(img[:, :])
    # print(img.shape, len(pair_lst), mean_by_mean)

    ifft_matr = ACF_by_periodogram2(img[:, :])
    print("ACF of image", ifft_matr[10:])
    shift_matr = np.fft.fftshift(ifft_matr)
    spectrum = np.fft.fft2(ifft_matr)
    imag = np.imag(spectrum)

    display_4 = ifft_matr / np.max(ifft_matr) * 255

    check_row = (ifft_matr[0, :])
    check_row -= mean_by_mean ** 2
    print(check_row)

    # count = 0
    # all_mse = []
    # params = []
    # for p in np.arange(0.1, 0.98, 0.1):
    #     for betta in np.arange(0.001, 0.021, 0.002):
    #         for alfa in np.arange(0.0001, 0.002, 0.0002):
    #             list_ACF = []
    #             print(p)
    #             for x in range(1000):
    #                 # for y in range(100):
    #                 list_ACF.append(check_row[0] * calc_ACF(p, betta, alfa, x))
    #
    #             params.append((p, betta, alfa))
    #             all_mse.append(mean_squared_error(list(check_row[:1000]), list(list_ACF[:1000])))
    #             print(list_ACF[:30])
    #             print(count, ":  ", p, betta, alfa)
    #             print(mean_squared_error(list(check_row[:30]), list(list_ACF[:30])))
    #             count += 1
    #
    # print("ALL ABOUT MSE",min(all_mse), all_mse.index(min(all_mse)))
    # need_params2 = params[all_mse.index(min(all_mse))]
    need_params2 = (0.5, 0.017, 0.0019)
    # need_params = (0.76, 0.1, 0.00005)

    # list_ACF = []
    # for x in range(3000):
    #     list_ACF.append(check_row[0]*calc_ACF(need_params[0], need_params[1], need_params[2], x))

    list_ACF2 = np.zeros((SIZE, SIZE))
    tmp_matr = np.zeros((SIZE_HALF, SIZE_HALF))
    for x in range(0, SIZE_HALF):
        for y in range(0, SIZE_HALF):
            tmp_matr[x][y] = (check_row[0] * calc_ACF2(need_params2[0], need_params2[1], need_params2[2], x, y))

    list_ACF2[SIZE_HALF:, SIZE_HALF:] = tmp_matr[:SIZE_HALF, :SIZE_HALF]
    # for x in range(0, 64):
    #     for y in range(0, 64):
    for x in range(SIZE_HALF):
        for y in range(SIZE_HALF):
            list_ACF2[SIZE_HALF - x, SIZE_HALF - y] = tmp_matr[x, y]
            list_ACF2[SIZE_HALF + x, SIZE_HALF - y] = tmp_matr[x, y]
            list_ACF2[SIZE_HALF - x, SIZE_HALF + y] = tmp_matr[x, y]

    # for i in range(SIZE):
    #     list_ACF2[0][i] = list_ACF2[i][0]= 64

    list_ACF2 = np.fft.fftshift(list_ACF2)
    spectrum_my = np.fft.fft2(list_ACF2)

    # display_4 = list_ACF2 / np.max(list_ACF2) * 255
    # img1 = Image.fromarray(display_4.astype('uint8'))
    # img1.save(r"D:/pythonProject/phase_wm\ACF_2d" + ".png")
    # list_ACF2 = np.array(list_ACF2).reshape((1, len(list_ACF2)))
    #
    # akf_2d = np.sqrt(np.outer(list_ACF2, list_ACF2))
    #
    # print(mean_squared_error(list(check_row[:100]), list(list_ACF2[:100])))
    #
    # list_ACF_exp = []
    # for x in range(3000):
    #     list_ACF_exp.append(check_row[0]*calc_ACF(need_params22[0], need_params22[1], need_params22[2], x))
    #
    # plt.plot(check_row[:SIZE], label="Average ACF")
    # plt.plot(list_ACF_exp[:100], label="Exper ACF")
    # plt.plot(list_ACF2[0, :SIZE], label="Model ACF")
    #
    # plt.title("Spatial")
    # plt.legend()
    # plt.show()
    print("check row", check_row)

    return check_row


def plot_ACF_video(path_video, my_square, top_treshold):
    matr_time = np.zeros((len(my_square), top_treshold))

    count = 0
    vidcap = cv2.VideoCapture(path_video)
    # matr_time = np.zeros(( vidcap.read()[1].shape[0], 2048))

    success = True
    while success and count < top_treshold:

        success, image = vidcap.read()

        image = image[:1080, :1920]
        print(count,np.mean(image[:,:,0]),np.var(image[:,:,0]))
        if count == 0:
            print(image.shape)
        for i in range(len(my_square)):
            matr_time[i, count] = image[my_square[i][0], my_square[i][1], 0]
        count += 1

    mean_matr_time = np.mean(matr_time, axis=0)
    plt.plot(mean_matr_time)
    plt.show()
    mean_by_mean = np.mean(matr_time)

    print("MBM",mean_by_mean)
    # ifft_matr = np.zeros(matr_time.shape)
    ifft_matr = np.zeros((len(my_square), top_treshold))
    for i in range(matr_time.shape[0]):
        # print(len(matr_time[i, :]))
        matr_time[i, 0]= 0
        ifft_matr[i, :] = ACF_by_periodogram(matr_time[i, :])
        # ifft_matr[i, :] = ifft_matr[i, :] - np.mean(matr_time[i, :]) ** 2
        # ifft_matr[i, :] /= np.max(ifft_matr[i, :])

    mean_ifft = np.mean(ifft_matr, axis=0)

    print("ACF", mean_ifft[:10])
    ifft_matr -= mean_by_mean ** 2
    mean_ifft -= mean_by_mean ** 2
    plt.plot(mean_ifft[:200],label="mean ifft")
    plt.plot(ifft_matr[0,:200],label="frame=0")
    plt.plot(ifft_matr[1000, :200],label="frame=1000")
    plt.plot(ifft_matr[1111, :200],label="frame=1111")
    plt.plot(ifft_matr[1511, :200],label="frame=1511")
    plt.plot(ifft_matr[1811, :200],label="frame=1811")
    plt.plot(ifft_matr[2061, :200],label="frame=2061")
    plt.plot(ifft_matr[2311, :200],label="frame=2311")
    plt.plot(ifft_matr[2711, :200],label="frame=2711")
    plt.legend()
    plt.show()
    # # mean_ifft /= np.max(mean_ifft)
    check_row = list(mean_ifft)

    count = 0
    # all_mse = []
    # params = []
    # for p in np.arange(0.1, 0.98, 0.1):
    #     for betta in np.arange(9.9e-4, 9e-2, 5e-3):
    #         for alfa in np.arange(9.9e-4, 9e-2, 5e-3):
    #             list_ACF = []
    #             print(p)
    #             for x in range(100):
    #                 for y in range(100):
    #                     list_ACF.append(check_row[0] * calc_ACF2(p, betta, alfa, x,y))
    #
    #             params.append((p, betta, alfa))
    #             all_mse.append(mean_squared_error(list(check_row[100:1000]), list(list_ACF[100:1000])))
    #             print(list_ACF[:30])
    #             print(count, ":  ", p, betta, alfa)
    #             print(mean_squared_error(list(check_row[:30]), list(list_ACF[:30])))
    #             count += 1
    #
    # print("ALL ABOUT MSE",min(all_mse), all_mse.index(min(all_mse)))
    # need_params = params[all_mse.index(min(all_mse))]
    # print("need params",need_params)
    # need_params = (0.76, 0.1, 0.00005)

    # list_ACF = []
    # for x in range(3000):
    #     list_ACF.append(check_row[0]*calc_ACF(need_params[0], need_params[1], need_params[2], x))

    print("check row", check_row)
    return check_row


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


def gener_field(list_ACF2, seed):
    step1 = list_ACF2 / np.max(list_ACF2) * 255
    # img1 = Image.fromarray(step1.astype('uint8'))
    # img1.save(r"D:/pythonProject/phase_wm\step1_" + str(seed) + ".png")

    var2 = np.abs(np.fft.fft2(list_ACF2))
    # var2[0][0] = 0
    # step15 = var2 / np.max(var2) * 255
    # img1 = Image.fromarray(step15[0:5, 0:5].astype('uint8'))
    # img1.save(r"D:/pythonProject/phase_wm\step15_" + str(seed) + ".png")

    var2 = np.sqrt(var2)
    # step2 = var2 / np.max(var2) * 255
    # img1 = Image.fromarray(step2[:5, :5].astype('uint8'))
    # img1.save(r"D:/pythonProject/phase_wm\step2_" + str(seed) + ".png")

    np.random.seed(seed)
    var1 = np.random.rand(SIZE, SIZE)
    # print(var1)

    var1 = np.fft.fft2(var1)
    # step3 = var1 / np.max(var1) * 255
    # img1 = Image.fromarray(np.real(var1).astype('uint8'))
    # img1.save(r"D:/pythonProject/phase_wm\exper_noise" + str(seed) + ".png")

    # img1 = Image.fromarray(np.imag(var1).astype('uint8'))
    # img1.save(r"D:/pythonProject/phase_wm\exper_noise_imag" + str(seed) + ".png")

    un_var = var1 * var2
    # step5 = un_var / np.max(un_var) * 255
    # step5_real, step5_imag = step5.real, step5.imag
    # step5_abs = np.abs(step5)
    # img1 = Image.fromarray(np.abs(step5).astype('uint8'))
    # img1.save(r"D:/pythonProject/phase_wm\step5_" + str(seed) + ".png")

    final_res = np.fft.ifft2(un_var)
    print("MO of synthes",np.mean(final_res))
    final_res -= np.mean(final_res)
    new_arr = np.mod(np.real(final_res), 256)
    imag = np.imag(final_res)
    print(np.where(new_arr == np.min(new_arr)), np.where(new_arr == np.max(new_arr)))
    # img1 = Image.fromarray(np.abs(final_res).astype('uint8'))
    # img1.save(r"D:/pythonProject/phase_wm\new_simtez_image" + str(seed) + ".png")
    # mo1, varance1 = np.mean(final_res), np.var(final_res)
    img2 = Image.fromarray(np.abs(new_arr).astype('uint8'))
    img2.save(r"D:/pythonProject/phase_wm\new_simtez_image_real" + str(seed) + ".png")
    mo2, varance2 = np.mean(new_arr), np.var(new_arr)
    print("MO/variance", mo2, varance2)
    print("MIMIMAX", np.min(final_res), np.max(final_res))
    print("MIMIMAX2", np.min(new_arr), np.max(new_arr))

    return np.real(final_res)


if __name__ == '__main__':
    np.random.seed(42)
    rand_list = np.random.choice(1080, 100, replace=False)
    np.random.seed(43)
    rand_list2 = np.random.choice(1080, 100, replace=False)
    vidcap = cv2.VideoCapture("cut_RealBarca120.mp4")
    success = True


    pair_lst = []
    for i in rand_list:
        for j in rand_list2:
            pair_lst.append([i,j])
    my_square = pair_lst

    """
    print(pair_lst)
    full_ACF = np.zeros((60,1920))
    for cnt in range(60):
        # image_orig = io.imread("D:/pythonProject/phase_wm/embedding_BBC/result"+str(cnt)+".png")[:,:,0]
        image_orig = io.imread("D:/pythonProject/phase_wm/mosaics/mosaic" + str(cnt) + ".png")
        full_ACF[cnt,:] = plot_ACF(image_orig)

    avg_ACF = np.mean(full_ACF,axis=0)
    np.save("D:/pythonProject/phase_wm/mosaics/avg.py",avg_ACF)
    plt.plot(avg_ACF)
    plt.show()
    """

    # for cnt in range(0,60,10):
    #     # image_orig = io.imread("D:/pythonProject/phase_wm/embedding_BBC/result"+str(cnt)+".png")[:,:,0]
    #     image_orig = io.imread("D:/pythonProject/phase_wm/mosaics/mosaic" + str(cnt) + ".png")
    #     curr_ACF=plot_ACF(image_orig)
    #     plt.plot(curr_ACF,label="mosaic"+str(cnt))
    #
    # avg = np.load("D:/pythonProject/phase_wm/mosaics/avg.py.npy")
    # plt.plot(avg,label="average")
    # plt.legend()
    # plt.show()

    # mdl = io.imread(r"D:/pythonProject/phase_wm/fold_model_video/final_img" + str(0) + ".png")
    # graph_mdl = plot_ACF(mdl)
    # plt.plot(graph_mdl[:200], label="Model ACF")
    # graph_video_orig = plot_ACF_video(r"cut_RealBarca120.mp4", my_square, 3072)
    # plt.plot(graph_video_orig[:200], label="Original video ACF")
    # graph_video_model = plot_ACF_video(r"D:/pythonProject/phase_wm/fold_model_video/need_video.mp4", my_square,2048)
    # plt.plot(graph_video_model[:200], label="Model ACF 5")
    # plt.legend()
    # plt.show()
    image = io.imread(r"D:\pythonProject\phase_wm\fold_model_video/final_img0.png")

    # mosaic = io.imread(r"D:\pythonProject\phase_wm/mosaic0.png")
    # txture = io.imread(r"D:\pythonProject\phase_wm/new_simtez_image0.png")
    # final_frame = np.where((txture + mosaic) < 255, txture + mosaic, 255)
    # img = Image.fromarray(final_frame.astype('uint8'))
    # img.save(r"D:/pythonProject/phase_wm/final_img" + str(100) + ".png")
    # image = io.imread(r"D:\pythonProject\phase_wm\exper_model/frame0.png")[:1920, :1080,0]
    # print(np.mean(image_orig),"variances",np.var(image_orig), np.var(image))

    # graph_model,graph_non_need_model = plot_ACF(image)
    #
    graph_video_orig = plot_ACF_video(r"cut_RealBarca120.mp4", my_square, 2048)
    import os

    # os.system(
    #         f"ffmpeg -y -i D:/pythonProject/phase_wm/fold_model_video/need_video14_4096_3703.mp4 -b:v {5.5}"
    #         f"M -vcodec libx264  D:/pythonProject/phase_wm/fold_model_video/need_video14_4096_3703_codec.mp4")
    # graph_video_model34_cod = plot_ACF_video(r"D:/pythonProject/phase_wm/fold_model_video"
    #                                          r"/need_video14_4096_3703_codec.mp4",
    #                                      my_square, 4096)

    graph_video_model34 = plot_ACF_video(r"D:/pythonProject/phase_wm/fold_model_video/need_video41_4096_9702.mp4",
                                         my_square, 4096)

    # graph_video_model30 = plot_ACF_video(r"D:/pythonProject/phase_wm/fold_model_video/need_video41_4096.mp4", my_square,4096)
    # graph_video_model30_2 = plot_ACF_video(r"D:/pythonProject/phase_wm/fold_model_video/need_video41_4096.mp4", my_square,2048)
    # graph_video_model30_3 = plot_ACF_video(r"D:/pythonProject/phase_wm/fold_model_video/need_video41_4096.mp4",
    #                                        my_square, 1024)
    # graph_video_model100 = plot_ACF_video(r"D:/pythonProject/phase_wm/fold_model_video/need_video100.mp4", my_square,2048)
    # graph_video_model13 = plot_ACF_video(r"D:/pythonProject/phase_wm/fold_model_video/need_video13.mp4", my_square,2048)
    # plt.plot(graph_orig, label="Original image ACF")
    # plt.plot(graph_model, label="Model ACF")
    plt.plot(graph_video_orig[:200], label="Original video ACF")
    # plt.plot(graph_video_model34_cod[:200], label="Compress model ACF 14")
    plt.plot(graph_video_model34[:200], label="Model ACF 14")
    # plt.plot(graph_video_model30[:200], label="Model ACF 41(4096)")
    # plt.plot(graph_video_model30_2[:200], label="Model ACF 41(2048)")
    # plt.plot(graph_video_model30_3[:200], label="Model ACF 41(1024)")
    # plt.plot(graph_video_model13[:200], label="Model ACF 13")
    # plt.plot(graph_video_model100[:200], label="Model ACF 100")
    plt.title("Time")
    plt.legend()
    plt.show()

    """
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

    # for i in range(10):

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
    """
