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


def calc_ACF(p, alf, betta, numb_frame):
    # p = 0.001
    # alf = 0.0066
    # betta = 0.072
    # r = math.sqrt(coord_x * coord_x + coord_y * coord_y)

    # R = (p * math.e ** (-alf * r) + (1 - p) * math.e ** (-betta * r)) * \
    R = (p * math.e ** (-alf * numb_frame) + (1 - p) * math.e ** (-betta * numb_frame))

    return R



def gencoordinates(m, n):
    seen = set()

    x, y = np.randint(m, n), np.randint(m, n)

    while True:
        seen.add((x, y))
        yield (x, y)
        x, y = np.randint(m, n), np.randint(m, n)
        while (x, y) in seen:
            x, y = np.randint(m, n), np.randint(m, n)


if __name__ == '__main__':
    rand_list = np.random.choice(1000, 1000,replace=False)
    # ACF_image()
    #
    # part_2d = bracket_1()
    # part_1d = bracket_2()
    vidcap = cv2.VideoCapture("cut_RealBarca.mp4")
    success = True

    pair_lst = [(rand_list[i-1],rand_list[i]) for i in range(1,len(rand_list))]
    ext_list = [(rand_list[i],rand_list[i-1]) for i in range(1,len(rand_list))]

    pair_lst.extend(ext_list)
    my_square = pair_lst



    matr_time = np.array([])
    # permut = list(product([250,275,555,551, 300, 350, 375, 400, 425, 450, 475, 500, 550, 600, 650, 675, 700, 750], repeat=2))
    # permut = list(product(my_square, repeat=2))
    print(len(pair_lst))
    count = 0

    matr_time = np.zeros((len(pair_lst), 2048))

    while success and count < 2048:

        success, image = vidcap.read()
        for i in range(len(pair_lst)):
            # print(i)
            matr_time[i, count] = image[pair_lst[i][0], pair_lst[i][1], 0]
        count += 1
    # image_array = io.imread("frames_orig_video/frame2200.png")[:, :, 0]
    # variance_list=np.var(image_array,axis=1)
    # image_array_rav = np.ravel(image_array)
    # variance_of_image = np.var(image_array)
    # print(variance_of_image,np.sum(image_array))
    mean_matr_time = np.mean(matr_time, axis=0)
    mean_by_mean = np.mean(mean_matr_time)
    ifft_matr = np.zeros(matr_time.shape)
    for i in range(matr_time.shape[0]):
        # print(len(matr_time[i, :]))
        ifft_matr[i, :] = ACF_by_periodogram(matr_time[i, :])
        # ifft_matr[i, :] = ifft_matr[i, :] - np.mean(matr_time[i, :]) ** 2
        # ifft_matr[i, :] /= np.max(ifft_matr[i, :])

    mean_ifft = np.mean(ifft_matr, axis=0)
    mean_ifft -= mean_by_mean ** 2
    mean_ifft /= np.max(mean_ifft)
    # lst10= image_array[0,:]
    # lst500 = image_array[300, :]
    # lst1000 = image_array[600, :]
    # lst900 = image_array[900, :]
    # print(len(lst10))
    # ifft10= ACF_by_periodogram(list_10)
    # ifft500 = ACF_by_periodogram(list_500)
    # ifft1000 = ACF_by_periodogram(list_1000)
    # ifft900 = ACF_by_periodogram(lst900)

    # ifft500= ACF_by_periodogram(list_500)
    # ifft1000 = ACF_by_periodogram(list_1000)
    # # for i in range(0,1001,300):
    #     plt.plot(ifft[i,:],label="string"+str(i))

    # print(ifft_matr[0,0])
    # cor_matr = ifft_matr/ifft_matr[0,0]

    # for i in range(0,1000,300):
    #     plt.plot(ifft_matr[i,:]/np.max(ifft_matr[i,:]),label="string " + str(i))

    plt.plot(mean_ifft[:200], label="Average ACF")
    # plt.plot(ifft1000, label="pixel[1000][1000]")
    # plt.plot(ifft900, label="string 900")
    plt.title("Road")
    plt.xlabel("Number of frame")
    plt.ylabel("ACF")

    all_mse = []
    params = []
    for p in np.arange(0.05, 1.01, 0.05):
        for betta in np.arange(0.007,0.011, 0.001):
            for alfa in np.arange(0.31,0.92,0.05):
                list_ACF = []
                for i in range(100):
                    list_ACF.append(calc_ACF(p, alfa, betta, i))

                params.append((p,betta,alfa))
                all_mse.append(mean_squared_error(mean_ifft[:30], list_ACF[:30]))

    print(min(all_mse),all_mse.index(min(all_mse)))
    need_params= params[all_mse.index(min(all_mse))]
    print(params[all_mse.index(min(all_mse))])

    list_ACF = []
    for i in range(100):
        list_ACF.append(calc_ACF(need_params[0], need_params[1], need_params[2], i))

    div_ACF_video = []
    div_ACF_model = []

    print(mean_ifft)

    for i in range(1, len(list_ACF)):
        div_ACF_video.append(mean_ifft[i] / mean_ifft[i - 1])
        div_ACF_model.append(list_ACF[i] / list_ACF[i - 1])

    from statistics import mean
    print(mean(div_ACF_video[40:100]))
    # print(sum(div_ACF_video[30:100]) / len(div_ACF_video[30:100]), div_ACF_video)
    print(list(mean_ifft[:30]))

    plt.title("RealBarca")
    plt.plot(list_ACF[:200], label="Model ACF")
    # plt.plot(acorr)
    plt.legend()
    plt.show()
    # pool = Pool()

    plt.title("B(k+1)/B(k). RealBarca. Volume of pixel=2000")
    plt.plot(div_ACF_video, label="Average of video")
    plt.plot(div_ACF_model, label="Model")
    plt.legend()
    plt.show()
    # Список параметров для вызова функции
    # parameters = [("Road.mp4", 0, 0), ("Road.mp4", 500, 500), ("Road.mp4", 1000, 1000)]
    #
    # results = pool.starmap(read_video, parameters)
    # # print(p.map(read_video, ["Road.mp4",0,0]))
    # print(results)
    # read_video("Road.mp4",0,0)
    # read_video("Road.mp4",500,500)
