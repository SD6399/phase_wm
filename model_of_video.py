import math
import matplotlib.pyplot as plt
import numpy as np
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


def calc_ACF(coord_x, coord_y, numb_frame):
    p = 0.01
    alf = 0.0016
    betta = 0.01
    r = math.sqrt(coord_x * coord_x + coord_y * coord_y)

    R = (p * math.e ** (-alf * r) + (1 - p) * math.e ** (-betta * r)) * (
            p * math.e ** (-alf * numb_frame) + (1 - p) * math.e ** (-betta * numb_frame))

    return R


if __name__ == '__main__':

    # ACF_image()
    #
    # part_2d = bracket_1()
    # part_1d = bracket_2()
    vidcap = cv2.VideoCapture("Road.mp4")
    success = True
    list_00 = []
    count = 0
    while success and count < 10:
        success, image = vidcap.read()
        if success:
            list_00.append(image[10, 10, 0])
        count += 1
    image_array = io.imread("frames_orig_video/frame0.png")[:, :, 0]
    variance_list=np.var(image_array,axis=1)
    image_array_rav = np.ravel(image_array)
    fft = np.fft.fft2(image_array)
    fft[0][0]=0
    abs_fft = np.power(np.abs(fft), 2)
    ifft = np.fft.ifft2(abs_fft)
    ifft = np.abs(ifft)/len(fft)
    print(fft.shape, fft[0:50])
    for i in range(0,1001,300):
        plt.plot(ifft[i,:],label="string"+str(i))
    plt.title("Road")
    plt.xlabel("Number of pixel")
    plt.ylabel("ACF")
    plt.legend()
    plt.show()
    acorr = sm.tsa.acf(list_00, nlags=100)

    list_ACF = []
    for i in range(1000):
        list_ACF.append(calc_ACF(10, 10, i))
    print(list_ACF)

    plt.plot(list_ACF[0:10])
    plt.plot(acorr)
    plt.show()
    pool = Pool()

    # Список параметров для вызова функции
    # parameters = [("Road.mp4", 0, 0), ("Road.mp4", 500, 500), ("Road.mp4", 1000, 1000)]
    #
    # results = pool.starmap(read_video, parameters)
    # # print(p.map(read_video, ["Road.mp4",0,0]))
    # print(results)
    # read_video("Road.mp4",0,0)
    # read_video("Road.mp4",500,500)
