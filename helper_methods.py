import numpy as np
import re
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
import csv

# from reedsolomon import Nbit

size_quadr = 16


def sort_spis(sp):
    spp = []
    spb = []
    res = []
    for i in sp:
        spp.append("".join(re.findall(r'\d', i)))
        spb.append("result")
    result = [int(item) for item in spp]
    result.sort()

    result1 = [str(item) for item in result]
    for k in range(len(sp)):
        res.append(spb[k] + result1[k] + ".png")
    return res


def img2bin(img):
    k = 0

    our_avg = np.mean(img)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            tmp = img[i, j]

            if tmp > our_avg:
                img[i, j] = 255
            else:
                img[i, j] = 0

            k += 1
    return img


def big2small(st_qr):
    qr = np.zeros((89, 89))

    for i in range(0, 1424, size_quadr):
        for j in range(0, 1424, size_quadr):
            qr[int(i / size_quadr), int(j / size_quadr)] = np.mean(st_qr[i:i + size_quadr, j:j + size_quadr])

    return qr


def small2big(sm_qr):
    qr = np.zeros((1424, 1424))

    for i in range(0, 89):
        for j in range(0, 89):
            tmp = sm_qr[i, j]
            qr[i * 16:i * 16 + 16, j * 16:j * 16 + 16].fill(tmp)

    return qr


def img2bin(img):
    k = 0

    our_avg = np.mean(img)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            tmp = img[i, j]

            if tmp > our_avg:
                img[i, j] = 255
            else:
                img[i, j] = 0

            k += 1
    return img


def disp(path):
    cnt = 0
    arr = np.array([])

    total_count = len(list(Path(path).iterdir()))

    list_diff = []
    while cnt < total_count:
        tmp = np.copy(arr)
        arr = io.imread(path + '/frame' + str(cnt) + ".png").astype(float)
        if cnt == 0:
            list_diff.append(0)

        else:
            diff_img = np.abs(arr - tmp)

            list_diff.append(np.mean(diff_img))
        if cnt % 100 == 0:
            print(cnt)
        cnt += 1

    max_val_list = max(list_diff)
    list_diff = list_diff / max_val_list / 2

    with open('RB_disp.csv', 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerow(list_diff)

    plt.plot(list_diff)
    plt.show()

    avg = sum(list_diff) / len(list_diff)

    upd_start = []
    for i in range(len(list_diff)):
        if abs((list_diff[i])) > (4 * avg):
            upd_start.append(i)

    return list_diff


def csv2list(path_csv):
    with open(path_csv, newline='') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        l = []
        for row in readCSV:
            l.append(row)
    need_list = l[0]
    result = [float(item) for item in need_list]
    return result


def bit_voting(image, count):
    vec_np = np.reshape(image, image.shape[0] * image.shape[1])[0:count]
    vec_np[vec_np == 255] = 1
    vec_vot = np.zeros(int(count / 7))
    for i in range(len(vec_vot)):
        for j in range(0, 7):
            vec_vot[i] += vec_np[j * len(vec_vot) + i]
    vec_vot[vec_vot < (7 / 2)] = 0
    vec_vot[vec_vot > (7 / 2)] = 255

    long_vec = np.zeros(vec_np.shape)
    for i in range(0, 7):
        tmp = i * len(vec_vot)
        tmp2 = (i + 1) * len(vec_vot)
        long_vec[tmp:tmp2] = vec_vot
    for i in range(image.shape[0] * image.shape[1] - count):
        vec_np = np.append(vec_np, 0)
    matr_aft_vot = np.reshape(vec_np, (image.shape[0], image.shape[1]))
    matr_aft_vot[matr_aft_vot == 1] = 255
    return matr_aft_vot


# bit_voting(io.imread(r"D:\dk\university\nirs\extract/wm_after_2_smooth_bin/result" + str(456) + ".png"), 7112)
# print(disp("C:/Users/user/PycharmProjects/phase_wm/frames_orig_video"))
csv2list('RB_disp.csv')
