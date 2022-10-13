import numpy as np
import math
from skimage import io
import csv
from skimage.util import random_noise
from matplotlib import pyplot as plt
from skimage.exposure import histogram
import re
import cv2, os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

size_quadr = 16


def sort_img_list(img_list):
    list_nmb = []
    list_wrd = []
    res = []
    for i in img_list:
        list_nmb.append("".join(re.findall(r'\d', i)))
        list_wrd.append("result")
    result = [int(item) for item in list_nmb]
    result.sort()

    result1 = [str(item) for item in result]
    for k in range(len(img_list)):
        res.append(list_wrd[k] + result1[k] + ".png")
    return res


def big2small(st_qr):
    qr = np.zeros((65, 65, 3))

    for i in range(0, 1040, size_quadr):
        for j in range(0, 1040, size_quadr):
            qr[int(i / size_quadr), int(j / size_quadr)] = np.mean(st_qr[i:i + size_quadr, j:j + size_quadr])

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


def randomize_tiles_shuffle_blocks(a, M, N):
    m, n = a.shape
    b = a.reshape(m // M, M, n // N, N).swapaxes(1, 2).reshape(-1, M * N)
    np.random.seed(42)
    np.random.shuffle(b)
    return b.reshape(m // M, n // N, M, N).swapaxes(1, 2).reshape(a.shape)


def read_video(path):
    vidcap = cv2.VideoCapture(path)
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if (success == True):
            cv2.imwrite(r"frames_orig\frame%d.png" % count, image)

        print("записан кадр", count)

        if cv2.waitKey(10) == 27:
            break
        count += 1


def exp_smooth(path, alf):
    cnt = 0
    arr_copy = np.asarray([])
    while cnt < count:
        arr = io.imread(path + str(cnt) + ".png")

        img_1_step = arr_copy
        if cnt == 0:
            arr_copy = arr.copy()
            arr_1_step = np.zeros((1080, 1920))
        else:
            arr_copy = np.float32(img_1_step) * alf + np.float32(arr) * (1 - alf)

        arr_copy[arr_copy > 255] = 255
        arr_copy[arr_copy < 0] = 0

        print("tmp kadr", cnt)
        cnt += 1
    return arr_copy

def disp(path):
    cnt = 0
    arr = np.array([])

    list_diff = []
    while cnt < 3000:
        tmp = np.copy(arr)
        arr = io.imread(path + str(cnt) + ".png").astype(float)
        if cnt == 0:
            list_diff.append(0)

        else:
            diff_img = np.abs(arr-tmp)
            print(np.mean(diff_img)," frame ", cnt)
            list_diff.append(np.mean(diff_img))
        cnt += 1

    avg = sum(list_diff) / len(list_diff)
    print(avg)
    upd_start = []
    for i in range(len(list_diff)):
        if abs((list_diff[i])) > (4*avg):
            upd_start.append(i)

    print("frame with change scene", upd_start)
    return upd_start


def embed(my_i, tt, count):
    cnt = 0
    PATH_IMG = r'some_qr.png'
    fi = math.pi / 2 / 255

    arr1 = io.imread(PATH_IMG)

    # arr1=cv2.cvtColor(arr1,cv2.COLOR_RGB2YCrCb)

    pict = np.zeros((1080, 1920, 3))
    # встраивание QR-кода в пустой контейнер большого размера
    pict[20:1060, 432:1472, 0] = arr1

    list0 = []
    list1 = []
    for i in range(4, 1076, size_quadr):
        for j in range(0, 1920, size_quadr):

            if ([i, j] not in list1) and (pict[i, j, 0] == 255):
                list1.append([int(i / size_quadr), int(j / size_quadr)])
            else:
                list0.append([int(i / size_quadr), int(j / size_quadr)])
    arr = np.zeros((1080, 1920, 3))
    arr[4:1076, :, 0] = randomize_tiles_shuffle_blocks(pict[4:1076, :, 0], size_quadr, size_quadr)

    list0_new = []
    list1_new = []

    for i in range(4, 1076, size_quadr):
        for j in range(0, 1920, size_quadr):
            if ([i, j] not in list1_new) and (arr[i, j, 0] == 255):
                list1_new.append([int(i / size_quadr), int(j / size_quadr)])
            else:
                list0_new.append([int(i / size_quadr), int(j / size_quadr)])

    while cnt < count:
        img = (io.imread("frames_orig_video/frame%d.png" % cnt))

        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

        temp = np.float32(fi) * np.float32(arr)
        wm = np.asarray((my_i * np.sin(cnt * tt + temp)))
        if my_i == 1:
            wm[wm > 0] = 1
            wm[wm < 0] = -1

        ycrcb[:, :, 0] = np.float32(ycrcb[:, :, 0] + wm[:, :, 0])
        ycrcb[ycrcb > 255] = 255
        ycrcb[ycrcb < 0] = 0

        ycrcb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        img = Image.fromarray(ycrcb.astype('uint8'))

        print("обработан кадр", cnt)
        img.convert('RGB').save(r"frames_after_emb\result" + str(cnt) + ".png")

        cnt += 1

    return list0, list1, list0_new, list1_new


def extract(alf, tt, rand_fr):

    with open('list0.csv', 'r') as f:
        list0 = list(csv.reader(f))[0]

    list0 = [eval(i) for i in list0]

    with open('list1.csv', 'r') as f:
        list1 = list(csv.reader(f))[0]

    list1 = [eval(i) for i in list1]

    with open('list0_new.csv', 'r') as f:
        list0_new = list(csv.reader(f))[0]

    list0_new = [eval(i) for i in list0_new]

    with open('list1_new.csv', 'r') as f:
        list1_new = list(csv.reader(f))[0]

    list1_new = [eval(i) for i in list1_new]

    PATH_VIDEO = r'frames_after_emb\RB_codH264.mp4'
    vidcap = cv2.VideoCapture(PATH_VIDEO)
    vidcap.open(PATH_VIDEO)

    alf0 = 0.96
    betta = 0.999

    count = rand_fr

    success = True
    while success:
        success, image = vidcap.read()
        if success:
            print('Read a new frame:%d ' % count, success)
            cv2.imwrite(r'extract\frame%d.png' % count, image)

        count += 1

    count = 3000

    # change_sc=disp("extract/frame")
    # change_sc.insert(0, 0)
    # change_sc.append(count)

    with open('change_sc.csv', 'r') as f:
        change_sc = list(csv.reader(f))[0]

    change_sc = [eval(i) for i in change_sc]

    # первичное сглаживание
    # f1 = exp_smooth("extract/frame", alf)
    img_smooth = np.load("exp_smooth_img.npy")

    cnt = rand_fr

    img_2_step = np.asarray([])
    img_1_step = np.asarray([])
    img_chn = img_1_step.copy()
    # вычитание усреднённого
    for scene in range(1, len(change_sc)):
        cnt = int(change_sc[scene - 1])
        while cnt < int(change_sc[scene]):
            arr = io.imread(r"extract\frame" + str(cnt) + ".png")  # убрать np.asarray

            img_aft_smooth = np.asarray([])

            img_aft_smooth = np.where(arr < img_smooth, 0, arr - img_smooth)

            print("diff", cnt)
            # извлечение ЦВЗ

            arr = img_aft_smooth
            a = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)

            img_2_step = img_1_step
            img_1_step = img_chn

            if cnt == change_sc[scene-1]:
                img_chn = a[:, :, 0]

                img_1_step = np.ones((1080, 1920))

            else:
                if cnt == change_sc[scene-1] + 1:
                    img_chn = 2 * betta * math.cos(tt) * np.float32(img_1_step) + np.float32(a[:, :, 0])

                else:
                    img_chn = 2 * betta * math.cos(tt) * np.float32(img_1_step) - (betta ** 2) * np.float32(
                        img_2_step) + np.float32(a[:, :, 0])

            yc = np.float32(img_chn) - betta * math.cos(tt) * np.float32(img_1_step)
            ys = betta * math.sin(tt) * np.float32(img_1_step)
            c = math.cos(tt * cnt) * np.float32(yc) + math.sin(tt * cnt) * np.float32(ys)
            s = math.cos(tt * cnt) * np.float32(ys) - math.sin(tt * cnt) * np.float32(yc)

            fi = np.where(c < 0, np.arctan((s / c)) + np.pi,
                          np.where(s >= 0, np.arctan((s / c)), np.arctan((s / c)) + 2 * np.pi))
            fi = np.where(fi < -np.pi / 4, fi + 2 * np.pi, fi)
            fi = np.where(fi > 9 * np.pi / 4, fi - 2 * np.pi, fi)

            wm = 255 * fi / 2 / math.pi
            # wm[wm>255]=255
            # wm[wm<0]=0

            img_aft_smooth = wm
            # a1 = cv2.cvtColor(a1, cv2.COLOR_YCrCb2RGB)
            img = Image.fromarray(img_aft_smooth.astype('uint8'))

            img.save(r'extract/wm/result' + str(cnt) + '.png')
            print('made', cnt)

            l_kadr = io.imread(r'extract/wm/result' + str(cnt) + '.png')

            fi = np.copy(l_kadr)
            fi_tmp = np.copy(fi)
            fi = (l_kadr * np.pi * 2) / 255

            dis = []

            coord1 = np.copy(fi)

            coord2 = np.copy(fi)
            coord1 = np.where(fi < np.pi, (fi / np.pi * 2 - 1) * (-1),
                              np.where(fi > np.pi, ((fi - np.pi) / np.pi * 2 - 1), fi))
            coord2 = np.where(fi < np.pi / 2, (fi / np.pi / 2),
                              np.where(fi > 3 * np.pi / 2, ((fi - 1.5 * np.pi) / np.pi * 2) - 1,
                                       ((fi - 0.5 * np.pi) * 2 / np.pi - 1) * (-1)))
            hist, bin_centers = histogram(coord1, normalize=False)
            hist2, bin_centers2 = histogram(coord2, normalize=False)

            prob = []
            prob2 = []
            mx_list = np.arange(bin_centers[0], bin_centers[-1], bin_centers[1] - bin_centers[0])
            for i in range(len(hist)):
                prob.append(hist[i] / sum(hist))
            mo = moment = 0
            for i in range(len(hist)):
                mo += bin_centers[i] * prob[i]
            for mx in mx_list:
                dis.append(abs(mo - mx))

            pr1 = 0
            pr2 = 0
            for i in range(len(dis)):
                if min(dis) == dis[i]:
                    pr1 = bin_centers[i]

            dis2 = []
            mx_list2 = np.arange(bin_centers2[0], bin_centers2[-1], bin_centers2[1] - bin_centers2[0])
            for i in range(len(hist2)):
                prob2.append(hist2[i] / sum(hist2))
            mo = 0
            for i in range(len(hist2)):
                mo += bin_centers2[i] * prob2[i]
            for mx in mx_list2:
                dis2.append(abs(mo - mx))

            x = min(dis2)

            for i in range(len(dis2)):
                if x == dis2[i]:
                    pr2 = bin_centers2[i]

            moment = np.where(pr1 < 0, np.arctan((pr2 / pr1)) + np.pi,
                              np.where(pr2 >= 0, np.arctan((pr2 / pr1)), np.arctan((pr2 / pr1)) + 2 * np.pi))

            if np.pi / 4 <= moment <= np.pi * 2 - np.pi / 4:
                fi_tmp = fi - moment + 0.5 * np.pi * 0.5
                fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
                fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)
            elif moment > np.pi * 2 - np.pi / 4:
                fi = np.where(fi < np.pi / 4, fi + 2 * np.pi, fi)
                fi_tmp = fi - moment + 0.5 * np.pi * 0.5
                fi_tmp = np.where(fi_tmp < -np.pi, fi_tmp + 2 * np.pi, fi_tmp)
                fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)

            elif moment < np.pi / 4:

                fi_tmp = fi - 2 * np.pi - moment + 0.5 * np.pi * 0.5
                fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
                fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)

            fi_tmp[fi_tmp < 0] = 0
            fi_tmp[fi_tmp > np.pi] = np.pi
            l_kadr = fi_tmp * 255 / np.pi

            cp = l_kadr.copy()
            # imgc = Image.fromarray(cp.astype('uint8'))
            # imgc.save(r"extract\after_normal_phas\result" + str(cnt) + ".png")
            pict = np.zeros((1080, 1920))

            for i in range(len(list1)):
                pict[4 + list1[i][0] * size_quadr:4 + (list1[i][0]) * size_quadr + size_quadr,
                list1[i][1] * size_quadr:(list1[i][1]) * size_quadr + size_quadr] = \
                    cp[(list1_new[i][0]) * size_quadr:list1_new[i][0] * size_quadr + size_quadr,
                    list1_new[i][1] * size_quadr: list1_new[i][1] * size_quadr + size_quadr]
            for i in range(len(list0)):
                pict[4 + list0[i][0] * size_quadr:4 + (list0[i][0]) * size_quadr + size_quadr,
                list0[i][1] * size_quadr:list0[i][1] * size_quadr + size_quadr] = \
                    cp[list0_new[i][0] * size_quadr:list0_new[i][0] * size_quadr + size_quadr,
                    list0_new[i][1] * size_quadr: list0_new[i][1] * size_quadr + size_quadr]

            c_qr = pict[20:1060, 432:1472]

            small_new_qr = np.zeros((65, 65))

            small_qr = big2small(c_qr)

            imgc = Image.fromarray(small_qr.astype('uint8'))
            imgc.save(r"extract\after_normal_phas_bin\comparing" + str(cnt) + ".png")

            small_new_qr[0:32, 0:32] = img2bin(small_qr[0:32, 0:32, 0])
            small_new_qr[32:65, 0:32] = img2bin(small_qr[32:65, 0:32, 0])
            small_new_qr[0:32, 32:65] = img2bin(small_qr[0:32, 32:65, 0])
            small_new_qr[32:65, 32:65] = img2bin(small_qr[32:65, 32:65, 0])

            mgc = Image.fromarray(small_new_qr.astype('uint8'))
            mgc.save(r"extract\after_normal_phas_local_bin/compar" + str(cnt) + ".png")
            if (cnt % 200) == 199:
                match_perc_bef_smooth.append(compare(r"extract\after_normal_phas_local_bin/compar" + str(cnt) + ".png"))

                print("mod 200 ", cnt)
            cnt += 1

        print("before ",match_perc_bef_smooth)
        # повторное сглаживание
        count = 3000
        cnt = (change_sc[scene-1])
        g2 = np.asarray([])
        f = np.copy(g2)
        alf2 = 0.95
        while cnt < (change_sc[scene]):

            arr = io.imread(r"extract\after_normal_phas_bin\comparing" + str(cnt) + ".png")
            # g2 - y(n-1)
            y_step_1 = f
            if cnt == change_sc[scene-1]:
                f = arr.copy()
                f_step_1 = np.zeros((65, 65))
            else:
                # y(n)=alfa*y(n-1)+x(n)*(1-alfa)
                f = y_step_1 * alf2 + arr * (1 - alf2)
                f[f > 255] = 255

            img = Image.fromarray(f.astype('uint8'))

            print("avg kadr", cnt)
            img.save(r"extract/wm_after_2_smooth/result" + str(cnt) + ".png")
            cnt += 1

        count = 3000
        cnt = (change_sc[scene-1])
        while cnt < (change_sc[scene]):
            if (cnt % 200) == 199:
                c_qr = io.imread(r"extract\wm_after_2_smooth\result" + str(cnt) + ".png")
                c_qr = img2bin(c_qr[:, :, 0])

                img1 = Image.fromarray(c_qr.astype('uint8'))
                img1.save(r"extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png")
                match_perc_aft_smooth.append(compare(r"extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png"))

            cnt += 1

    return r"extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png"


def generate_video():
    image_folder = r'C:\Users\user\PycharmProjects\phase_wm\frames_after_emb'  # make sure to use your folder
    video_name = 'RB_codH264.mp4'
    os.chdir(r"C:\Users\user\PycharmProjects\phase_wm\frames_after_emb")

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".png")]
    sort_name_img = sort_img_list(images)

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

    cnt = 0

    for image in sort_name_img:
        print("video is writing", cnt)
        video.write(cv2.imread(os.path.join(image_folder, image)))
        cnt += 1

    cv2.destroyAllWindows()
    video.release()


def compare(p):  # сравнивание извлечённого QR с исходным
    orig_qr = io.imread(r'some_qr.png')
    orig_qr = np.where(orig_qr > 129, 255, 0)
    small_qr = big2small(orig_qr)
    compare_matrix = np.zeros((65, 65))
    extract_qr = io.imread(p)
    extract_qr = np.where(extract_qr > 129, 255, 0)

    k = 0
    mas_avg = []
    for i in range(0, 65):
        for j in range(0, 65):
            if np.mean(small_qr[i, j]) == np.mean(extract_qr[i, j]):
                compare_matrix[int(i), int(j)] = 1
                mas_avg.append(1)
            else:
                compare_matrix[i, j] = 0
                mas_avg.append(0)

    for i in mas_avg:
        if i == 1:
            k += 1
    return k / len(mas_avg)


i = 1
percantage_for_optim = []
alfa = 0.93
tetta = 2.8
square_size = 4
for_fi = 6
# dispr=1

match_perc_bef_smooth = []
match_perc_aft_smooth = []

PATH_VIDEO = r'RealBarca.mp4'

# read_video_cadr(PATH_VIDEO)

rand_k = 0
count = 3000

while tetta < 2.82:
    # list0, list1, list0_new, list1_new = embed(i, tetta, count)
    # print("number's shuffle squares", list0),
    # print(list1)
    # print(list0_new)
    # print(list1_new)
    # generate_video()

    sp = []
    a = extract(alfa, tetta, rand_k)

    percantage_for_optim.append(compare("extract/wm_after_2_smooth_bin/result2999.png"))
    print("current percent", match_perc_bef_smooth)
    print("current percent", match_perc_aft_smooth)
    # i+=1
    tetta += 1.3

print(percantage_for_optim)

fig = plt.figure()
ax = fig.add_subplot(111, label="1")
plt.plot([i for i in np.arange(199, 3000, 0.02)], percantage_for_optim)
plt.plot([i for i in np.arange(199, 3000, 200)], match_perc_bef_smooth)
#plt.plot([i for i in np.arange(1, 5.1, 1)], match_perc_aft_smooth)

plt.show()
