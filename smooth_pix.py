import math
from skimage import io
from skimage.util import random_noise
from scipy import interpolate
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from statistics import mean
from skimage.exposure import histogram, equalize_hist
import re
import csv
import cv2, os
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

size_quadr = 16


def sort_spis(sp):
    spp = []
    spb = []
    res = []
    for i in (sp):
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
    qr = np.zeros((65, 65, 3))

    for i in range(0, 1040, size_quadr):
        for j in range(0, 1040, size_quadr):
            qr[int(i / size_quadr), int(j / size_quadr)] = np.mean(st_qr[i:i + size_quadr, j:j + size_quadr])

    return qr


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
            diff_img = np.abs(arr - tmp)
            print(np.mean(diff_img), " frame ", cnt)
            list_diff.append(np.mean(diff_img))
        cnt += 1

    avg = sum(list_diff) / len(list_diff)
    print(avg)
    upd_start = []
    for i in range(len(list_diff)):
        if abs((list_diff[i])) > (4 * avg):
            upd_start.append(i)

    print("frame with change scene", upd_start)
    return list_diff


def disp_pix(path,coord_x,coord_y):
    cnt = 1
    arr = np.array([])

    list_diff = []
    while cnt < 3000:
        tmp = np.copy(arr[coord_x,coord_y])
        arr = io.imread(path + str(cnt) + ".png").astype(float)[coord_x,coord_y]

        diff_pix = np.abs(arr - tmp)
        print(np.mean(diff_pix), " frame ", cnt)
        list_diff.append(np.mean(diff_pix))
        cnt += 1

    avg = sum(list_diff) / len(list_diff)
    print(avg)
    upd_start = []
    for i in range(len(list_diff)):
        if abs((list_diff[i])) > (4 * avg):
            upd_start.append(i)

    print("frame with change scene", upd_start)
    return list_diff


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


def white_black():
    qr = io.imread(r'some_qr.png')
    mass = []
    for i in range(0, 1040, 16):
        for j in range(0, 1040, 16):
            if np.mean(qr[i:i + 16, j:j + 16]) == 255:
                mass.append(1)
            else:
                mass.append(0)

    return (mass.count(1)), (mass.count(0))


def embed(my_i, tt, count):
    cnt = 0
    PATH_IMG = r'C:\Users\user\PycharmProjects\phase_wm\some_qr_M.png'
    fi = math.pi / 2 / 255

    arr1 = cv2.imread(PATH_IMG)
    arr1 = cv2.cvtColor(arr1, cv2.COLOR_RGB2YCrCb)

    while cnt < count:
        imgg = (io.imread(r"C:\Users\user\PycharmProjects\phase_wm\frames_orig_video/frame%d.png" % cnt))

        a = cv2.cvtColor(imgg, cv2.COLOR_RGB2YCrCb)

        temp = np.float32(fi) * np.float32(arr1)
        wm = np.asarray((my_i * np.sin(cnt * tt + temp)))
        if my_i == 1:
            wm[wm > 0] = 1
            wm[wm < 0] = -1

        tmp = np.float32(a[20:1060, 440:1480] + wm)
        a[a > 255] = 255
        a[a < 0] = 0
        a[20:1060, 440:1480, 0] = tmp[:, :, 0]

        tmp = cv2.cvtColor(a, cv2.COLOR_YCrCb2RGB)
        img = Image.fromarray(tmp.astype('uint8'))

        print("обработан кадр", cnt)
        img.convert('RGB').save(r"frames_after_emb\result" + str(cnt) + ".png")

        cnt += 1


stop_kadr1 = []
stop_kadr2 = []
stop_kadr3 = []
stop_kadr4 = []
stop_kadr5 = []

def disp_pix(coord_x,coord_y,path,kef_avg):
    cnt=0
    count=3000
    list_diff=[]
    while cnt < count:

        if cnt>0:
            tmp = np.copy(arr)
        arr = io.imread(path[cnt])[coord_x, coord_y,0]

        if cnt == 0:
            list_diff.append(0)

        else:
            diff_img = np.abs(int(arr) - int(tmp))
            #print(diff_img, " frame ", cnt)
            list_diff.append(np.mean(diff_img))

        cnt+=1

    mean_diff=(mean(list_diff))
    #for i in list_diff:
    list_big_diap=[]
    length=1
    for i in range(len(list_diff)):
        if abs(list_diff[i] - list_diff[i-1]) > kef_avg*mean_diff:
           list_big_diap.append(i)


    print(len(list_big_diap))
    return list_big_diap



def extract(coord_x,coord_y,alf, tt, rand_fr):
    PATH_VIDEO = r'C:\Users\user\PycharmProjects\phase_wm\frames_after_emb\RB_codH264.mp4'
    vidcap = cv2.VideoCapture(PATH_VIDEO)
    vidcap.open(PATH_VIDEO)

    alf0 = 0.96
    betta = 0.999
    alf2 = 0.95

    count = rand_fr
    zz=[]
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            print('Read a new frame:%d ' % count, success)
            cv2.imwrite(r'C:\Users\user\PycharmProjects\phase_wm\extract\frame%d.png' % count, image[coord_x,coord_y])
            zz.append(image[coord_x,coord_y])

        count += 1

    count = 3000

    cnt = rand_fr
    g = np.asarray([])
    f = g.copy()
    f1 = f.copy()
    d = g.copy()
    d1=d.copy()

    # первичное сглаживание
    disp_list=[]
    disp_list = disp_pix(coord_x, coord_y, zz, 4)
    disp_list.insert(0, 0)
    disp_list.append(3000)

    for scene in range(1, len(change_sc)):
        cnt = change_sc[scene - 1]
        while cnt < change_sc[scene]:
            arr = io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract/frame" + str(cnt) + ".png")
            a = arr
            #g1=d1 # !!!!!!!!!!!!!
            d1 = f1
            if cnt == change_sc[scene-1]:
                f1 = a.copy()
                d1 = np.zeros(1)
            #elif cnt == change_sc[scene-1] + 1:
            else:
                f1 = np.float32(d1) * alf + np.float32(a) * (1 - alf)
            # else:
            #     f1 = (1-alf)*(1-alf)*a+(1-alf)*alf*d1+alf*g1

            f1[f1 > 255] = 255
            f1[f1 < 0] = 0
            img = Image.fromarray(f1.astype('uint8'))
            img.save(r'C:\Users\user\PycharmProjects\phase_wm\extract\first_smooth/result' + str(cnt) + '.png')
            print("tmp kadr", cnt)
            cnt += 1

    cnt = rand_fr

    # вычитание усреднённого
    while cnt < count:

        arr = io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract\frame" + str(cnt) + ".png")

        a = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
        a1 = np.asarray([])
        f1=np.float32(io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract\first_smooth\result" + str(cnt) + ".png"))
        #f1=np.float32(f1)
        f1 = cv2.cvtColor(f1, cv2.COLOR_RGB2YCrCb)
        a1 = np.where(a < f1, 0, a - f1)

        print("diff", cnt)

        # извлечение ЦВЗ
        arr = a1
        a = arr

        g = d
        d = f

        if cnt == rand_fr:
            f = a[:, :, 0]
            d = f.copy()
            d = np.ones(1)

        else:
            if cnt == rand_fr + 1:
                f = 2 * betta * math.cos(tt) * np.float32(d) + np.float32(a[:, :, 0])

            else:
                f = 2 * betta * math.cos(tt) * np.float32(d) - (betta ** 2) * np.float32(g) + np.float32(a[:, :, 0])

        yc = np.float32(f) - betta * math.cos(tt) * np.float32(d)
        ys = betta * math.sin(tt) * np.float32(d)
        c = math.cos(tt * cnt) * np.float32(yc) + math.sin(tt * cnt) * np.float32(ys)
        s = math.cos(tt * cnt) * np.float32(ys) - math.sin(tt * cnt) * np.float32(yc)

        fi = np.where(c < 0, np.arctan((s / c)) + np.pi,
                      np.where(s >= 0, np.arctan((s / c)), np.arctan((s / c)) + 2 * np.pi))
        fi = np.where(fi < -np.pi / 4, fi + 2 * np.pi, fi)
        fi = np.where(fi > 9 * np.pi / 4, fi - 2 * np.pi, fi)
        wm = 255 * fi / 2 / math.pi

        # wm[wm>255]=255
        # wm[wm<0]=0

        a1 = wm
        # a1 = cv2.cvtColor(a1, cv2.COLOR_YCrCb2RGB)

        # plt.show()

        # привдение к рабочему диапазону

        l_kadr = a1

        fi = np.copy(l_kadr)
        fi_tmp = np.copy(fi)
        fi = (l_kadr * np.pi * 2) / 255

        dis = []
        koord1 = np.copy(fi)

        koord2 = np.copy(fi)
        koord1 = np.where(fi < np.pi, (fi / np.pi * 2 - 1) * (-1),
                          np.where(fi > np.pi, ((fi - np.pi) / np.pi * 2 - 1), fi))
        koord2 = np.where(fi < np.pi / 2, (fi / np.pi / 2),
                          np.where(fi > 3 * np.pi / 2, ((fi - 1.5 * np.pi) / np.pi * 2) - 1,
                                   ((fi - 0.5 * np.pi) * 2 / np.pi - 1) * (-1)))
        hist, bin_centers = histogram(koord1, normalize=False)
        hist2, bin_centers2 = histogram(koord2, normalize=False)

        ver = []
        ver2 = []
        mx_sp = np.arange(bin_centers[0], bin_centers[-1], bin_centers[1] - bin_centers[0])
        for i in range(len(hist)):
            ver.append(hist[i] / sum(hist))
        mo = moment = 0
        for i in range(len(hist)):
            mo += bin_centers[i] * ver[i]
        for mx in mx_sp:
            dis.append(abs(mo - mx))

        pr1 = 0
        pr2 = 0
        for i in range(len(dis)):
            if min(dis) == dis[i]:
                pr1 = (bin_centers[i])

        mx_sp2 = np.array([])
        dis2 = []
        mx_sp2 = np.arange(bin_centers2[0], bin_centers2[-1], bin_centers2[1] - bin_centers2[0])
        for i in range(len(hist2)):
            ver2.append(hist2[i] / sum(hist2))
        mo = 0
        for i in range(len(hist2)):
            mo += bin_centers2[i] * ver2[i]
        for mx in mx_sp2:
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

        print(my_exit)
        fi_tmp[fi_tmp < 0] = 0
        fi_tmp[fi_tmp > np.pi] = np.pi
        l_kadr = fi_tmp * 255 / np.pi

        cnt += 1

    print(l_kadr)
    print(coord_x,coord_y)

    return l_kadr


def generate_video():
    image_folder = r'C:\Users\user\PycharmProjects\phase_wm\frames_after_emb'  # make sure to use your folder
    video_name = 'RB_codH264.mp4'
    os.chdir(r"C:\Users\user\PycharmProjects\phase_wm\frames_after_emb")

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".png")]
    sort_name_img = sort_spis(images)

    print(sort_name_img)

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
    video.release()  # releasing the video generated


def compare(path):  # сравнивание извлечённого QR с исходным
    orig_qr = io.imread(r'C:\Users\user\PycharmProjects\phase_wm\some_qr_M.png')
    orig_qr = np.where(orig_qr > 127, 255, 0)
    small_qr = big2small(orig_qr)
    sr_matr = np.zeros((1040, 1040, 3))
    myqr = io.imread(path)
    myqr = np.where(myqr > 127, 255, 0)

    k = 0
    mas_avg = []
    for i in range(0, 65):
        for j in range(0, 65):

            if np.mean(small_qr[i, j]) == np.mean(myqr[i, j]):
                sr_matr[i, j] = 1
                mas_avg.append(1)
            else:
                sr_matr[i, j] = 0
                mas_avg.append(0)

    for i in mas_avg:
        if i == 1:
            k += 1
    return k / len(mas_avg)


def diff_pix_between_neugb(qr1, qr2):
    k = 0
    mas_avg = []
    for i in range(0, 65):
        for j in range(0, 65):

            if qr1[i, j] == qr2[i, j]:
                mas_avg.append(1)
            else:
                mas_avg.append(0)

    for i in mas_avg:
        if i == 0:
            k += 1
    return k


i = 1
my_exit = []
my_exit1 = []
my_exit2 = []
alfa = 0.91
tetta = 0.3
squ_size = 4
for_fi = 6
# dispr=1

# графики-сравнения по различныи параметрам

PATH_VIDEO = r'RealBarca.mp4'

with open('change_sc.csv', 'r') as f:
    change_sc = list(csv.reader(f))[0]

change_sc = [eval(i) for i in change_sc]

#read_video(PATH_VIDEO)

rand_k = 0
count = 3000


# disp_list=disp(r"C:\Users\user\PycharmProjects\phase_wm\frames_orig_video\frame")
# print(np.array(disp_list).argsort()[::-1][:100])
# print(np.array(sorted(disp_list)[::-1][:100]))


hm_list=[]
while alfa < 0.92:
    sp = []
    my_matrix=np.zeros((1040,1040))
    for i in range(1040):
        for j in range(1040):
            my_matrix[i,j] = extract(i,j,alfa, tetta, rand_k)

    print("all")
    print(my_matrix)
    hand_made= [0,118,404,414,524,1002,1391,1492,1972,2393,2466,2999]
    exit_list=[]
    res = np.zeros((65, 65))
    res_bin = np.zeros((65, 65))
    for i in range(1, len(hand_made)):
        print(hand_made[i])
        tnp = io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract/wm_after_2_smooth/result" + str(
            hand_made[i]-1 ) + ".png")
        res[tnp >= np.mean(tnp)] += (hand_made[i] - hand_made[i - 1])
        res[tnp < np.mean(tnp)] -= (hand_made[i] - hand_made[i - 1])
        # res2=img2bin(tnp)
        img = Image.fromarray(res.astype('uint8'))
        img.save(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas/sumframe" + str(i) + ".png")

        # print(compare(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas/sumframe" +str(i)+ ".png"), change_sc[i])
        res_bin[res >= 0] = 255
        res_bin[res < 0] = 0

        img = Image.fromarray(img2bin(res_bin).astype('uint8'))
        img.save(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas/sumframe_res" + ".png")
        exit_list.append(compare(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas/sumframe_res" + ".png"))
    print(exit_list)
    hm_list.append(compare(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas/sumframe_res" + ".png"))
    #print(compare(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas_bin/result" + str(i-1) + ".png"))

    print("current percent", stop_kadr1)
    print("current percent", stop_kadr2)

    # fig, ax = plt.subplots()
    # ax.plot([i for i in range(0, 3000)], stop_kadr1)
    # ax.plot([i for i in range(0, 3000)], stop_kadr2)
    #
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(100))  # Вертикальное выравнивание
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # поворачиваем подписи
    #
    # plt.xlabel("Номер кадра")
    # plt.ylabel("Изменение точности извлечения по кадрам")
    # plt.show()


    tmp = np.zeros((65, 65))
    # for i in range(1, 3000,15):
    #     print(i)
    #     my_exit.append(compare(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas_bin/result" + str(i) + ".png"))

    # sort_list=(np.array([my_exit]).argsort())

    #tetta+=0.5
    alfa +=0.1

