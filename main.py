import math
from skimage import io
from skimage.util import random_noise
from scipy import interpolate
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
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
    while cnt < total_count:
        tmp = np.copy(arr)
        arr = io.imread(path + str(cnt) + ".png").astype(float)
        if cnt == 0:
            list_diff.append(0)

        else:
            diff_img = np.abs(arr - tmp)

            list_diff.append(np.mean(diff_img))
        cnt += 1

    avg = sum(list_diff) / len(list_diff)

    upd_start = []
    for i in range(len(list_diff)):
        if abs((list_diff[i])) > (4 * avg):
            upd_start.append(i)

    return list_diff


def disp_pix(path,coord_x,coord_y):
    cnt = 1
    arr = np.array([])

    list_diff = []
    while cnt < total_count:
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
        if success:
            cv2.imwrite(r"frames_orig_video\frame%d.png" % count, image)

        print("записан кадр", count)

        if cv2.waitKey(10) == 27:
            break
        count += 1
    return count

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
    PATH_IMG = r'C:\Users\user\PycharmProjects\phase_wm\qr_ver18_L.png'
    fi = math.pi / 2 / 255

    st_qr = io.imread(PATH_IMG)
    #arr1 = cv2.cvtColor(st_qr, cv2.COLOR_RGB2YCrCb)

    data_length = st_qr.size
    # Here we shuffle matrix
    shuf_order = np.arange(data_length)
    np.random.seed(42)
    np.random.shuffle(shuf_order)

    st_qr_1d = st_qr.ravel()
    shuffled_data = st_qr_1d[shuf_order]  # Shuffle the original data
    matr_shuf = np.resize(shuffled_data, (1424, 1424,3))

    # transpose matrix

    qr_1d = np.ravel(matr_shuf)
    res = np.resize(qr_1d, (1057, 1920))
    res[-1, 256 - 1920:] = 0

    while cnt < count:
        imgg = (io.imread(r"C:\Users\user\PycharmProjects\phase_wm\frames_orig_video/frame%d.png" % cnt))

        a = cv2.cvtColor(imgg, cv2.COLOR_RGB2YCrCb)

        temp = np.float32(fi) * np.float32(res)
        wm = np.asarray((my_i * np.sin(cnt * tt + temp)))
        if my_i == 1:
            wm[wm > 0] = 1
            wm[wm < 0] = -1

        tmp = np.float32(a[0:1057, :,0] + wm)
        a[a > 255] = 255
        a[a < 0] = 0
        a[0:1057,:, 0] = tmp

        tmp = cv2.cvtColor(a, cv2.COLOR_YCrCb2RGB)
        img = Image.fromarray(tmp.astype('uint8'))

        img.convert('RGB').save(r"C:\Users\user\PycharmProjects\phase_wm\frames_after_emb\result" + str(cnt) + ".png")
        print("wm embed", cnt)
        cnt += 1


def extract(alf, tt,  rand_fr):
    PATH_VIDEO = r'C:\Users\user\PycharmProjects\phase_wm\frames_after_emb\RB_codH264.mp4'
    vidcap = cv2.VideoCapture(PATH_VIDEO)
    vidcap.open(PATH_VIDEO)

    alf0 = 0.96
    betta = 0.999
    alf2 = 0.95

    count = 0
    count = rand_fr

    success = True
    while success:
        success, image = vidcap.read()
        if success:

            cv2.imwrite(r'C:\Users\user\PycharmProjects\phase_wm\extract\frame%d.png' % count, image[20:1060, 440:1480])
            print("frame extract", count)
        count += 1

    count = total_count

    cnt = rand_fr
    g = np.asarray([])
    f = g.copy()
    f1 = f.copy()
    d = g.copy()
    d1=d.copy()

    cnt = 0

    for scene in range(1, len(change_sc)):
        cnt = change_sc[scene - 1]
        while cnt < change_sc[scene]:
            arr = io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract/frame" + str(cnt) + ".png")
            a = arr
            # g1=d1 # !!!!!!!!!!!!!
            d1 = f1
            if cnt == change_sc[scene-1]:
                f1 = a.copy()
                d1 = np.zeros((1040, 1040))
            # elif cnt == change_sc[scene-1] + 1:
            else:
                f1 = np.float32(d1) * alf + np.float32(a) * (1 - alf)
            # else:
            #     f1 = (1-alf)*(1-alf)*a+(1-alf)*alf*d1+alf*g1

            f1[f1 > 255] = 255
            f1[f1 < 0] = 0
            img = Image.fromarray(f1.astype('uint8'))
            print("first smooth", cnt)
            img.save(r'C:\Users\user\PycharmProjects\phase_wm\extract\first_smooth/result' + str(cnt) + '.png')

            cnt += 1

    cnt = rand_fr

    # вычитание усреднённого
    while cnt < count:
        # cnt = change_sc[scene - 1]
        # while cnt < change_sc[scene]:

        arr = io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract\frame" + str(cnt) + ".png")

        a = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
        a1 = np.asarray([])
        f1=np.float32(io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract\first_smooth\result" + str(cnt) + ".png"))
        #f1=np.float32(f1)
        f1 = cv2.cvtColor(f1, cv2.COLOR_RGB2YCrCb)
        a1 = np.where(a < f1, 0, a - f1)

        # извлечение ЦВЗ
        arr = a1
        a = arr

        g = d
        d = f

        if cnt == rand_fr:
            f = a[:, :, 0]
            d = f.copy()
            d = np.ones((1040, 1040))

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
        img = Image.fromarray(a1.astype('uint8'))
        img.save(r'C:\Users\user\PycharmProjects\phase_wm\extract/wm/result' + str(cnt) + '.png')


        # привдение к рабочему диапазону

        l_kadr = io.imread(r'C:\Users\user\PycharmProjects\phase_wm\extract/wm/result' + str(cnt) + '.png')

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

        fi_tmp[fi_tmp < 0] = 0
        fi_tmp[fi_tmp > np.pi] = np.pi
        l_kadr = fi_tmp * 255 / np.pi

        small_frame = big2small(l_kadr)
        img = Image.fromarray(small_frame.astype('uint8'))
        img.save(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas/result" + str(cnt) + ".png")

        l_kadr = io.imread(
            r'C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas/result' + str(cnt) + '.png').astype(
            float)
        cp = l_kadr[:, :, 0].copy()
        our_avg = np.mean(cp)

        k = -1
        matr_avg = np.zeros((65, 65))
        for i in range(0, 65):
            for j in range(0, 65):
                k += 1
                # matr_avg[int(i / 16), int(j / 16)] = np.mean(cp[i:i + 16, j:j + 16])
                if cp[i, j] > our_avg:
                    cp[i, j] = 255
                else:
                    cp[i, j] = 0

        imgc = Image.fromarray(cp.astype('uint8'))

        imgc.save(
            r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas_bin/result" + str(cnt) + ".png")
        print("wm extract", cnt)
        if cnt %200 ==196:

            stop_kadr1.append(compare(
                r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas_bin/result" + str(cnt) + ".png"))

        cnt += 1


    ### повторное сглаживание

    count = total_count

    cnt = 0
    g2 = np.asarray([])
    f = np.copy(g2)
    alf2 = 0.95

    while cnt < count:

        arr = io.imread(
            r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas_bin/result" + str(cnt) + ".png")
        # g2 - y(n-1)
        y_step_1 = f
        if cnt == 0:
            f = arr.copy()
            f_step_1 = np.zeros((65, 65))
        else:
            # y(n)=alfa*y(n-1)+x(n)*(1-alfa)
            f = y_step_1 * alf2 + arr * (1 - alf2)
            f[f > 255] = 255

        img = Image.fromarray(f.astype('uint8'))
        img.save(r"C:\Users\user\PycharmProjects\phase_wm\extract/wm_after_2_smooth/result" + str(cnt) + ".png")
        print("wm 2 smooth", cnt)
        cnt += 1

    count = total_count
    cnt=0
    while cnt < count:
        c_qr = io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract\wm_after_2_smooth\result" + str(cnt) + ".png")
        c_qr = img2bin(c_qr)

        img1 = Image.fromarray(c_qr.astype('uint8'))
        img1.save(r"C:\Users\user\PycharmProjects\phase_wm\extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png")
        print("2 smooth bin", cnt)
        if cnt % 200 == 196:
            stop_kadr2.append(
                compare(r"C:\Users\user\PycharmProjects\phase_wm\extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png"))

        cnt += 1

    return r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas_bin/result" + str(2999) + ".png"


def generate_video():
    image_folder = r'C:\Users\user\PycharmProjects\phase_wm\frames_after_emb'  # make sure to use your folder
    video_name = 'RB_codH264.mp4'
    os.chdir(r"C:\Users\user\PycharmProjects\phase_wm\frames_after_emb")

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".png")]
    sort_name_img = sort_spis(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter(video_name, fourcc, 29.97, (width, height))

    cnt = 0
    for image in sort_name_img:
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
alfa = 0.01
tetta = 0.18
squ_size = 4
for_fi = 6
# dispr=1

# графики-сравнения по различныи параметрам

PATH_VIDEO = r'cut_RealBarca.mp4'

with open('change_sc.csv', 'r') as f:
    change_sc = list(csv.reader(f))[0]

change_sc = [eval(i) for i in change_sc]

#count=read_video(PATH_VIDEO)

rand_k = 0
total_count = 2997

hm_list=[]

while tetta < 0.19:
    stop_kadr1=[]
    stop_kadr2=[]
    sp = []
    embed(i,tetta,total_count)
    generate_video()
    a = extract(alfa, tetta, rand_k)
    print("all")

    hand_made= [0,118,404,414,524,1002,1391,1492,1972,2393,2466,total_count]
    exit_list=[]
    res = np.zeros((65, 65))
    res_bin = np.zeros((65, 65))
    for i in range(1, len(hand_made)):

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

    print(tetta,alfa, "current percent", stop_kadr1 )
    print(tetta,alfa, "current percent", stop_kadr2)

    tetta +=0.05


fig = plt.figure()
ax = fig.add_subplot(111, label="1")
plt.plot([i for i in np.arange(196, total_count, 200)], stop_kadr1)
plt.plot([i for i in np.arange(196, total_count, 200)], stop_kadr2)
plt.show()
