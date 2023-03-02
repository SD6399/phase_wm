import math
from skimage import io
from reedsolo import RSCodec
import ffmpeg
from skimage.exposure import histogram
import csv
import cv2, os
import numpy as np
from PIL import Image, ImageFile
# from qrcode_1 import read_qr, correct_qr
from helper_methods import small2big, big2small, sort_spis, img2bin
from reedsolomon import extract_RS

ImageFile.LOAD_TRUNCATED_IMAGES = True
size_quadr = 16


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


def disp_pix(path, coord_x, coord_y):
    cnt = 1
    arr = np.array([])

    list_diff = []
    while cnt < total_count:
        tmp = np.copy(arr[coord_x, coord_y])
        arr = io.imread(path + str(cnt) + ".png").astype(float)[coord_x, coord_y]

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


def embed(my_i, tt, count, var):
    cnt = 0

    PATH_IMG = r'C:\Users\user\PycharmProjects\phase_wm\RS_cod89x89.png'
    fi = math.pi / 2 / 255

    st_qr = cv2.imread(PATH_IMG)
    st_qr = cv2.cvtColor(st_qr, cv2.COLOR_RGB2YCrCb)

    data_length = st_qr[:, :, 0].size
    # Here we shuffle matrix
    shuf_order = np.arange(data_length)

    np.random.seed(42)
    np.random.shuffle(shuf_order)

    st_qr_1d = st_qr[:, :, 0].ravel()
    shuffled_data = st_qr_1d[shuf_order]  # Shuffle the original data
    # transpose matrix

    res = np.resize(shuffled_data, (1057, 1920))
    res[-1, 256 - 1920:] = 0
    # fi = np.random.uniform(low=0, high=np.pi / 2 / 255, size=(res.shape[0], res.shape[1]))

    while cnt < count:
        imgg = io.imread(r"C:\Users\user\PycharmProjects\phase_wm\frames_orig_video/frame%d.png" % cnt)
        # a = imgg
        a = cv2.cvtColor(imgg, cv2.COLOR_RGB2YCrCb)

        temp = np.float32(fi) * np.float32(res)
        wm = np.asarray((my_i * np.sin(cnt * tt + temp)))
        if my_i == 1:
            wm[wm > 0] = 1
            wm[wm < 0] = -1

        a[0:1057, :, 0] = np.where(np.float32(a[0:1057, :, 0] + wm) > 255, 255,
                                   np.where(a[0:1057, :, 0] + wm < 0, 0, np.float32(a[0:1057, :, 0] + wm)))
        # a[a>255]=255
        # a[a<0]=0
        tmp = cv2.cvtColor(a, cv2.COLOR_YCrCb2RGB)
        # tmp=a

        row, col, ch = tmp.shape
        mean = 0

        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        gauss[gauss < 0] = 0
        noisy = tmp + gauss

        img = Image.fromarray(noisy.astype('uint8'))

        img.convert('RGB').save(r"C:\Users\user\PycharmProjects\phase_wm\frames_after_emb\result" + str(cnt) + ".png")
        if cnt % 300 == 0:
            print("wm embed", cnt)
        cnt += 1

    print(shuf_order)


def read2list(file):
    # открываем файл в режиме чтения utf-8
    file = open(file, 'r', encoding='utf-8')

    # читаем все строки и удаляем переводы строк
    lines = file.readlines()
    lines = [line.rstrip('\n') for line in lines]

    file.close()

    return lines


def extract(alf, tt, rand_fr):
    qr_for_compare = io.imread(r'C:\Users\user\PycharmProjects\phase_wm\qr_ver18_H.png')
    PATH_VIDEO = r'C:\Users\user\PycharmProjects\phase_wm\frames_after_emb\RB_codec.mp4'
    vidcap = cv2.VideoCapture(PATH_VIDEO)
    vidcap.open(PATH_VIDEO)

    betta = 0.999

    # count = 0
    count = int(rand_fr)

    success = True
    while success:
        success, image = vidcap.read()
        if success:
            cv2.imwrite(r'C:\Users\user\PycharmProjects\phase_wm\extract\frame%d.png' % count, image)
            if count % 300 == 0:
                print("frame extract", count)
        count += 1

    count = total_count

    cnt = int(rand_fr)
    g = np.asarray([])
    f = g.copy()
    f1 = f.copy()
    d = g.copy()
    d1 = d.copy()

    while cnt < count:
        arr = io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract/frame" + str(cnt) + ".png")
        a = arr
        # g1=d1 # !!!!!!!!!!!!!
        d1 = f1
        if cnt == rand_fr:
            f1 = a.copy()
            d1 = np.zeros((1080, 1920))
        # elif cnt == change_sc[scene-1] + 1:
        else:
            f1 = np.float32(d1) * alf + np.float32(a) * (1 - alf)
        # else:
        #     f1 = (1-alf)*(1-alf)*a+(1-alf)*alf*d1+alf*g1

        f1[f1 > 255] = 255
        f1[f1 < 0] = 0
        img = Image.fromarray(f1.astype('uint8'))
        if cnt % 300 == 0:
            print("first smooth", cnt)
        img.save(r'C:\Users\user\PycharmProjects\phase_wm\extract\first_smooth/result' + str(cnt) + '.png')

        cnt += 1

    cnt = int(rand_fr)
    count = total_count
    # count=1000
    shuf_order = read2list(r'C:\Users\user\PycharmProjects\phase_wm\shuf.txt')
    shuf_order = [eval(i) for i in shuf_order]
    # вычитание усреднённого
    while cnt < count:

        arr = np.float32(io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract\frame" + str(cnt) + ".png"))

        a = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
        # a = arr
        a1 = np.asarray([])
        f1 = np.float32(
            io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract\first_smooth\result" + str(cnt) + ".png"))
        # f1=np.float32(f1)
        f1 = cv2.cvtColor(f1, cv2.COLOR_RGB2YCrCb)
        a1 = np.where(a < f1, f1 - a, a - f1)

        res_1d = np.ravel(a1[0:1057, :, 0])[:256 - 1920]
        start_qr = np.resize(res_1d, (1424, 1424))

        unshuf_order = np.zeros_like(shuf_order)
        unshuf_order[shuf_order] = np.arange(start_qr.size)
        unshuffled_data = np.ravel(start_qr)[unshuf_order]
        matr_unshuf = np.resize(unshuffled_data, (1424, 1424))

        # извлечение ЦВЗ
        arr = matr_unshuf
        a = arr

        g = d
        d = f

        if cnt == rand_fr:
            f = a
            d = f.copy()
            d = np.ones((1424, 1424))

        else:
            if cnt == rand_fr + 1:
                f = 2 * betta * math.cos(tt) * np.float32(d) + np.float32(a)

            else:
                f = 2 * betta * math.cos(tt) * np.float32(d) - (betta ** 2) * np.float32(g) + np.float32(a)

        yc = np.float32(f) - betta * math.cos(tt) * np.float32(d)
        ys = betta * math.sin(tt) * np.float32(d)
        c = math.cos(tt * cnt) * np.float32(yc) + math.sin(tt * cnt) * np.float32(ys)
        s = math.cos(tt * cnt) * np.float32(ys) - math.sin(tt * cnt) * np.float32(yc)

        fi = np.where(c < 0, np.arctan((s / c)) + np.pi,
                      np.where(s >= 0, np.arctan((s / c)), np.arctan((s / c)) + 2 * np.pi))
        fi = np.where(fi < -np.pi / 4, fi + 2 * np.pi, fi)
        fi = np.where(fi > 9 * np.pi / 4, fi - 2 * np.pi, fi)
        wm = 255 * fi / 2 / math.pi

        wm[wm > 255] = 255
        wm[wm < 0] = 0

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
        # list001.append(coord1[0,0])
        coord2 = np.where(fi < np.pi / 2, (fi / np.pi / 2),
                          np.where(fi > 3 * np.pi / 2, ((fi - 1.5 * np.pi) / np.pi * 2) - 1,
                                   ((fi - 0.5 * np.pi) * 2 / np.pi - 1) * (-1)))
        # list_phas.append(coord2[0, 0])
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
                pr1 = bin_centers[i]

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

        # list_pr.append(pr1)
        # list_pr2.append(pr2)

        moment = np.where(pr1 < 0, np.arctan((pr2 / pr1)) + np.pi,
                          np.where(pr2 >= 0, np.arctan((pr2 / pr1)), np.arctan((pr2 / pr1)) + 2 * np.pi))

        # tmpmom=moment
        # list002.append(moment)
        if np.pi / 4 <= moment <= np.pi * 2 - np.pi / 4:
            fi_tmp = fi - moment + 0.5 * np.pi * 0.5
            fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
            fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)

        elif moment > np.pi * 2 - np.pi / 4:
            fi = np.where(fi < np.pi / 4, fi + 2 * np.pi, fi)
            fi_tmp = fi - moment + 0.5 * np.pi * 0.5
            fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
            fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)

        elif moment < np.pi / 4:
            fi_tmp = fi - 2 * np.pi - moment + 0.5 * np.pi * 0.5
            fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
            fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)

        # list001.append(fi_tmp[0,0])
        fi_tmp[fi_tmp < 0] = 0
        fi_tmp[fi_tmp > np.pi] = np.pi
        l_kadr = fi_tmp * 255 / np.pi

        small_frame = big2small(l_kadr)
        img = Image.fromarray(small_frame.astype('uint8'))
        img.save(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas/result" + str(cnt) + ".png")

        l_kadr = io.imread(
            r'C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas/result' + str(cnt) + '.png').astype(
            float)
        cp = l_kadr.copy()
        our_avg = np.mean(cp)

        k = -1

        for i in range(0, 89):
            for j in range(0, 89):
                k += 1
                if cp[i, j] > our_avg:
                    cp[i, j] = 255
                else:
                    cp[i, j] = 0

        # cp = correct_qr(cp)
        imgc = Image.fromarray(cp.astype('uint8'))

        imgc.save(
            r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas_bin/result" + str(cnt) + ".png")
        # print("wm extract", cnt)
        if cnt % 100 == 96:
            # tmp1 = read_qr(
            #     small2big(
            #         io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas_bin/result" + str(
            #             cnt) + ".png")))
            # tmp2 = read_qr(qr_for_compare)
            # stop_kadr1_bin.append(tmp1 == tmp2)
            extract_RS(io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas_bin/result" + str(cnt) + ".png"),rsc)
            stop_kadr1.append(compare(
                r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas_bin/result" + str(cnt) + ".png"))
            print(cnt, stop_kadr1)

        cnt += 1

    count = total_count

    cnt = int(rand_fr)
    g2 = np.asarray([])
    f = np.copy(g2)
    alf2 = 0.13

    while cnt < count:

        arr = io.imread(
            r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas/result" + str(cnt) + ".png")
        # g2 - y(n-1)
        y_step_1 = f
        if cnt == rand_fr:
            f = arr.copy()
            f_step_1 = np.zeros((89, 89))
        else:
            # y(n)=alfa*y(n-1)+x(n)*(1-alfa)
            f = y_step_1 * alf2 + arr * (1 - alf2)
            f[f > 255] = 255

        img = Image.fromarray(f.astype('uint8'))
        img.save(r"C:\Users\user\PycharmProjects\phase_wm\extract/wm_after_2_smooth/result" + str(cnt) + ".png")
        if cnt % 300 == 0:
            print("wm 2 smooth", cnt)
        cnt += 1

    count = total_count
    cnt = int(rand_fr)
    while cnt < count:
        c_qr = io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract\wm_after_2_smooth\result" + str(cnt) + ".png")
        c_qr = img2bin(c_qr)
        # c_qr = correct_qr(c_qr)
        img1 = Image.fromarray(c_qr.astype('uint8'))
        img1.save(r"C:\Users\user\PycharmProjects\phase_wm\extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png")
        if cnt % 300 == 0:
            print("2 smooth bin", cnt)
        if cnt % 100 == 96:
            #     tmp1 = read_qr(
            #         small2big(
            #             io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract/wm_after_2_smooth_bin/result" + str(
            #                 cnt) + ".png")))
            #     tmp2 = read_qr(qr_for_compare)
            #     stop_kadr2_bin.append(tmp1 == tmp2)
            extract_RS(io.imread(
                r"C:\Users\user\PycharmProjects\phase_wm\extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png"),
                       rsc)
            stop_kadr2.append(
                compare(
                    r"C:\Users\user\PycharmProjects\phase_wm\extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png"))

        cnt += 1

    return r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas_bin/result" + str(2996) + ".png"


def generate_video(bitr):
    image_folder = r'C:\Users\user\PycharmProjects\phase_wm\frames_after_emb'  # make sure to use your folder
    video_name = 'need_video.mp4'
    os.chdir(r"C:\Users\user\PycharmProjects\phase_wm\frames_after_emb")

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".png")]
    sort_name_img = sort_spis(images)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'H264')

    video = cv2.VideoWriter(video_name, -1, 29.97, (width, height))

    cnt = 0
    for image in sort_name_img:
        if cnt % 300 == 0:
            print(cnt)
        video.write(cv2.imread(os.path.join(image_folder, image)))
        cnt += 1
    cv2.destroyAllWindows()
    video.release()

    os.system(
        f"ffmpeg -y -i C:/Users/user/PycharmProjects/phase_wm/frames_after_emb/need_video.mp4 -b:v {bitr}M -vcodec libx264  C:/Users/user/PycharmProjects/phase_wm/frames_after_emb/RB_codec.mp4")


def compare(path):  # сравнивание извлечённого QR с исходным
    orig_qr = io.imread(r'C:\Users\user\PycharmProjects\phase_wm\RS_cod89x89.png')
    orig_qr = np.where(orig_qr > 127, 255, 0)
    small_qr = big2small(orig_qr)
    sr_matr = np.zeros((1424, 1424, 3))
    myqr = io.imread(path)
    myqr = np.where(myqr > 127, 255, 0)

    k = 0
    mas_avg = []
    for i in range(0, 89):
        for j in range(0, 89):

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
    for i in range(0, 89):
        for j in range(0, 89):

            if qr1[i, j] == qr2[i, j]:
                mas_avg.append(1)
            else:
                mas_avg.append(0)

    for i in mas_avg:
        if i == 0:
            k += 1
    return k


my_exit = []
my_exit1 = []
my_exit2 = []

squ_size = 4
for_fi = 6
# dispr=1

# графики-сравнения по различныи параметрам

PATH_VIDEO = r'Road.mp4'

with open('change_sc.csv', 'r') as f:
    change_sc = list(csv.reader(f))[0]

change_sc = [eval(i) for i in change_sc]

# count=read_video(PATH_VIDEO)

rand_k = 0
total_count = 2999

hm_list = []
rsc = RSCodec(nsym=28, nsize=31)
alfa = 0.01
tetta = 1
ampl = 3
sp = []

bitr = 6.2
for var in np.arange(0, 1.2, 2):
    embed(ampl, tetta, total_count, var)
    generate_video(bitr)
    stop_kadr1 = []
    stop_kadr2 = []
    stop_kadr1_bin = []
    stop_kadr2_bin = []

    info = ffmpeg.probe(r'C:\Users\user\PycharmProjects\phase_wm\RealBarca.mp4')
    print(info['streams'][0]['bit_rate'])
    info = ffmpeg.probe(r'C:\Users\user\PycharmProjects\phase_wm\frames_after_emb\RB_codec.mp4')
    print(info['streams'][0]['bit_rate'])
    info = ffmpeg.probe(r'C:\Users\user\PycharmProjects\phase_wm\frames_after_emb\need_video.mp4')
    print(info['streams'][0]['bit_rate'])
    print('GEN')
    a = extract(alfa, tetta, rand_k)
    print("all")

    hand_made = [0, 118, 404, 414, 524, 1002, 1391, 1492, 1972, 2393, 2466, total_count]
    exit_list = []
    res = np.zeros((89, 89))
    res_bin = np.zeros((89, 89))

    # for ii in range(1, len(hand_made)):
    #     tnp = io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract/wm_after_2_smooth/result" + str(
    #         hand_made[ii] - 1) + ".png")
    #     res[tnp >= np.mean(tnp)] += (hand_made[ii] - hand_made[ii - 1])
    #     res[tnp < np.mean(tnp)] -= (hand_made[ii] - hand_made[ii - 1])
    #     # res2=img2bin(tnp)
    #     img = Image.fromarray(res.astype('uint8'))
    #     img.save(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas/sumframe" + str(ii) + ".png")
    #
    #     # print(compare(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas/sumframe" +str(i)+ ".png"), change_sc[i])
    #     res_bin[res >= 0] = 255
    #     res_bin[res < 0] = 0
    #
    #     img = Image.fromarray(img2bin(res_bin).astype('uint8'))
    #     img.save(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas/sumframe_res" + ".png")
    #     exit_list.append(compare(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas/sumframe_res" + ".png"))
    #     print(exit_list)
    #     hm_list.append(compare(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas/sumframe_res" + ".png"))
    #     # print(compare(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas_bin/result" + str(i-1) + ".png"))

    print("RANDOM FRAME", rand_k)
    print(ampl, tetta, alfa, "current percent", stop_kadr1)
    print(ampl, tetta, alfa, "current percent", stop_kadr2)
    print(ampl, tetta, alfa, "current percent", stop_kadr1_bin)
    print(ampl, tetta, alfa, "current percent", stop_kadr2_bin)

    # tetta += 0.1

# fig = plt.figure()
# ax = fig.add_subplot(111, label="1")
# plt.plot([i for i in np.arange(196, total_count, 200)], stop_kadr1)
# plt.plot([i for i in np.arange(196, total_count, 200)], stop_kadr2)
# plt.show()
