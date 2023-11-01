import math
from skimage import io
# from reedsolo import RSCodec
from skimage.exposure import histogram
import cv2
import os
import numpy as np
from PIL import Image, ImageFile
# from qrcode_1 import read_qr, correct_qr
from helper_methods import small2big, big2small, sort_spis, img2bin
from helper_methods import csv2list, bit_voting
# from reedsolomon import extract_RS, rsc, Nbit
# from read_xml import Getting
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_video(path):
    """

    :param path: path of video
    :return: number of video frames
    """
    vidcap = cv2.VideoCapture(path)
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            cv2.imwrite(r"D:/pythonProject/phase_wm\frames_orig_video\frame%d.png" % count, image)
        if count % 25 == 24:
            print("записан кадр", count)

        if cv2.waitKey(10) == 27:
            break
        count += 1
    return count


def embed(folder_orig_image, folder_to_save, binary_image, amplitude, tt):
    """
    Procedure embedding
    :param binary_image: embedding code
    :param folder_orig_image: the folder from which the original images are taken
    :param folder_to_save: the folder where the images from the watermark are saved
    :param amplitude: embedding amplitude
    :param tt: reference frequency parameter
    """

    fi = math.pi / 2 / 255
    st_qr = cv2.imread(binary_image)
    st_qr = cv2.cvtColor(st_qr, cv2.COLOR_RGB2YCrCb)

    data_length = st_qr[:, :, 0].size
    shuf_order = np.arange(data_length)

    np.random.seed(42)
    np.random.shuffle(shuf_order)

    # Expand the binary image into a string
    st_qr_1d = st_qr[:, :, 0].ravel()
    shuffled_data = st_qr_1d[shuf_order]  # Shuffle the original data

    # 1d-string in the image
    pict = np.resize(shuffled_data, (1057, 1920))
    # the last elements are uninformative. Therefore, we make zeros
    pict[-1, 256 - 1920:] = 0
    images = [img for img in os.listdir(folder_orig_image)
              if img.endswith(".png")]

    # The list should be sorted by numbers after the name
    sort_name_img = sort_spis(images, "frame")
    cnt = 0

    while cnt < len(sort_name_img):
        # Reads in BGR format
        imgg = cv2.imread(folder_orig_image + sort_name_img[cnt])
        # translation to the YCrCb space
        a = cv2.cvtColor(imgg, cv2.COLOR_BGR2YCrCb)
        a = a.astype(float)

        temp = fi * pict
        # A*sin(m * teta + fi)
        wm = np.array((amplitude * np.sin(cnt * tt + temp)))

        # if my_i == 1:
        #     wm[wm > 0] = 1
        #     wm[wm < 0] = -1
        # Embedding in the Y-channel
        a[0:1057, :, 0] = np.clip((a[0:1057, :, 0] + wm), 0, 255)

        a = a.astype(np.uint8)
        tmp = cv2.cvtColor(a, cv2.COLOR_YCrCb2BGR)

        # Converting the YCrCb matrix to BGR
        img_path = os.path.join(folder_to_save)
        cv2.imwrite(img_path + "frame" + str(cnt) + ".png", tmp)

        if cnt % 300 == 0:
            print("wm embed", cnt)

        cnt += 1


def read2list(file):
    """

    :param file: file which transform to list
    :return: list of values
    """
    # opening the file in utf-8 reading mode
    file = open(file, 'r', encoding='utf-8')
    # we read all the lines and delete the newline characters
    lines = file.readlines()
    lines = [line.rstrip('\n') for line in lines]
    file.close()

    return lines


def extract(alf, tt, rand_fr):
    """
    Procedure embedding
    :param alf: primary smoothing parameter
    :param tt:reference frequency
    :param rand_fr: the frame from which the extraction begins
    :return: the path to the final image
    """
    PATH_VIDEO = r'D:/pythonProject/phase_wm\frames_after_emb\RB_codec.mp4'
    vidcap = cv2.VideoCapture(PATH_VIDEO)
    vidcap.open(PATH_VIDEO)

    betta = 0.999
    list00 = []

    count = int(rand_fr)

    success = True
    while success:
        success, image = vidcap.read()
        if success:
            cv2.imwrite(r'D:/pythonProject/phase_wm\extract\frame%d.png' % count, image)
            list00.append((image[0, 0, 0], image[444, 444, 0]))
            if count % 300 == 0:
                print("frame extract", count)
        count += 1
    # print("pixels after saving", list00)
    # count = total_count
    #
    cnt = int(rand_fr)
    g = np.asarray([])
    f = g.copy()
    f1 = f.copy()
    # d = g.copy()
    # d1 = d.copy()
    #
    # while cnt < 120:
    #     arr = io.imread(r"D:/pythonProject/phase_wm\extract/frame" + str(cnt) + ".png")
    #
    #     # g1=d1 # !!!!!!!!!!!!!
    #     d1 = f1
    #     if cnt == rand_fr:
    #         f1 = arr.astype('float32')
    #         d1 = np.zeros((1080, 1920), dtype='float32')
    #     # elif cnt == change_sc[scene-1] + 1:
    #     else:
    #         f1 = d1 * alf + arr * (1 - alf)
    #
    #     np.clip(f1, 0, 255, out=f1)
    #     img = Image.fromarray(f1.astype('uint8'))
    #     if cnt % 10 == 0:
    #         print("first smooth", cnt)
    #     img.save(r'D:/pythonProject/phase_wm\extract\first_smooth/result' + str(cnt) + '.png')
    #
    #     cnt += 1

    cnt = int(rand_fr)
    g = np.asarray([])
    f = g.copy()
    d = g.copy()
    count = 120

    # reading a shuffled object
    shuf_order = read2list(r'D:/pythonProject/phase_wm\shuf.txt')
    shuf_order = [eval(i) for i in shuf_order]
    # subtracting the average
    while cnt < count:

        arr = np.float32(cv2.imread(r"D:/pythonProject/phase_wm\extract\frame" + str(cnt) + ".png"))
        a = cv2.cvtColor(arr, cv2.COLOR_BGR2YCrCb)
        # a = arr
        # f1 = np.float32(
        #     cv2.imread(r"D:/pythonProject/phase_wm\extract\first_smooth\result" + str(cnt) + ".png"))
        # # f1=np.float32(f1)
        # f1 = cv2.cvtColor(f1, cv2.COLOR_RGB2YCrCb)
        # a1 = np.where(a < f1, f1 - a, a - f1)
        res_1d = np.ravel(a[0:1057, :, 0])[:256 - 1920]
        start_qr = np.resize(res_1d, (1424, 1424))

        unshuf_order = np.zeros_like(shuf_order)
        unshuf_order[shuf_order] = np.arange(start_qr.size)
        unshuffled_data = np.ravel(start_qr)[unshuf_order]
        matr_unshuf = np.resize(unshuffled_data, (1424, 1424))

        # extraction of watermark
        a = matr_unshuf
        g = np.copy(d)
        d = np.copy(f)

        if cnt == rand_fr:
            f = np.copy(a)
            d = np.ones((1424, 1424))

        else:
            if cnt == rand_fr + 1:
                f = 2 * betta * np.cos(tt) * np.float32(d) + np.float32(a)

            else:
                f = 2 * betta * np.cos(tt) * np.float32(d) - (betta ** 2) * np.float32(g) + np.float32(a)

        yc = np.float32(f) - betta * np.cos(tt) * np.float32(d)
        ys = betta * np.sin(tt) * np.float32(d)
        c = np.cos(tt * cnt) * np.float32(yc) + np.sin(tt * cnt) * np.float32(ys)
        s = np.cos(tt * cnt) * np.float32(ys) - np.sin(tt * cnt) * np.float32(yc)

        try:
            fi = np.where(c < 0, np.arctan((s / c)) + np.pi,
                          np.where(s >= 0, np.arctan((s / c)), np.arctan((s / c)) + 2 * np.pi))
        except ZeroDivisionError:
            fi = np.full(f.shape, 255)
        fi = np.nan_to_num(fi)
        fi = np.where(fi < -np.pi / 4, fi + 2 * np.pi, fi)
        fi = np.where(fi > 9 * np.pi / 4, fi - 2 * np.pi, fi)

        wm = 255 * fi / 2 / math.pi

        wm[wm > 255] = 255
        wm[wm < 0] = 0

        a1 = wm
        # # a1 = cv2.cvtColor(a1, cv2.COLOR_YCrCb2RGB)
        # img = Image.fromarray(big2small(a1).astype('uint8'))
        # img.save(r'D:/pythonProject/phase_wm\extract/wm/result' + str(cnt) + '.png')
        # bringing to the operating range

        # l_kadr = io.imread(r'D:/pythonProject/phase_wm\extract/wm/result' + str(cnt) + '.png')
        # compr= l_kadr==a1
        # fi = np.copy(l_kadr)
        fi = (a1 * np.pi * 2) / 255

        coord1 = np.where(fi < np.pi, (fi / np.pi * 2 - 1) * (-1), ((fi - np.pi) / np.pi * 2 - 1))
        coord2 = np.where(fi < np.pi / 2, (fi / np.pi / 2),
                          np.where(fi > 3 * np.pi / 2, ((fi - 1.5 * np.pi) / np.pi * 2) - 1,
                                   ((fi - 0.5 * np.pi) * 2 / np.pi - 1) * (-1)))

        # noinspection PyTypeChecker
        hist, bin_centers = histogram(coord1, normalize=False)
        # noinspection PyTypeChecker
        hist2, bin_centers2 = histogram(coord2, normalize=False)

        mx_sp = np.arange(bin_centers2[0], bin_centers2[-1], bin_centers2[1] - bin_centers2[0])
        ver = hist2 / np.sum(hist)
        mo = np.sum(bin_centers2 * ver)
        dis = np.abs(mo - mx_sp)
        pr1 = np.min(dis)

        mx_sp2 = np.arange(bin_centers2[0], bin_centers2[-1], bin_centers2[1] - bin_centers2[0])
        ver2 = hist2 / np.sum(hist2)
        mo = np.sum(bin_centers2 * ver2)
        dis2 = np.abs(mo - mx_sp2)
        x = np.min(dis2)

        # GPT
        idx = np.argmin(np.abs(dis2 - x))
        pr2 = bin_centers2[idx]

        moment = np.where(pr1 < 0, np.arctan((pr2 / pr1)) + np.pi,
                          np.where(pr2 >= 0, np.arctan((pr2 / pr1)), np.arctan((pr2 / pr1)) + 2 * np.pi))

        if np.pi / 4 <= moment <= np.pi * 2 - np.pi / 4:
            fi_tmp = fi - moment + 0.5 * np.pi * 0.5

        elif moment > np.pi * 2 - np.pi / 4:
            fi = np.where(fi < np.pi / 4, fi + 2 * np.pi, fi)
            fi_tmp = fi - moment + 0.5 * np.pi * 0.5

        else:
            fi_tmp = fi - 2 * np.pi - moment + 0.5 * np.pi * 0.5

        fi_tmp = np.where(fi_tmp < -np.pi / 4, fi_tmp + 2 * np.pi, fi_tmp)
        fi_tmp = np.where(fi_tmp > 9 * np.pi / 4, fi_tmp - 2 * np.pi, fi_tmp)
        fi_tmp[fi_tmp < 0] = 0
        fi_tmp[fi_tmp > np.pi] = np.pi
        l_kadr = fi_tmp * 255 / np.pi

        small_frame = big2small(l_kadr)
        img = Image.fromarray(small_frame.astype('uint8'))
        img.save(r"D:/pythonProject/phase_wm\extract/after_normal_phas/result" + str(cnt) + ".png")

        l_kadr = io.imread(
            r'D:/pythonProject/phase_wm\extract/after_normal_phas/result' + str(cnt) + '.png').astype(
            float)
        cp = l_kadr.copy()
        our_avg = np.mean(cp)
        cp = np.where(cp > our_avg, 255, 0)

        # cp = bit_voting(cp, Nbit)
        imgc = Image.fromarray(cp.astype('uint8'))

        imgc.save(
            r"D:/pythonProject/phase_wm\extract/after_normal_phas_bin/result" + str(cnt) + ".png")
        # print("wm extract", cnt)
        if cnt % 5 == 4:
            v = vot_by_variance(r"D:/pythonProject/phase_wm\extract\after_normal_phas_bin", 0, cnt, 0.045)
            vot_sp.append(max(v, 1 - v))

            stop_kadr1.append(max(compare(
                r"D:/pythonProject/phase_wm\extract/after_normal_phas_bin/result" + str(cnt) + ".png"),
                1 - compare(
                    r"D:/pythonProject/phase_wm\extract/after_normal_phas_bin/result" + str(
                        cnt) + ".png")))
            print(tt, cnt, stop_kadr1)
            print("after voting",tt, vot_sp)

        cnt += 1
    # count = total_count

    # cnt = int(rand_fr)
    # g2 = np.asarray([])
    # f = np.copy(g2)
    # alf2 = 0.13
    #
    # while cnt < count:
    #
    #     arr = io.imread(
    #         r"D:/pythonProject/phase_wm\extract/after_normal_phas/result" + str(cnt) + ".png")
    #     # g2 - y(n-1)
    #     y_step_1 = f
    #     if cnt == rand_fr:
    #         f = arr.copy()
    #         f_step_1 = np.zeros((89, 89))
    #     else:
    #         # y(n)=alfa*y(n-1)+x(n)*(1-alfa)
    #         f = y_step_1 * alf2 + arr * (1 - alf2)
    #         f[f > 255] = 255
    #
    #     img = Image.fromarray(f.astype('uint8'))
    #     img.save(r"D:/pythonProject/phase_wm\extract/wm_after_2_smooth/result" + str(cnt) + ".png")
    #     if cnt % 300 == 0:
    #         print("wm 2 smooth", cnt)
    #     cnt += 1
    #
    # count = total_count
    # cnt = int(rand_fr)
    # while cnt < count:
    #     c_qr = io.imread(r"D:/pythonProject/phase_wm\extract\wm_after_2_smooth\result" + str(cnt) + ".png")
    #     c_qr = img2bin(c_qr)
    #     # c_qr = correct_qr(c_qr)
    #     img1 = Image.fromarray(c_qr.astype('uint8'))
    #     img1.save(r"D:/pythonProject/phase_wm\extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png")
    #     if cnt % 300 == 0:
    #         print("2 smooth bin", cnt)
    #     if cnt % 50 == 46:
    #         v = vot_by_variance(r"D:/pythonProject/phase_wm\extract\after_normal_phas_bin", 0, cnt, 0.045)
    #         vot_sp.append(v)
    #         # if extract_RS(io.imread(
    #         #         r"D:/pythonProject/phase_wm\extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png"),
    #         #         rsc, Nbit) != b'':
    #         #     stop_kadr2_bin.append(1)
    #         # else:
    #         #     stop_kadr2_bin.append(0)
    #     stop_kadr2.append(
    #         compare(
    #             r"D:/pythonProject/phase_wm\extract/wm_after_2_smooth_bin/result" + str(cnt) + ".png"))
    #
    #     cnt += 1

    return r"D:/pythonProject/phase_wm\extract/after_normal_phas_bin/result" + str(2996) + ".png"


def generate_video(bitr):
    """
    Sequence of frames transform to compress video
    :param bitr: bitrate of output video
    """
    image_folder = r'D:/pythonProject/phase_wm\frames_after_emb'  # make sure to use your folder
    video_name = 'need_video.mp4'
    os.chdir(r"D:/pythonProject/phase_wm\frames_after_emb")

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".png")]
    sort_name_img = sort_spis(images, "frame")
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    # fourcc = cv2.VideoWriter_fourcc(*'H264')

    video = cv2.VideoWriter(video_name, -1, 29.97, (width, height))

    cnt = 0
    for image in sort_name_img:
        # if cnt % 300 == 0:

        video.write(cv2.imread(os.path.join(image_folder, image)))
        if cnt % 299 == 0:
            print(cnt)
        cnt += 1
    cv2.destroyAllWindows()
    video.release()

    os.system( f"ffmpeg -y -i D:/pythonProject/phase_wm/frames_after_emb/need_video.mp4 -b:v {bitr}M -vcodec"
               f" libx264  D:/pythonProject/phase_wm/frames_after_emb/RB_codec.mp4")


def compare(path):
    """
     Comparing the extracted QR with the original one
    :param path: path to code for comparison
    :return: percentage of similarity
    """

    orig_qr = io.imread(r"D:/pythonProject/phase_wm\qr_ver18_H.png")
    orig_qr = np.where(orig_qr > 127, 255, 0)
    small_qr = big2small(orig_qr)
    # sr_matr = np.zeros((1424, 1424, 3))
    myqr = io.imread(path)
    myqr = np.where(myqr > 127, 255, 0)

    sr_matr = small_qr == myqr
    k = np.count_nonzero(sr_matr)
    return k / sr_matr.size


def vot_by_variance(path_imgs, start, end, treshold):
    var_list = csv2list(r"D:/pythonProject/\phase_wm/RB_disp.csv")[start:end]
    sum_matrix = np.zeros((89, 89))
    np_list = np.array(var_list)
    need_ind = [i for i in range(len(np_list)) if np_list[i] > treshold]
    i = start
    count = 0
    while i < end:
        c_qr = io.imread(path_imgs + r"/result" + str(i) + ".png")
        c_qr[c_qr == 255] = 1
        if (i - start) not in need_ind:
            sum_matrix += c_qr
            count += 1
        else:
            i += 1
        i += 1

    sum_matrix[sum_matrix <= count * 0.5] = 0
    sum_matrix[sum_matrix > count * 0.5] = 255
    img1 = Image.fromarray(sum_matrix.astype('uint8'))
    img1.save(r"D:/pythonProject/phase_wm\voting" + ".png")
    comp = (compare(r"D:/pythonProject/phase_wm\voting" + ".png"))
    print(count)
    print(comp)
    # extract_RS(sum_matrix, rsc, Nbit)

    return comp


if __name__ == '__main__':

    ampl = 1
    alfa = 0.01
    PATH_IMG = r"D:/pythonProject//phase_wm\qr_ver18_H.png"
    # count = read_video(r'cut_RealBarca120.mp4')
    for teta in [3]:
        rand_k = 0
        vot_sp = []
        stop_kadr1 = []
        stop_kadr2 = []
        stop_kadr1_bin = []
        stop_kadr2_bin = []

        total_count = 2997
        input_folder = "D:/pythonProject/phase_wm/frames_orig_video/"
        output_folder = "D:/pythonProject/phase_wm/frames_after_emb/"

        # embed(input_folder, output_folder, PATH_IMG, ampl, teta)
        generate_video(5.5)
        extract(alfa, teta, rand_k)
