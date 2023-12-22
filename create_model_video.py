import math
from itertools import repeat
from multiprocessing import Pool
import multiprocessing
import numpy as np
from skimage import io
from PIL import Image
import os, re
import cv2
from model_of_video import gener_field, calc_ACF2, SIZE, SIZE_HALF, plot_ACF
from model_of_moving import FieldGenerator
import matplotlib.pyplot as plt
from helper_methods import disp

hc_const = 9702.05325519
alf = 0.995
ro = 0.959
var_disp = 500
# per_of_jumps = int(1 / (1 - alf))
# print(per_of_jumps)
# count_of_frame = 512
#
# count_jump = math.ceil(500 / 25)
#
# rand_jump = np.random.choice(range(500), count_jump, replace=False)
# print(np.sort(rand_jump))
# rand_jump = np.append(rand_jump, 0)

# rand_jump = [0, 36, 77, 82, 120, 136, 184, 243, 278, 285, 290, 291, 308, 345, 348, 365, 375, 394, 403, 467]
rand_jump = [ 11,  16,  70,  79, 100, 120, 190, 212, 218, 224, 229, 254, 256, 272, 315, 363, 433, 458, 460, 471]
print("list", rand_jump)


def generate_video(path):
    image_folder = path  # make sure to use your folder
    video_name = "40change_ro=" + str(ro)+"_"+str(var_disp)+ ".mp4"
    os.chdir(path)

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".png")]
    sort_name_img = sort_spis(images, "need_sum")
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, -1, 29.97, (width, height))

    cnt = 0
    for image in sort_name_img:
        if cnt % 100 == 0:
            print(cnt)
        video.write(cv2.imread(os.path.join(image_folder, image)))
        cnt += 1
    cv2.destroyAllWindows()
    video.release()


def sort_spis(sp, word):
    spp = []
    spb = []
    res = []
    for i in sp:
        spp.append("".join(re.findall(r'\d', i)))
        spb.append(word)
    result = [int(item) for item in spp]
    result.sort()

    result1 = [str(item) for item in result]
    for k in range(len(sp)):
        res.append(spb[k] + result1[k] + ".png")
    return res


def add_noise(img, mean, var, seed):
    row, col = img.shape
    img = img.astype(float)
    sigma = var ** 0.5
    np.random.seed(seed)
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    # print("Gauss max and min",np.max(gauss), np.min(gauss))
    new_img = np.where(img + gauss > 255, 255, np.where(img + gauss < 0, 0, img + gauss))
    # new_img = img + gauss

    return new_img


def process_range(start, end, result_queue):
    results = []
    for i in range(start, end):
        sq = gener_field(list_ACF2, rand_jump[i])
        mosaic = FieldGenerator.draw_mosaic_field(20, ro, 1080, 1920, 0, var_disp, i)
        mosaic_aft_noise = add_noise(mosaic, 0, 49, 42 * i)
        texture_aft_noise = add_noise(sq, 0, 100, rand_jump[i])
        final_frame = np.where(texture_aft_noise[:1080, :1920] + mosaic_aft_noise[:1080, :1920] > 255, 255,
                               np.where(texture_aft_noise[:1080, :1920] + mosaic_aft_noise[:1080, :1920] < 0, 0,
                                        texture_aft_noise[:1080, :1920] + mosaic_aft_noise[:1080, :1920]))
        img1 = Image.fromarray(final_frame.astype('uint8'))
        img1.save(r"D:/pythonProject/phase_wm/sum_mosaic" + str(rand_jump[i]) + ".png")
        results.append(final_frame)
    result_queue.put(results)


def func_sum_noise(folder_to_img, fold_to_save):
    list_of_change_frame = []
    for file in os.listdir(folder_to_img):
        if file.endswith(".png"):
            if "sum_mosaic" in file:
                digit_indices = [file[index] for index, char in enumerate(file) if char.isdigit()]
                result_string = ''.join(digit_indices)
                list_of_change_frame.append(int(result_string))
    sort_list_img = (np.sort(list_of_change_frame))
    print(sort_list_img)
    cnt = 0
    for ind in sort_list_img[:-1]:
        img = io.imread(folder_to_img + "/sum_mosaic" + str(ind) + ".png")
        print(sort_list_img[list(sort_list_img).index(ind) + 1])

        while cnt < sort_list_img[list(sort_list_img).index(ind) + 1]:
            texture_aft_noise = add_noise(img, 0, 100, cnt)

            img1 = Image.fromarray(texture_aft_noise.astype('uint8'))
            img1.save(fold_to_save + "/need_sum" + str(cnt) + ".png")
            print(cnt, ind)
            cnt += 1

            # cnt += 1
            # img = io.imread(file)
            # texture_aft_noise = add_noise(img, 0, 100, cnt)


#
# # need_params2 = (0.5, 0.01, 0.014)
# # need_params2 = (0.5, 0.3, 0.1) # - works correctly 25.09 at morning and was so good graphic by spatial

need_params2 = (0.5, 0.1, 0.1)

list_ACF2 = np.zeros((SIZE, SIZE))
tmp_matr = np.zeros((SIZE_HALF, SIZE_HALF))
for x in range(0, SIZE_HALF):
    for y in range(0, SIZE_HALF):
        tmp_matr[x][y] = (hc_const * calc_ACF2(need_params2[0], need_params2[1], need_params2[2], x, y))

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
#
# # rand_jump = np.random.randint(3000, size=int(3000 / per_of_jumps))


# rand_jump=np.sort(rand_jump)
#
# print(rand_jump)
# #
# image_orig = io.imread("D:/pythonProject/phase_wm/embedding_BBC/result0.png")[:, :, 0]
# print("Mean of orig image", np.mean(image_orig))
# graph_orig = plot_ACF(image_orig)
# np.save(r"D:/pythonProject/phase_wm/graph_ACF_MO72.py",graph_orig)
graph_orig = np.load(r"D:/pythonProject/phase_wm/graph_ACF_MO72.npy")
graph_new = np.load("D:/pythonProject/phase_wm/mean ACF.npy")
mosaic = np.zeros((2048, 2048))
texture_aft_noise = np.zeros((2048, 2048))

# M = pool.starmap(gener_field, zip(repeat(list_ACF2), rand_jump))
# if i in rand_jump:
#     texture = gener_field(list_ACF2, i)[:1080, :1920]
#     print("mean of texture", np.mean(texture))
# L = pool.starmap(add_noise, zip(repeat(list_ACF2), rand_jump))
# add_noise(texture, 0, 100, i)
#     print("mean of texture after sum noise", np.mean(texture_aft_noise))
#     # print(np.var(texture))
#     # print("0 0 :",texture[0][0])
#
#     # mosaic = FieldGenerator.draw_mosaic_field(5, 0.9959, 1080, 1920, 127, 2100, i)
#     mosaic = FieldGenerator.draw_mosaic_field(20, 0.9959, 1080, 1920, 0, 1465, i)
#
#     mosaic += (62 - np.mean(mosaic))
#     # mosaic += 72
#     print("mean of mosaic before+=", np.mean(mosaic), "var of mosaic", np.var(mosaic))
#     # mosaic = io.imread("D:/pythonProject/phase_wm/mosaics/mosaic"+str(list(rand_jump).index(i))+".png")
#     print("Field was regenerated ")
#
# if i == 0:
#     txt = plot_ACF(texture)
#     plot_tan = plot_ACF(texture_aft_noise)
#     print("VARIANCe of texture", txt[0])
#     print(mosaic.shape)
#     print(np.var(mosaic))
#     img1 = Image.fromarray(mosaic.astype('uint8'))
#     img1.save(r"D:/pythonProject/phase_wm/mosaic" + str(i) + ".png")
#     print("--------------")
#
#     msc = plot_ACF(mosaic[:1080, :1920])
#
#     row, col = mosaic[:1080, :1920].shape
#     img = msc.astype(float)
#
#     # sigma = 64 ** 0.5
#     # np.random.seed(i)
#     # gauss = np.random.normal(0, sigma, (row, col))
#     # gauss = gauss.reshape(row, col)
#     # sigma2 = 49 ** 0.5
#     # np.random.seed(42*i)
#     # gauss2 = np.random.normal(0, sigma2, (row, col))
#     # gauss2 = gauss2.reshape(row, col)
#
#     print("mean texture", np.mean(txt))
#     # summ = plot_ACF(texture+mosaic+gauss)
#     print("--------------")
#     print("Noise ACF")
#     # noise = plot_ACF(gauss)
#     # noise2 = plot_ACF(gauss2)
#
#     # mosaic_aft_noise = add_noise(mosaic,0,49,42*i)
#
#     # plt.plot(txt[:200], label="Texture ACF")
#     # plt.plot(msc[:200], label="Mosaic ACF")
#
#     # plt.plot(plot_tan[:200], label="Texture + noise2")
#
#     # plt.plot(graph_orig[:200], label="Original image ACF")
#     plt.plot(graph_new[:200], label="Mean ACF by original video")
#
#     # Текстурный шум
#
# # добавление мозаичного шума
# mosaic_aft_noise = add_noise(mosaic, 0, 29 * 29, i + 3)
# # mosaic_aft_noise = np.copy(mosaic)
# # texture_aft_noise = texture
# # mosaic_aft_noise = mosaic
#
# final_frame = np.where(texture_aft_noise[:1080, :1920] + mosaic_aft_noise[:1080, :1920] > 255, 255,
#                        np.where(texture_aft_noise[:1080, :1920] + mosaic_aft_noise[:1080, :1920] < 0, 0,
#                                 texture_aft_noise[:1080, :1920] + mosaic_aft_noise[:1080, :1920]))
#
# # if i == 1:
# #     lag = io.imread("D:/pythonProject/phase_wm/fold_model_video/final_img" + str(i - 1) + ".png")
# #
# #     tmp = final_frame - lag
# #     print(i, np.mean(tmp))
#
# # final_frame = texture_aft_noise[:1080,:1920] + mosaic_aft_noise[:1080,:1920]
# if i == 0:
#     print("max", np.max(texture_aft_noise), "min", np.min(texture_aft_noise))
#     print("max", np.max(mosaic_aft_noise), "min", np.min(mosaic_aft_noise))
#     plot_man = plot_ACF(mosaic_aft_noise)
#     # plt.plot(plot_man[:200], label="Mosaic + noise")
#     full_pict = np.where(texture_aft_noise + mosaic_aft_noise > 255, 255,
#                          np.where(texture_aft_noise + mosaic_aft_noise < 0, 0,
#                                   texture_aft_noise + mosaic_aft_noise))
#     summ = plot_ACF(final_frame)
#     print("mean sum", np.mean(summ))
#     print("THIS SUM", summ)
#     plt.plot(summ[:200], label="Model ACF")
#     print(np.var(summ))
#     print("REGENERATED FIELDS", i)
#     # for_save=plot_ACF(final_frame)
#     #
#     # plt.plot(for_save[:200],label = "for saving")
#     plt.title("Spatial")
#     plt.legend()
#     plt.show()
#
# img1 = Image.fromarray(final_frame.astype('uint8'))
# img1.save(r"D:/pythonProject/phase_wm/fold_model_video/final_img" + str(i) + ".png")


if __name__ == '__main__':

    # num_processes = multiprocessing.cpu_count()
    # result_queue = multiprocessing.Queue()
    # jobs = []
    #
    # for i in range(num_processes):
    #     start = i * (len(rand_jump) // num_processes)
    #     end = (i + 1) * (len(rand_jump) // num_processes)
    #     process = multiprocessing.Process(target=process_range, args=(start, end, result_queue))
    #     jobs.append(process)
    #     process.start()
    #
    # print("first loop was finished")
    # print(jobs)
    # for job in jobs:
    #     print("Job completed", jobs.index(job))
    #     job.join()
    #
    # print("all ")
    # final_results = []
    # for i in range(num_processes):
    #     final_results += result_queue.get()
    #
    # print(final_results)


    func_sum_noise("D:\pythonProject\phase_wm", "D:\pythonProject\phase_wm\BBC_method_sintez")
    generate_video("D:\pythonProject\phase_wm\BBC_method_sintez")
# disp(r"D:\pythonProject\phase_wm\fold_model_video",word='/final_img',name_of_doc="variance_Hand_Make.csv")
