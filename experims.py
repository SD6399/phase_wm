import os
import re
from model_of_video import gener_field, calc_ACF2
import cv2
from model_of_moving import FieldGenerator
from model_of_video import plot_ACF, plot_ACF_video
import skimage.io
import matplotlib.pyplot as plt
from PIL import Image
from helper_methods import csv2list
from sklearn.metrics import mean_squared_error


def sort_spis(sp):
    spp = []
    spb = []
    res = []
    for i in sp:
        spp.append("".join(re.findall(r'\d', i)))
        spb.append("final_img")
    result = [int(item) for item in spp]
    result.sort()

    result1 = [str(item) for item in result]
    for k in range(len(sp)):
        res.append(spb[k] + result1[k] + ".png")
    return res


def generate_video():
    image_folder = r'D:/pythonProject/phase_wm/fold_model_video'  # make sure to use your folder
    video_name = 'need_video.mp4'
    os.chdir(r"D:/pythonProject/phase_wm/fold_model_video")

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".png")]
    sort_name_img = sort_spis(images)
    print(sort_name_img)
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    # fourcc = cv2.VideoWriter_fourcc(*'H264')

    video = cv2.VideoWriter(video_name, -1, 29.97, (width, height))

    cnt = 0
    for image in sort_name_img:
        # if cnt % 300 == 0:
        print(cnt)
        video.write(cv2.imread(os.path.join(image_folder, image)))
        cnt += 1
    cv2.destroyAllWindows()
    video.release()


def read_video(path):
    vidcap = cv2.VideoCapture(path)
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if success:
            cv2.imwrite(r"D:/pythonProject/phase_wm/exper_model\frame%d.png" % count, image)

        print("записан кадр", count)

        if cv2.waitKey(10) == 27:
            break
        count += 1
    return count


import numpy as np
import csv

# var_list = []
# mean_list = []
# for i in range(2048):
#     print(i)
#     img = cv2.imread("D:/phase_wm_graps/BBC/frames_orig_video/frame" + str(int(i)) + ".png")[:, :, 0]
#     mean_list.append(np.mean(img))
#     var_list.append(np.var(img))
#
# with open('RB_variance_0.csv', 'w') as myfile:
#     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
#     wr.writerow(var_list)

var_list = np.array(csv2list("RB_variance_0.csv"))

lmax = np.max(var_list)
diff_list = []
for i in range(1, len(var_list)):
    diff_list.append(abs(var_list[i] - var_list[i - 1]))

diff_list = np.array(diff_list)
diff_list /= lmax

inds = np.argwhere(diff_list > 0.1)
inds += 1

# generate_video()
vid = cv2.VideoCapture(r"cut_RealBarca120.mp4")
# read_video(r"D:/pythonProject/phase_wm/fold_model_video\need_video.mp4")

# s = 0
# svar = 0
# success = True
# for i in range(2048):
#     success, image = vid.read()
#     img_fold = cv2.imread("D:/phase_wm_graps/BBC/frames_orig_video/frame" + str(i) + ".png")
#     # mean_vid = np.mean(image[:,:,0])
#     mean_vid = np.mean(img_fold[:, :, 0])
#     var_vid = np.var(img_fold[:, :, 0])
#     s += mean_vid
#     svar += var_vid
#     print(i, s / (i + 1), svar / (i + 1))
#
# all_mean = s / 2048
# all_svar = svar / 2048
# print("True mean", all_mean)
# print("True var", all_svar)
# for i in range(0,100):
#     imh = skimage.io.imread("fold_model_video/final_img"+str(i)+".png")
#     imh2 =skimage.io.imread("fold_model_video/final_img"+str(i+1)+".png")
#
#     res_img = np.where(imh - imh2>0,imh - imh2,imh2-imh)
#     img1 = Image.fromarray(res_img.astype('uint8'))
#     img1.save(r"D:/pythonProject/phase_wm/exper_model/final_img" + str(i) + ".png")
#     print(i, np.min(res_img),np.max(res_img), res_img[0,-1])

"""
image_orig = skimage.io.imread("D:/pythonProject/phase_wm/embedding_BBC/result0.png")[:,:,0]
graph_orig = plot_ACF(image_orig)

sintez_image = skimage.io.imread("D:/pythonProject/phase_wm/fold_model_video/final_img0.png")
graph_sint = plot_ACF(sintez_image)
acf_list=[]
"""
np.random.seed(42)
rand_list = np.random.choice(1000, 1000, replace=False)
pair_lst = [(rand_list[i - 1], rand_list[i]) for i in range(1, len(rand_list))]
ext_list = [(rand_list[i], rand_list[i - 1]) for i in range(1, len(rand_list))]


#
# pair_lst.extend(ext_list)
# my_square = pair_lst
# #
# graph_video_orig= plot_ACF_video(r"D:/pythonProject/phase_wm/cut_RealBarca120.mp4", my_square,2048)
# # graph_video_synt= plot_ACF_video(r"D:/pythonProject/phase_wm/fold_model_video/need_video82_16384.mp4", my_square,4096)
# graph_video_synt2= plot_ACF_video(r"D:/pythonProject/phase_wm/fold_model_video/need_video28_16384_6363.mp4", my_square,16384)

# plt.plot(graph_video_orig, label="Original video ACF")


def check_exp(row):
    ratio = []
    for i in range(1, len(row)):
        ratio.append(row[i] / row[i - 1])

    return ratio


# for i in range(50,60):
#     mosaic = FieldGenerator.draw_mosaic_field(5, 0.9959, 1080, 1920, 127, 2100, 4+i)
#
#     img1 = Image.fromarray(mosaic.astype('uint8'))
#     img1.save(r"D:/pythonProject/phase_wm/mosaics/mosaic" + str(i) + ".png")
# ratio = check_exp(graph_video_orig)
# print("mean 200-400",np.mean(ratio[200:400]))
# ratio_syntnes = check_exp(graph_video_synt2)
# # ratio_syntnes2 = check_exp(graph_video_synt2)
# # print("Two means",np.mean(ratio_syntnes2[20:100]))
# plt.plot(ratio[:400],label = "orig video - ACF ")
# plt.plot(ratio_syntnes[:400], label = "Synthesis image(n=28) - ACF ")
# # plt.plot(ratio_syntnes2[:100], label = "Synthesis image(n=34) - ACF ")
# plt.title("B(n+1)/B(n)")
# plt.legend()
# plt.show()


#
# mosaic = FieldGenerator.draw_mosaic_field(5, 0.996, 2048, 2048, 47, 2254, 0)
# plot_ACF(mosaic)
# # img1 = Image.fromarray(mosaic.astype('uint8'))
# # img1.save(r"D:/pythonProject/phase_wm/mosaic_5_line" + str(0.996) + ".png")
# print("----------------------------------------------------------")
# exp_read = skimage.io.imread(r"D:/pythonProject/phase_wm/mosaic_5_line" + str(0.996) + ".png")
# plot_ACF(exp_read)

img_class = skimage.io.imread(r"D:\pythonProject\phase_wm\extract/frame200.png")
img_old = skimage.io.imread(r"D:\pythonProject\phase_wm\old_experiments\extract/frame200.png")
compr = img_class==img_old
# var_list = csv2list(r"D:/pythonProject/phase_wm/change_sc.csv")
matr = np.zeros((2048, 1920))
inds = np.ravel(inds)
inds = np.insert(inds, 0, 0)
for i,el in enumerate(inds[:-1]):

    img = skimage.io.imread("D:/phase_wm_graps/BBC/frames_orig_video/frame" + str(int(i)) + ".png")[:, :, 0]
    acf = plot_ACF(img)
    for k in range(el,inds[i+1]):
        matr[k, :] = acf

    # if i == 0:
    #     plt.plot(acf, label="ACF of image" + str(int(i)))

# plt.legend()
# plt.show()

matr_mean = np.mean(matr, axis=0)
np.save("mean ACF",matr_mean)
plt.plot(matr_mean)
plt.show()

"""
for my_ro in np.arange(0.9921,0.9925,0.0001):
    mosaic = FieldGenerator.draw_mosaic_field(5, my_ro, 2048, 2048, 76, 2254,0)
    img1 = Image.fromarray(mosaic.astype('uint8'))
    img1.save(r"D:/pythonProject/phase_wm/mosaic_5_line" + str(my_ro) + ".png")
    msc = plot_ACF(mosaic)
    curr_mse = mean_squared_error(graph_orig[:100],msc[:100])
    print("po=",my_ro,"mean_squared_error",curr_mse)
    acf_list.append(curr_mse)
    plt.plot(msc[:200], label="Mosaic ACF. po = " + str(my_ro))

plt.plot(graph_orig[:200], label="Orig ACF")
plt.legend()
plt.show()

# im1 = skimage.io.imread("D:\pythonProject\phase_wm/mosaic0.png")
# im2 = skimage.io.imread("D:\pythonProject\phase_wm/mosaic_765.png")
# print(plot_ACF(im1))
# print(plot_ACF(im2))
"""

img = skimage.io.imread("D:/pythonProject/phase_wm/fold_model_video/final_img0.png")
acf_img = plot_ACF(img)

plt.plot(acf_img[:200], label="Original image ACF")
plt.title("Spatial")
plt.legend()
plt.show()

SIZE = 2048
SIZE_HALF = 1024
hc_const = 2363.05325519

alf = 0.1
betta = 0.1

txt_field_full = np.zeros((50, 20))

for i in range(50):
    # for p in np.arange(0.1,0.96,0.1):
    need_params2 = (0.5, alf, betta)
    count_of_frame = 3000
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

    texture = gener_field(list_ACF2, i)
    txt_field_full[i, :] = plot_ACF(texture)[:20]
    img1 = Image.fromarray(texture.astype('uint8'))
    img1.save(r"D:/pythonProject/phase_wm/fields/field" + str(i) + ".png")
    # txt = plot_ACF(texture)
    # print("alf=",alf,"bett=",betta,"VARIANCe of texture", txt[0])
    # plt.plot(txt[:20],label="alf="+str(alf) + "bett=" + str(betta))

avg_texture = np.mean(txt_field_full, axis=0)
plt.plot(avg_texture)
plt.legend()

plt.show()
