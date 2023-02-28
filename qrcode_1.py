import numpy as np
import qrcode
import cv2
import os
from skimage import io
from PIL import Image
import pyzbar.pyzbar as pyzbar

size_quadr = 16


def big2small(st_qr):
    qr = np.zeros((89, 89))

    for i in range(0, 1424, size_quadr):
        for j in range(0, 1424, size_quadr):
            q = np.mean(st_qr[0:15, 0:15])
            q = np.mean(st_qr[i:i + size_quadr, j:j + size_quadr])
            qr[int(i / size_quadr), int(j / size_quadr)] = int(np.mean(st_qr[i:i + size_quadr, j:j + size_quadr]))

    return qr


def small2big(sm_qr):
    qr = np.zeros((1424, 1424))

    for i in range(0, 89):
        for j in range(0, 89):
            tmp = sm_qr[i, j]
            qr[i * 16:i * 16 + 16, j * 16:j * 16 + 16].fill(tmp)

    return qr


def correct_qr(damage_img):
    N = 7
    for i in range(0, N):
        if i % 2 == 0 or i == 6 - i:
            damage_img[i:N - i, i:N - i] = 0
        else:
            damage_img[i:N - i, i:N - i] = 255
    damage_img[N:N + 1, 0:(N + 1)] = 255
    damage_img[0:(N + 1), N:N + 1] = 255

    for i in range(0, N, 1):
        if i % 2 == 0 or i == 6 - i:
            damage_img[(89 - N + i):(89 - i), i:N - i] = 0
        else:
            damage_img[(89 - N + i):(89 - i), i:N - i] = 255
    damage_img[(89 - N - 1):(89 - N - 1) + 1, 0:(N + 1)] = 255
    damage_img[(89 - N):(89), (N):(N) + 1] = 255

    for i in range(0, N, 1):
        if i % 2 == 0 or i == 6 - i:
            damage_img[i:N - i, (89 - N + i):(89 - i)] = 0
        else:
            damage_img[i:N - i, (89 - N + i):(89 - i)] = 255
    damage_img[0:N + 1, (89 - N - 1):(89 - N - 1) + 1] = 255
    damage_img[(N - 1) + 1:N + 1, (89 - N - 1):89 + 1] = 255

    damage_img[78:81, 0:6] = 255
    damage_img[78:81, 0] = 0
    damage_img[79, 1] = 0
    damage_img[79, 4:6] = 0
    damage_img[78, 3] = 0
    damage_img[80, 3] = 0
    # damage_img[78, 7] = 0
    damage_img[80, 3] = 0
    damage_img[78, 6] = 0
    damage_img[80, 6] = 0

    damage_img[0:6, 78:81] = 255
    damage_img[0, 78:81] = 0
    damage_img[1, 79] = 0
    damage_img[4:6, 79] = 0
    # damage_img[6, 78] = 0
    # damage_img[6, 80] = 0
    damage_img[80:85,80:85]=0
    damage_img[-8:-5, -8:-5] = 255
    damage_img[-7, -7] = 0

    for i in range(8,81,2):
        damage_img[6,i]=0
        damage_img[6, i+1] = 255

    for i in range(8,81,2):
        damage_img[i,6]=0
        damage_img[i+1,6] = 255

    damage_img[8, 82:89] = 255
    damage_img[8, 84] = 0
    damage_img[8, 82] = 0

    damage_img[82:89, 8] = 255
    damage_img[85:87, 8] = 0
    damage_img[82, 8] = 0

    damage_img[:9, 8] = 255
    damage_img[8, :9] = 255
    damage_img[8, 2:4] = 0
    damage_img[8, 6:8] = 0
    damage_img[4, 8] = 0
    damage_img[6:9, 8] = 0


    return damage_img


def correct_qr_invert(damage_img):
    N = 7
    for i in range(0, N):
        if i % 2 == 0 or i == 6 - i:
            damage_img[i:N - i, i:N - i] = 255
        else:
            damage_img[i:N - i, i:N - i] = 0
    damage_img[N:N + 1, 0:(N + 1)] = 0
    damage_img[0:(N + 1), N:N + 1] = 0

    for i in range(0, N, 1):
        if i % 2 == 0 or i == 6 - i:
            damage_img[(89 - N + i):(89 - i), i:N - i] = 255
        else:
            damage_img[(89 - N + i):(89 - i), i:N - i] = 0
    damage_img[(89 - N - 1):(89 - N - 1) + 1, 0:(N + 1)] = 0
    damage_img[(89 - N):(89), (N):(N) + 1] = 0

    for i in range(0, N, 1):
        if i % 2 == 0 or i == 6 - i:
            damage_img[i:N - i, (89 - N + i):(89 - i)] = 255
        else:
            damage_img[i:N - i, (89 - N + i):(89 - i)] = 0
    damage_img[0:(N) + 1, (89 - N - 1):(89 - N - 1) + 1] = 0
    damage_img[(N - 1) + 1:(N) + 1, (89 - N - 1):89 + 1] = 0



    return damage_img


def gener_qr(txt):
    qr = qrcode.QRCode(
        version=18,
        error_correction=qrcode.constants.ERROR_CORRECT_Q,
        box_size=16,
        border=0,
    )

    qr.add_data(txt)

    qr.make(fit=True)

    img = (qr.make_image(fill_color="black", back_color="white"))
    img = np.where(img, 255, 0)
    print(img.shape)

    img2 = Image.fromarray(img.astype("uint8"))
    img2.save(r"C:\Users\user\PycharmProjects\phase_wm\qr_ver18_Q.png")
    print("QR was generated")


def read_qr(image):
    # image1 =big2small(image)

    result = pyzbar.decode(image)
    if not result:
        return ""
    else:

        return result[0].data.decode("utf-8")


#
# my_img=np.full((89,89),128)
# correct_qr(my_img)
#
# img2=Image.fromarray(my_img.astype("uint8"))
# img2.save("11qr_ver18_L.png")

# list_char=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n' ,'o', 'p' ,'q' ,'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# for i in list_char:
#     gener_qr(3*'https://ru.wikipedia.org/wiki/QR-%D0%BA%D0%BE%D0%B4#/media/%D0%A4%D0%B0%D0%B9%D0%BB:QRCode-2-Structure.png'+i,i)

# images = [img for img in os.listdir("C:/Users/user/PycharmProjects/phase_wm/test_gener_qr")
#               if img.endswith(".png")]
#
# must_for_qr= io.imread("C:/Users/user/PycharmProjects/phase_wm/test_gener_qr/my_qra.png")
# tmp= np.ones((1424,1424))
# for i in images:
#     img = io.imread("C:/Users/user/PycharmProjects/phase_wm/test_gener_qr/" + i)
#     must_for_qr=np.where(img!=255,must_for_qr,np.where(tmp!=255,must_for_qr,np.where(must_for_qr==255,255,127)))
#     must_for_qr = np.where(img != 0, must_for_qr, np.where(tmp != 0, must_for_qr, np.where(must_for_qr==0,0,127)))
#     # must_for_qr[img == 0] = 100
#     tmp = img
#
# img = Image.fromarray(must_for_qr.astype('uint8'))
# img.save(r"C:/Users/user/PycharmProjects/phase_wm/test_gener_qr/final_qr.png")
# print(must_for_qr)
# filename = r"C:\Users\user\PycharmProjects\phase_wm\extract\after_normal_phas_bin\result1620.png"
#
# #print(read_qr(filename))

# gener_qr(9*"Let;s go to reading QR-code must have")

# stop_kadr1_bin=[]
must_qr = io.imread(r'C:\Users\user\PycharmProjects\phase_wm\qr_ver18_H.png')
comp_qr = np.copy(must_qr)
comp_qr2= comp_qr
# print(read_qr(comp_qr))
# must_qr[50:150,800:900]=128
# must_qr[50:150,400:500]=128
# must_qr[100:1350,96:116]=128
# must_qr[96:116,100:1450] = 128
# must_qr[100:1350,470:499]=128
# must_qr[100:1350,870:899]=128
# must_qr[1200:1450,1300:1399]=128
# must_qr[200:300,0:199]=128


for i in range(1000,1060):
    np.random.seed(i)
    a=np.random.randint(0,89)*16
    np.random.seed(i * 10)
    b = np.random.randint(0, 89) * 16
    print(a,b)
    must_qr[a:a + 16, b:b + 16][must_qr[a:a+16,b:b+16] > 252] = 128
    must_qr[a:a + 16, b:b + 16][must_qr[a:a + 16, b:b + 16] == 0] = 129
# must_qr[0:200,200:299]=128
# must_qr[110:300,1300:1443]=128
img2 = Image.fromarray(must_qr.astype("uint8"))
img2.save(r"C:\Users\user\PycharmProjects\phase_wm\test_gener_qr\11.png")

# for cnt in range(96,3000,100):
#     if cnt % 100 == 96:
#         print(read_qr(
#             small2big(correct_qr(io.imread(r"C:\Users\user\PycharmProjects\phase_wm\extract/after_normal_phas_bin/result" + str(
#                 cnt) + ".png")))),cnt)
small_qr = correct_qr(big2small(must_qr))
c=((small_qr==big2small(comp_qr)))
print(c.size-np.count_nonzero(c))

quiet= np.zeros((1500,1550))
quiet[quiet==0]=255
tmp= small2big(small_qr)
quiet[38:1462,38:1462]= tmp
img2 = Image.fromarray(quiet.astype("uint8"))
img2.save(r"C:\Users\user\PycharmProjects\phase_wm\test_gener_qr\12.png")
print("------------------------------------")
print(read_qr(tmp))
