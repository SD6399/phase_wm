import numpy as np
from reedsolo import RSCodec
from PIL import Image
from helper_methods import small2big, big2small
from skimage import io
from itertools import combinations
import random
from random import sample
from collections import Counter


def create_RS_cod(mes, rsc):
    tmp = rsc.encode(mes)

    rmes, rmesecc, errata_pos = rsc.decode(tmp)
    print(rmesecc)
    extract_txt_1 = []

    for i in range(len(rmesecc)):
        extract_txt_1.append((rmesecc[i]))

    listbin = [bin(byte)[2:] for byte in extract_txt_1]
    s = 0
    binstr = np.array([])
    for i in range(len(listbin)):
        while len(listbin[i]) < 8:
            listbin[i] = '0' + listbin[i]
        s += len(listbin[i])
        binstr = np.append(binstr, list(listbin[i]))

    binstr = binstr.astype(int)
    while len(binstr) < 89 * 89:
        binstr = np.append(binstr, 0)

    print(s)
    print(len(mes))
    print(rsc.nsym / len(mes))
    small_matrix = np.resize(binstr, (89, 89))
    matr4embed = small2big(small_matrix)
    matr4embed[matr4embed == 1] = 255
    # print(s)
    img = Image.fromarray(matr4embed.astype('uint8'))
    img.convert('RGB').save(r"C:\Users\user\PycharmProjects\phase_wm\RS_cod89x89.png")

    return s


# The input should be a binary image
def extract_RS(image_RS, rsc, count):
    final_extract = b""
    # print(len(extract_txt),len(rmesecc))
    # final_extract_b=bytearray(final_extract,'utf-8')
    # matrbin = big2small(image_RS)
    matrbin = image_RS
    matrbin[matrbin == 255] = 1
    listbin = np.reshape(matrbin, (89 * 89))
    listbin = listbin.astype(int)
    data_length= len(listbin) - (89 * 89 - count)
    mas_symb=[]
    for i in range(0, data_length, 8):
        tmp = ''.join(str(x) for x in listbin[i:i + 8])
        ch = bytes([(int(tmp, 2))])
        mas_symb.append(ch)

    for i in range(0,int(len(mas_symb)/7)):
        voit = []

        for j in range(i, i+len(mas_symb), int(len(mas_symb) / 7)):
            voit.append(mas_symb[j])
        mpc = Counter(voit).most_common(1)[0][0]
        for j in range(i, i+ len(mas_symb), int(len(mas_symb) / 7)):
            mas_symb[j] = mpc

    for ch in mas_symb:
        final_extract += ch
    # print(final_extract,len(final_extract))
    try:
        # print(final_extract)
        rmes1, rmesecc1, errata_pos1 = rsc.decode(final_extract)
        print(rmes1)

    except:
        rmes1 = ''
        print("Error of decoder")

    return rmes1


rsc = RSCodec(nsym=106, nsize=127)
mes = 7*b'Correct extraction of'
print(len(mes))
Nbit = create_RS_cod(mes, rsc)

# extr_RS = io.imread(r"C:/Users/user/PycharmProjects\phase_wm\RS_cod89x89.png")
# left = 0
# right = 3488
# n = 89
# count = 0
# for sid in range(0, 100, 1):
#     extr_R = io.imread(r"C:/Users/user/PycharmProjects\phase_wm\RS_cod89x89.png")
#     random.seed(sid)
#     sampl = sample(list(combinations(range(0, n), 2)), right)
#     # print(sampl)
#     for i in range(left, right):
#
#         extr_R[sampl[i][0] * 16:sampl[i][0] * 16 + 16, sampl[i][1] * 16:sampl[i][1] * 16 + 16] = np.where(
#             extr_R[sampl[i][0] * 16:sampl[i][0] * 16 + 16, sampl[i][1] * 16:sampl[i][1] * 16 + 16] == 255, 0, 255)
#
#     # extr_RS[0:1040,0:16*16]=0
#     comp=extr_RS==extr_R
#     at=(np.sum(comp==True))
#     # print(at/extr_R.size)
#     # print((right - left) / (89 * 89))
#     if (extract_RS(extr_R, rsc, Nbit)) != '':
#         count += 1
#
# print(count)
