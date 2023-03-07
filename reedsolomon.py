import numpy as np
from reedsolo import RSCodec
from PIL import Image
from helper_methods import small2big, big2small
from skimage import io


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
    print(rsc.nsym/len(mes))
    small_matrix = np.resize(binstr, (89, 89))
    matr4embed = small2big(small_matrix)
    matr4embed[matr4embed == 1] = 255
    # print(s)
    img = Image.fromarray(matr4embed.astype('uint8'))
    img.convert('RGB').save(r"C:\Users\user\PycharmProjects\phase_wm\RS_cod89x89.png")


# The input should be a binary image
def extract_RS(image_RS, rsc):
    final_extract = b""
    # print(len(extract_txt),len(rmesecc))
    # final_extract_b=bytearray(final_extract,'utf-8')
    # matrbin = big2small(image_RS)
    matrbin = image_RS
    matrbin[matrbin == 255] = 1
    listbin = np.reshape(matrbin, (89*89))
    listbin = listbin.astype(int)
    for i in range(0,len(listbin)-(89*89-7744),8):
        tmp = ''.join(str(x) for x in listbin[i:i+8])
        ch = bytes([(int(tmp, 2))])
        final_extract += ch

    # print(final_extract,len(final_extract))
    try:
        rmes1, rmesecc1, errata_pos1 = rsc.decode(final_extract)
        print(rmes1)
        print(list(errata_pos1))
    except:
        print("Error of decoder")


rsc = RSCodec(nsym=120, nsize=121)
mes = 1*b'Correct1'
create_RS_cod(mes, rsc)
# extr_RS=io.imread(r"C:\Users\user\PycharmProjects\phase_wm\RS_cod89x89.png")
# extr_RS[0:1040,0:512]=0
# extract_RS(extr_RS,rsc)
