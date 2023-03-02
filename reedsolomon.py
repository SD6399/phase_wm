import numpy as np
from reedsolo import RSCodec, rs_encode_msg, init_tables
from PIL import Image
from qrcode_1 import small2big, big2small
from skimage import io

def create_RS_cod(mes, rsc):
    tmp = rsc.encode(mes)

    rmes, rmesecc, errata_pos = rsc.decode(tmp)

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

    small_matrix = np.resize(binstr, (89, 89))
    matr4embed = small2big(small_matrix)
    matr4embed[matr4embed == 1] = 255
    print(s)
    img = Image.fromarray(matr4embed.astype('uint8'))
    img.convert('RGB').save(r"C:\Users\user\PycharmProjects\phase_wm\RS_cod89x89.png")


# The input should be a binary image
def extract_RS(image_RS, rsc):
    final_extract = b""
    # print(len(extract_txt),len(rmesecc))
    # final_extract_b=bytearray(final_extract,'utf-8')
    matrbin = big2small(image_RS)
    matrbin[matrbin == 255] = 1
    listbin = np.reshape(matrbin, (89*89))
    listbin = listbin.astype(int)
    for i in range(0,len(listbin)-8,8):
        tmp = ''.join(str(x) for x in listbin[i:i+8])
        ch = bytes([(int(tmp, 2))])
        final_extract += ch

    # print(final_extract,len(final_extract))
    rmes1, rmesecc1, errata_pos1 = rsc.decode(final_extract)
    print(rmes1)
    print(len(mes))
    print(rsc.nsym / 2 / len(mes))


rsc = RSCodec(nsym=28, nsize=31)
mes = 2 * b'Correct extraction of watermark from this video'
# create_RS_cod(mes, rsc)
extr_RS=io.imread(r"C:\Users\user\PycharmProjects\phase_wm\RS_cod89x89.png")
extr_RS[7:10,111:120]=126
extract_RS(extr_RS,rsc)

# size = 8
# bValues = [binaryString[i:i+size] for i in range(0, len(binaryString), size)]
# string = ""
# for bValue in bValues:
#     integer = int(bValue, 2)
#     character = chr(integer)
#     print(character)
#     string += character


# import pyreedsolomon
#
# import numpy as np
#
# rs_dr = pyreedsolomon.Reed_Solomon(8,223,255,0x11D,0,1,32)
#
# data = np.random.randint(0,256,150).astype(np.uint8)
#
# data_enc = rs_dr.encode(data)
#
# # create a few errors
# err_idx = [23,53,12,97,102]
#
#
# data_enc[err_idx] = 255
#
# data_dec, n_errors = rs_dr.decode(data_enc)
#
# verify = np.all(data_dec==data)
#
# print(f"Decoding succes: {verify}. errors corrected {n_errors}")
