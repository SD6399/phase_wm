from reedsolo import RSCodec,rs_encode_msg,init_tables

# init_tables(0x11d)
# nk = 2
# packet= rs_encode_msg('123456789', nk)


rsc = RSCodec(nsym=25,nsize=255)  # 10 ecc symbols

mes = 11 * b'Correct extraction'
tmp = rsc.encode(mes)
print(tmp)
tmp[72] = 5
tmp[4] = 5
tmp[5] = 5
tmp[7] = 5
tmp[6] = 5

tmp[-1] = 1
rmes, rmesecc, errata_pos = rsc.decode(tmp)
print(rsc.check(tmp))
print(rsc.check(rmesecc))
print(len(tmp), tmp)
print(rmesecc)
print(rmes)
bytes1=bytes(rmesecc)
bin_str=(bin(int(rmesecc.hex(), 16)))
print(len(bin_str))
#
listbin=[bin(byte) for byte in bytes1]
binaryString = bin_str[2:]

# for el in listbin:
#     integer = int(el, 2)
#     character = chr(integer)
#
#
size = 8
bValues = [binaryString[i:i+size] for i in range(0, len(binaryString), size)]
string = ""
for bValue in bValues:
    integer = int(bValue, 2)
    character = chr(integer)
    print(character)
    string += character


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
