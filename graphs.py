from matplotlib import pyplot as plt
import numpy as np


fig, ax = plt.subplots()

# s=bytes([bytes1[7]])

font = {
    'size': 12, }


sp1=[]
for i in range(0,14):
    sp1.append(0)
sp1.append(1)
sp1.append(1)
sp1.append(1)
sp1.append(0)
# for i in range(0,4):
#     sp1.append(0)
for i in range(0,12):
    sp1.append(1)


# sp1.append(0)
# sp1.append(1)


# for i in range(0,15):
#     sp1.append(1)

sp2=[]
for i in range(0,27):
    sp2.append(0)

sp2.append(1)
sp2.append(0)
sp2.append(1)
# for i in range(0,8):
#     sp2.append(1)

print(sp2)
sp3=[]
for i in range(0,23):
    sp3.append(0)

for i in range(0,7):
    sp3.append(1)

# sp4=[]
# for i in range(0,6):
#     sp4.append(0)
# for i in range(0,24):
#     sp4.append(1)

# sp5=[]
# for i in range(0, 21):
#     sp5.append(0)
# for i in range(0, 9):
#     sp5.append(1)


ax.plot([i for i in range(96, 3000, 100)],
[0.6836258048226234, 0.8250220931700543, 0.8276732735765686, 0.8021714429996213, 0.8030551698017927, 0.8327231410175483, 0.8647897992677692, 0.8352480747380381, 0.8472415099103648, 0.8533013508395405, 0.8322181542734504, 0.8669359929301855, 0.8565837646761773, 0.8931953036232799, 0.9125110465850271, 0.8967302108319657, 0.860623658628961, 0.9032950385052393, 0.8938265370534023, 0.8679459664183815, 0.8505239237470016, 0.9036737785633128, 0.9046837520515086, 0.9166771872238354, 0.9226107814669865, 0.9489963388461052, 0.9472288852417624, 0.9447039515212725, 0.9257669486175988, 0.9699532887261709]
        , label='Bitrate = 5M')

ax.plot([i for i in range(96, 3000, 100)],
[0.6341371039010226, 0.743340487312208, 0.7337457391743467, 0.7321045322560282, 0.727685898245171, 0.7478853680090898, 0.7834869334679965, 0.7723772250978412, 0.767706097714935, 0.7323570256280773, 0.7651811639944451, 0.7622774902158819, 0.7439717207423305, 0.7952278752682742, 0.813912384799899, 0.8058325968943315, 0.8140386314859235, 0.7972478222446661, 0.7861381138745108, 0.7932079282918824, 0.7678323444009595, 0.7998990026511804, 0.8158060850902664, 0.8266633000883726, 0.8091150107309684, 0.8521651306653201, 0.8413079156672137, 0.8777932079282919, 0.8669359929301855, 0.8894079030425451]
        , label='Bitrate = 3.5M')

ax.plot([i for i in range(96, 3000, 100)],
[0.5889407903042545, 0.6401969448301982, 0.6451205655851534, 0.6528216134326474, 0.64221689180659, 0.6510541598283045, 0.6607751546521904, 0.6525691200605984, 0.6554727938391617, 0.6564827673273577, 0.6564827673273577, 0.6662037621512436, 0.6477717459916678, 0.6779447039515213, 0.6888019189496276, 0.6944830198207297, 0.6997853806337584, 0.6995328872617094, 0.6881706855195051, 0.673778563312713, 0.6696124226739049, 0.6900643858098725, 0.6856457517990153, 0.6972604469132685, 0.712536295922232, 0.6961242267390481, 0.7170811766191137, 0.7499053149854816, 0.7279383916172201, 0.7637924504481758]
        , label='Bitrate = 2M')

ax.step([i for i in range(96, 3000, 100)],
sp1
        , label='Bitrate = 5M')

ax.step([i for i in range(96, 3000, 100)],
sp2
        , label='Bitrate = 3.5M')

ax.step([i for i in range(96, 3000, 100)],
[0]*30
        , label='Bitrate = 2M')


# ax.step([i for i in range(96, 3000, 100)],
# sp3
#         , label='Start frame = 20')
#
# ax.step([i for i in range(96, 3000, 100)],
# sp4
# , label='Start frame = 30')

# ax.step([i for i in range(96, 3000, 100)],
# sp5
# , label='Variance of noise = 9')


plt.title("RealBarca. A=2 ", fontdict=font)
plt.xlabel("Number of frame", fontdict=font)
plt.ylabel("Successful extraction", fontdict=font)
plt.yticks([0.7, 0.8, 0.9,  1], size=11)
plt.xticks(np.arange(0, 3000, 200), size=11)
plt.legend()

plt.show()


# ax.plot([i for i in range(2, 6)],
# [0,0,0,0]
#         , label='Version L')
# ax.plot([i for i in range(2, 6)],
# [0.2,0,0,0]
#         , label='Version M')
# ax.plot([i for i in range(2, 6)],
# [0.95,0.49,0.01,0]
#         , label='Version Q')
# ax.plot([i for i in range(2,6)],
# [1,1,0.94,0.04]
#         )
#
# plt.title("Reed-Solomon code. Percentage of successful extractions ", fontdict=font)
# plt.xlabel("Percentage of damaged bits", fontdict=font)
# plt.ylabel("Percentage of successful extractions", fontdict=font)
# # plt.yticks([0.7, 0.8, 0.9,  1], size=11)
# # plt.xticks(np.arange(0, 3000, 200), size=11)
# plt.legend()
#
# plt.show()