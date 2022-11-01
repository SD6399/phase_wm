import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from statistics import mean


def disp_pix(coord_x,coord_y,path):
    cnt=0
    count=3000
    list_diff=[]
    while cnt < count:
        if cnt>0:
            tmp = np.copy(arr)
        arr = io.imread(path+r"\frame" + str(cnt) + ".png")[coord_x,coord_y,0]

        if cnt == 0:
            list_diff.append(0)

        else:
            diff_img = np.abs(int(arr) - int(tmp))
            #print(diff_img, " frame ", cnt)
            list_diff.append(np.mean(diff_img))

        cnt+=1

    mean_diff=(mean(list_diff))
    #for i in list_diff:
    list_big_diap=[]
    length=1
    for i in range(len(list_diff)):
        if (list_diff[i] and list_diff[i-1]) < 2*mean_diff:
            length+=1
            if i==2999:
                if length > 50:
                    list_big_diap.append((i-length+1,i))

        else:
            if length>50:
                list_big_diap.append((i-length+1,i))
            length = 0

    return list_big_diap


def exp_smooth(coord_x,coord_y,path, alf):
    disp_list= disp_pix(coord_x,coord_y,r"C:\Users\user\PycharmProjects\phase_wm\extract")
    print(disp_list)
    cnt = 0
    count = 3000

    arr_copy = np.asarray([])

    for scene in range(len(disp_list)):
        cnt = disp_list[scene][0]
        while cnt < disp_list[scene][1]:

            arr = io.imread(path + r'\frame' + str(cnt) + ".png")[coord_x,coord_y,0]

            img_1_step = arr_copy
            if cnt == 0:
                arr_copy = arr.copy()

            else:
                arr_copy = np.float32(img_1_step) * alf + np.float32(arr) * (1 - alf)
                print(arr,arr_copy)


            print("tmp kadr", cnt)
            cnt += 1
    return arr_copy


print(exp_smooth(0,0,r"C:\Users\user\PycharmProjects\phase_wm\extract",0.93))
#print(disp_pix(0,0,r"C:\Users\user\PycharmProjects\phase_wm\extract"))
# print(disp_pix(111,111,r"C:\Users\user\PycharmProjects\phase_wm\extract"))