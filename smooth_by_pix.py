import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from statistics import mean


def disp_pix(coord_x,coord_y,path,kef_avg):
    cnt=0
    count=3000
    list_diff=[]
    while cnt < count:

        if cnt>0:
            tmp = np.copy(arr)
        arr = io.imread(path + r"\frame" + str(cnt) + ".png")[coord_x, coord_y, 0]

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
        if abs(list_diff[i] - list_diff[i-1]) > kef_avg*mean_diff:
           list_big_diap.append(i)


    print(len(list_big_diap))
    return list_big_diap

#disp_list = [0, 77, 244, 396, 736, 1243, 1392, 1468, 2244, 2465, 3000]
def exp_smooth(coord_x,coord_y,path, alf,kef):
    disp_list= disp_pix(coord_x,coord_y,r"C:\Users\user\PycharmProjects\phase_wm\extract",kef)
    disp_list.insert(0,0)
    disp_list.append(3000)

    print(disp_list)
    cnt = 0
    count = 3000

    diff_val=np.array([])
    pix_val = np.array([])
    smooth_val = np.array([])
    arr_copy = np.asarray([])

    for scene in range(1,len(disp_list)):
        cnt = disp_list[scene-1]
        while cnt < disp_list[scene]:

            arr = io.imread(path + r'\frame' + str(cnt) + ".png")[coord_x,coord_y,0]
            pix_val = np.append(pix_val, arr)
            img_1_step = arr_copy
            if cnt == disp_list[scene-1]:
                arr_copy = arr.copy()
                smooth_val = np.append(smooth_val, arr_copy)
                diff_val=np.append(diff_val,arr_copy-arr)

            else:
                arr_copy = np.float32(img_1_step) * alf + np.float32(arr) * (1 - alf)
                smooth_val=np.append(smooth_val,arr_copy)
                diff_val=np.append(diff_val,arr_copy-arr)

            print("tmp kadr", cnt)
            cnt += 1
        print(len(diff_val))
    return pix_val,smooth_val

def exp_smooth1(coord_x,coord_y,path, alf,kef):
    disp_list= disp_pix(coord_x,coord_y,r"C:\Users\user\PycharmProjects\phase_wm\extract",kef)
    disp_list.insert(0,0)
    disp_list.append(3000)

    print(disp_list)
    cnt = 0
    count = 3000

    diff_val=np.array([])
    pix_val = np.array([])
    smooth_val = np.array([])
    arr_copy = np.asarray([])

    while cnt < count:

        arr = io.imread(path + r'\frame' + str(cnt) + ".png")[coord_x,coord_y,0]
        pix_val = np.append(pix_val, arr)
        img_1_step = arr_copy
        if cnt == 0:
            arr_copy = arr.copy()
            smooth_val = np.append(smooth_val, arr_copy)
            diff_val=np.append(diff_val,arr_copy-arr)

        else:
            arr_copy = np.float32(img_1_step) * alf + np.float32(arr) * (1 - alf)
            smooth_val=np.append(smooth_val,arr_copy)
            diff_val=np.append(diff_val,arr_copy-arr)


        print("tmp kadr", cnt)
        cnt += 1
    print(len(diff_val))
    return smooth_val

"""
fig, axes = plt.subplots()

nl1= exp_smooth1(0,0,r"C:/Users/user\PycharmProjects\phase_wm\extract",0.53,2)
nl3,nl4=exp_smooth(0,0,r"C:/Users/user\PycharmProjects\phase_wm\extract",0.53,2)
print(nl1)
print(nl3)
print(nl4)


axes.plot([i for i in np.arange(0,3000)],nl1, label = "обычное сгл")
axes.plot([i for i in np.arange(0,3000)],nl3, label = "без сгл")
axes.plot([i for i in np.arange(0,3000)],nl4, label = "сгл со сцен")


plt.legend()
plt.show()
"""

tmp=np.ones(3000)

for cnt in range(0,3000):
    arr = io.imread(r'C:\Users\user\PycharmProjects\phase_wm\extract\first_smooth/result' + str(cnt) + '.png')[0,0,0]
    arr1 = io.imread(r'C:\Users\user\PycharmProjects\phase_wm\extract\first_smooth_by_scene/result' + str(cnt) + '.png')[0,0,0]
    tmp[cnt] = int(arr)-int(arr1)

for i in tmp:
    if i!=0:
        print(i)
