import cv2
from multiprocessing import Pool


def f(x):
    return x*x



def avg(lst):
    return sum(lst)/len(lst)


def read_video(path,coord_x,coord_y):
    vidcap = cv2.VideoCapture(path)
    count = 0
    list00=[]

    dsp00=[]

    temp00=[]

    success = True
    while success:
        success, image = vidcap.read()
        if success:
            if count != 0:
                temp00.append(image[coord_x,coord_y,0]*p00)
            p00= int(image[coord_x,coord_y,0])
            list00.append(p00)
            dsp00.append(p00*p00)

        print("записан кадр", count)

        if cv2.waitKey(10) == 27:
            break
        count += 1
    mog00 = avg(list00)

    avg00_2 = avg(dsp00)
    av_2_00 = avg(temp00)
    print("MO", mog00)
    print("MO^2", avg00_2, )
    print("Temporary", av_2_00, )
    print("Variance", avg00_2-mog00*mog00)
    print("ACF", av_2_00 - mog00*mog00)

    return mog00,avg00_2-mog00*mog00, av_2_00 - mog00*mog00


if __name__ == '__main__':
    pool = Pool()

    # Список параметров для вызова функции
    parameters = [("Road.mp4", 0,0), ("Road.mp4", 500,500), ("Road.mp4", 1000,1000)]

    results = pool.starmap(read_video, parameters)
    # print(p.map(read_video, ["Road.mp4",0,0]))
    print(results)
    # read_video("Road.mp4",0,0)
    # read_video("Road.mp4",500,500)