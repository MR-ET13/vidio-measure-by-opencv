import cv2
import numpy as np

import HanShu
from HanShu import Get_Feature_Points, ROI, Point_Distance
import matplotlib.pyplot as plt
import numpy.fft as nf
from scipy.spatial import distance as dist

fname = 'Pictures_File/Triangular_profile/4_16_1/v0_120fps_1.MOV'


def Video_test():
    """
    振动测量
    :return: 无
    """
    # 问题帧+1
    test_f = 1
    vc = cv2.VideoCapture(fname)
    ret, frame = vc.read()
    i_roi = ROI(frame.copy())
    i_area = HanShu.Area_test(fname)
    Point_all = []
    # Point0 = Get_Feature_Points(frame, i_roi, num_f=0, area_test=False, AREA=i_area)
    Point0 = HanShu.Get_Point0(
        fname, i_roi, num_f=0, area_test=False, AREA=i_area, n=500)
    Point_all.append(Point0)
    Data = []
    Data1 = []
    Data2 = []
    num_f = 0
    select = 1
    while True:
        ret, frame = vc.read()
        num_f = num_f + 1
        if num_f == test_f:
            print('Point0: ', Point0)
        if frame is None:
            break
        if ret == True:
            Point = Get_Feature_Points(
                frame, i_roi, num_f, area_test=False, AREA=i_area)[select]
            Point_all.append(Point)
            dis = Point_Distance(Point0[select], Point)
            dis1 = Point[1] - Point0[select][1]
            dis2 = Point[0] - Point0[select][0]
            Data.append(dis)
            Data1.append(dis1)
            Data2.append(dis2)
    Data.insert(0, 0)
    Data1.insert(0, 0)
    Data2.insert(0, 0)
    fps = 120
    x = range(0, len(Data))
    t = np.linspace(start=0, stop=len(Data)/fps, num=len(Data))

    plt.figure()
    j = 783

    # 振幅绘图
    # dis
    plt.subplot(131)
    HanShu.a_plot(x, Data, j, 'Distance')

    # plt.subplot(122), plt.plot(Data_fft_fqe, Data_fft)

    # Y
    plt.subplot(132)
    HanShu.a_plot(x, Data1, j, 'Y-Distance')

    # X
    plt.subplot(133)
    HanShu.a_plot(x, Data2, j, 'X-Distance')

    # 傅里叶频率分析
    HanShu.f_plot(x, Data1, fps, 'Y')
    HanShu.f_plot(x, Data, fps, 'Distance')
    HanShu.f_plot(x, Data2, fps, 'X')

    plt.show()
    vc.release()


Video_test()
# HanShu.Area_test(fname)
# HanShu.fft_sin()
# HanShu.dis_test()
# HanShu.test([[0, 0], [1, 1], [-1, 2]], [[0, 0], [2, 1], [3, 3]])
