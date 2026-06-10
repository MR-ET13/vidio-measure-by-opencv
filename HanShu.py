import cv2
import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import numpy.fft as nf

I_roi = []


def cv_show(name, img):
    """
    显示图片
    :param name: 显示框标题
    :param img: 图片
    :return: 无
    """
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Get_Feature_Points(img, i_roi, num_f=0, area_test=False, AREA=0):
    """
    获取特征点的坐标值
    :param img: 图片
    :param i_roi: roi区域
    :param num_f: 当前帧序号
    :param area_test: 是否进行面积标定
    :param AREA: 标定面积值
    :return: 如果是标定面积，返回轮廓检测值；如果是获取特征点，返回亚像素坐标值
    """
    global I_roi
    # 问题帧+1
    test_f = 1
    # ROI
    I_roi = i_roi
    img = img[i_roi[0]:i_roi[1], i_roi[2]:i_roi[3]]
    # cv_show('img_roi', img)

    # 图像灰度化
    img_gay = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv_show('img_gay', img_gay)

    # 腐蚀
    # kernel = np.ones((3, 3), np.uint8)
    # img_erode = cv2.erode(img_gay, kernel, iterations=2)
    # cv_show('img_open', np.hstack((img_gay, img_erode)))

    # 开运算
    # kernel = np.ones((3,3), np.uint8)
    # img_open = cv2.morphologyEx(img_gay, cv2.MORPH_OPEN, kernel)
    # cv_show('img_open', np.hstack((img_gay, img_open)))

    # 中值滤波
    img_median = cv2.medianBlur(img_gay, 3)
    # cv_show('img_median', np.hstack((img_gay, img_median)))

    # 均值化
    # img_equ = cv2.equalizeHist(img_gay)
    # cv_show('img_median', np.hstack((img_gay, img_equ)))

    # 角点检测
    # dst = cv2.cornerHarris(img_gay, 2, 3, 0.04)

    # 边缘检测--中值滤波
    img_canny = cv2.Canny(img_median, 50, 200)
    # img_erode_canny = cv2.Canny(img_erode, 50, 200)
    # img_open_canny = cv2.Canny(img_open, 50, 200)
    img_median_canny = cv2.Canny(img_gay, 50, 200)

    # 问题帧显示-边缘检测，灰度图，中值滤波图，中值滤波边缘图
    if num_f == test_f:
        cv_show('img_median_canny  img_gay  img_median img_gay_canny',
                np.hstack((img_canny, img_gay, img_median, img_median_canny)))

    # 轮廓检测-三角形
    cnts = cv2.findContours(
        img_canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnts = sorted(cnts, key=Counter_Area, reverse=False)
    cnts_0 = cv2.drawContours(img.copy(), cnts, 0, (0, 0, 255), 1)
    cnts_all = cv2.drawContours(img.copy(), cnts, -1, (0, 0, 255), 1)
    cnts_1 = cv2.drawContours(img.copy(), cnts, 1, (0, 0, 255), 1)
    if num_f == test_f:
        cv_show('t', cnts_0)

    # x最大，x最小，y最大, y最小的角点
    point = [0, 0]
    point1 = [img.shape[1], 0]
    point2 = [0, 0]
    point3 = [0, img.shape[0]]
    for ppoint in cnts[0]:
        if ppoint[0][0] > point[0]:
            point = ppoint[0]
        if ppoint[0][0] < point1[0]:
            point1 = ppoint[0]
            # cv2.circle(img, (int(point1[0]), int(point1[1])), 2, (0, 255, 0), 1)
        if ppoint[0][1] > point2[1]:
            point2 = ppoint[0]
        if ppoint[0][1] < point3[1]:
            point3 = ppoint[0]

    # 与标定面积偏差，面积标定时去掉
    if area_test == False:
        area_pl = abs(AREA - max(Area_Triangles(point, point1,
                      point2), Area_Triangles(point3, point1, point2)))
        if area_pl > 500:
            img_c = img.copy()
            cv2.circle(img_c, (int(point[0]), int(
                point[1])), 2, (0, 255, 0), 1)
            cv2.circle(img_c, (int(point1[0]), int(
                point1[1])), 2, (0, 255, 0), 1)
            cv2.circle(img_c, (int(point2[0]), int(
                point2[1])), 2, (0, 255, 0), 1)
            cv2.circle(img_c, (int(point3[0]), int(
                point3[1])), 2, (0, 0, 255), 1)
            cv_show('img_c', img_c)
            # 将第二个轮廓合并
            for ppoint in cnts[1]:
                if ppoint[0][0] > point[0]:
                    point = ppoint[0]
                if ppoint[0][0] < point1[0]:
                    point1 = ppoint[0]
                if ppoint[0][1] > point2[1]:
                    point2 = ppoint[0]
                if ppoint[0][1] < point3[1]:
                    point3 = ppoint[0]

    cv2.circle(img, (int(point[0]), int(point[1])), 2, (255, 0, 0), 1)
    cv2.circle(img, (int(point1[0]), int(point1[1])), 2, (0, 255, 0), 1)
    cv2.circle(img, (int(point2[0]), int(point2[1])), 2, (0, 0, 255), 1)
    cv2.circle(img, (int(point3[0]), int(point3[1])), 2, (0, 0, 255), 1)

    # cv_show('img', img)

    # 角点检测ROI
    mask = np.zeros_like(img_gay)
    delta = 10
    pt_0 = [point1[0]-delta, point3[1]-delta]
    pt_1 = [point[0]+delta, point3[1]-delta]
    pt_2 = [point[0]+delta, point2[1]+delta]
    pt_3 = [point1[0]-delta, point2[1]+delta]
    vertices = np.array([[pt_0, pt_1, pt_2, pt_3]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, 255)
    # cv_show('mask', mask)
    img_Roi_Angular_point = cv2.bitwise_and(img_gay, mask)
    # cv_show('Roi_Angular_point', np.hstack((img_gay, img_Roi_Angular_point)))

    # Harris像素级角点检测
    Angular_float = np.float32(img_Roi_Angular_point)
    dst = cv2.cornerHarris(Angular_float, 2, 3, 0.04)
    # dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01*dst.max(), 255, cv2.THRESH_BINARY)
    dst = np.uint8(dst)
    # cv_show('dst', dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # SubPix亚像素级角点检测
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(Angular_float, np.float32(
        centroids), (5, 5), (-1, -1), criteria)
    # 亚像素级角点绘图
    for p in corners:
        cv2.circle(img, (int(p[0]), int(p[1])), 4, (255, 255, 255), 1)
    Angular_Point = []
    # cv_show('img', img)

    # 寻找距离三角形角点最近的亚像素级角点坐标
    point1_close = Proximity_Point1(point1, corners)
    point2_close = Proximity_Point1(point2, corners)
    point_point3_close = Proximity_Point2(point, point3, corners)
    Angular_Point.append(point_point3_close)
    Angular_Point.append(point1_close)
    Angular_Point.append(point2_close)
    cv2.circle(img, (int(point1_close[0]), int(
        point1_close[1])), 4, (0, 255, 0), 1)
    cv2.circle(img, (int(point2_close[0]), int(
        point2_close[1])), 4, (0, 0, 255), 1)
    cv2.circle(img, (int(point_point3_close[0]), int(
        point_point3_close[1])), 4, (255, 0, 0), 1)
    # Angular_Point.sort(key=Point_sort_x)
    # td = [0, 0]
    # if len(Angular_Point) == 0:
    #     Angular_Point.append(point1)
    #     Angular_Point.append(point2)
    #     Angular_Point.append(point3)
    #     td[0] = td[0] + 1
    # if len(Angular_Point) == 1:
    #     if Proximity_Point(Angular_Point[0], point) or Proximity_Point(Angular_Point[0], point3):
    #         Angular_Point.append(point1)
    #         Angular_Point.append(point2)
    #     if Proximity_Point(Angular_Point[0], point2):
    #         Angular_Point.append(point1)
    #         Angular_Point.append(point3)
    #     if Proximity_Point(Angular_Point[0], point1):
    #         Angular_Point.append(point2)
    #         Angular_Point.append(point3)
    #     td[1] = td[1] + 1
    #     Angular_Point.sort(key=Point_sort_x)

    # cv_show('img', img)

    cv2.putText(img, '(' + str(point[0]) + ', ' + str(point[1]) + ')', (int(point[0]) - 40, int(point[1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))

    # 有问题的帧绘图，序号加一
    # 第0边界, 第1边界, 绘制过的img
    if num_f == test_f and num_f != 0:
        cv_show('first_cnt  second_cnt  all_cnt  img_done',
                np.hstack((cnts_0, cnts_1, cnts_all, img)))
        print('Angular_Point: ', Angular_Point)
    if num_f == 0:
        cv_show('first_cnt  second_cnt  all_cnt  img_done',
                np.hstack((cnts_0, cnts_1, cnts_all, img)))
    if area_test == True:
        # 返回三角形角点
        return point, point1, point2, point3
    else:
        # 返回亚像素级角点
        return Angular_Point

    # 寻找轮廓点坐标
    # mask_out_edge = np.zeros(img_gay.shape, np.uint8)
    # cv2.drawContours(mask_out_edge, cnts, 0, 255, 1)
    # cv2.imshow('mask', mask_out_edge)
    # pixel_pont = cv2.findNonZero(mask_out_edge)
    # (x1, y1) = pixel_pont[0, 0]

    # (x2, y2) = cnts[0][0, 0]
    # cnt_gray = cv2.cvtColor(cnts[0], cv2.COLOR_BGR2GRAY)
    # cnt_gray_float = np.float32(mask_out_edge)
    # dst = cv2.cornerHarris(cnt_gray_float, 2, 3, 0.04)
    # img[dst>0.01 * dst.max()] = [0, 0, 255]
    # cv2.imshow('Jiaodian', img)

    # 轮廓近似
    # peri = cv2.arcLength(cnts[0], True)
    # approx = cv2.approxPolyDP(cnts[0], 0.001 * peri, True)
    # cnts_approx = cv2.drawContours(img.copy(), [approx], -1, (0, 0, 255), 1)
    # (x, y, w, h) = cv2.boundingRect(approx)
    # cnts_rect = cv2.rectangle(img.copy(), (x, y), (x+w, y+h), (255, 0, 0), 2, cv2.LINE_AA)
    # x = 0
    # y = 0
    # n = len(approx)
    # for point in approx:
    #     x = x + point[0][0]
    #     y = y + point[0][1]
    # cv2.circle(img, (int(x / n), int(y / n)), 2, (0, 255, 0), 2)
    # cv2.putText(img, '('+str(x/n)+', '+str(y/n)+')', (int(x/n)-40, int(y/n)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
    # cv_show('contours', np.hstack((cnts_0, cnts_approx, img)))
    # return x / n, y / n


# 求距离
def Point_Distance(p1, p2):
    """
    求两点距离
    :param p1: 第一个点
    :param p2: 第二个点
    :return: 欧式距离值
    """
    # if p2[1] > p1[1]:
    #     dis = dist.euclidean(p2, p1)
    # else:
    #     dis = dist.euclidean(p2, p1)
    return dist.euclidean(p2, p1)

# 取近距离点，相对于一个点


def Proximity_Point1(p1, cornor):
    """
    返回角点中距离指定点最近的点
    :param p1: 指定点
    :param cornor: 角点集
    :return: 最近点
    """
    p0 = cornor[0]
    d0 = dist.euclidean(p1, p0)
    for p in cornor:
        d = dist.euclidean(p1, p)
        if d < d0:
            p0 = p
            d0 = d
    return p0

# 取近距离点，相对于两个个点


def Proximity_Point2(pp, p1, cornor):
    """
    返回角点中距离指定两点最近的点
    :param pp: 0点
    :param p1: 3点
    :param cornor: 角点集
    :return: 最近点
    """
    p0 = cornor[0]
    d0 = dist.euclidean(p1, p0) + dist.euclidean(pp, p0)
    for p in cornor:
        d = dist.euclidean(p1, p) + dist.euclidean(pp, p)
        if d < d0:
            p0 = p
            d0 = d
    return p0


# 轮廓面积比排序
def Counter_Area(cnt):
    """
    筛选最优轮廓
    :param cnt: 指定轮廓
    :return: 轮廓重心与roi区域中点的距离
    """
    global i_roi
    x_sum = 0
    y_sum = 0
    for p in cnt:
        x_sum = x_sum + p[0][0]
        y_sum = y_sum + p[0][1]
    P1 = [x_sum / len(cnt), y_sum / len(cnt)]
    P2 = [(I_roi[3] - I_roi[2]) / 2, (I_roi[1] - I_roi[0]) / 2]
    return dist.euclidean(P2, P1)

# ROI区域测试


def ROI(img):
    """
    找到roi区域
    :param img: 图片
    :return: roi的高度范围和宽度范围
    """
    print(img.shape)
    h = 70
    w = 220
    h1, w1 = 688, 490
    h2 = h1 + h
    w2 = w1 + w
    img = img[h1:h2, w1:w2]
    print(img.shape)
    pt1 = (0, int((h2 - h1) / 2))
    pt1_ = (img.shape[1], int((h2 - h1) / 2))
    pt2 = (int((w2 - w1) / 2), 0)
    pt2_ = (int((w2 - w1) / 2), img.shape[0])
    point_color = (0, 255, 0)  # BGR
    thickness = 1
    lineType = 4
    cv2.line(img, pt1, pt1_, point_color, thickness, lineType)
    cv2.line(img, pt2, pt2_, point_color, thickness, lineType)
    cv_show('ROI_test', img)
    return h1, h2, w1, w2

# 三点求三角形面积


def Area_Triangles(p1, p2, p3):
    """
    三点求三角形面积
    :param p1: 第一个点
    :param p2: 第二点
    :param p3: 第三点
    :return: 三角形面积
    """
    return 0.5 * ((p2[0] - p1[0]) ** 2 * (p3[1] - p1[1]) ** 2 + (p2[1] - p1[1]) ** 2 * (p3[0] - p1[0]) ** 2) ** 0.5

# 面积标定


def Area_Test(img):
    """
    返回特征点组成三角形面积
    :param img: 图片
    :return: 三角形面积
    """
    Point = Get_Feature_Points(img, ROI(img.copy()), area_test=True)
    return Area_Triangles(Point[0], Point[1], Point[2])


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    以高度或宽度重新裁剪图片
    :param image: 图片
    :param width: 宽度
    :param height: 高度
    :param inter:
    :return: 裁剪后的图片
    """
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# 数据处理


def Data_cope(x, Data, c=-1, j=0):
    """
    数据处理
    :param x: Data序号
    :param Data: 位移值
    :param c: 平稳测试时的稳定截取段
    :param j: 位移测试时的振动段
    :return: 上包络拟合，下包络拟合，均值拟合，均值下标，均值点，输出拟合最值和中值
    """
    step = 50
    start = j
    stop = start + step - 1
    start_m = 0
    stop_m = start_m + step - 1
    max = []
    max_index = []
    min = []
    min_index = []
    average = []
    average_index = []
    while stop < len(Data):
        flag = 0
        arr = np.array(Data[start:stop])
        for v in arr:
            if abs(v) > 3:
                flag = 1
                break
        if flag == 1:
            start = start + step
            stop = stop + step
            continue
        max.append(np.max(arr))
        max_index.append(start + np.argmax(arr))
        min.append(np.min(arr))
        min_index.append(start + np.argmin(arr))
        # average.append(np.sum(arr) / len(arr))
        # average_index.append((start + stop + 1) / 2)
        start = start + step
        stop = stop + step
    while stop_m < len(Data):
        flag = 0
        arr = np.array(Data[start_m:stop_m])
        for v in arr:
            if abs(v) > 3:
                flag = 1
                break
        if flag == 1:
            start_m = start_m + step
            stop_m = stop_m + step
            continue
        average.append(np.sum(arr) / len(arr))
        average_index.append((start_m + stop_m + 1) / 2)
        start_m = start_m + step
        stop_m = stop_m + step
    z1 = np.polyfit(max_index, max, 5)
    p1 = np.poly1d(z1)
    z2 = np.polyfit(min_index, min, 5)
    p2 = np.poly1d(z2)
    z3 = np.polyfit(average_index, average, 5)
    p3 = np.poly1d(z3)
    up = p1(x[j:])
    down = p2(x[j:])
    median = p3(x)
    if c != -1:
        index = range(c, int(np.max(average_index)))
        for i in range(0, 3):
            if i == 0:
                print('dis_max = ', sum(p1(index))/len(p1(index)))
            if i == 1:
                print('dis_min = ', sum(p2(index)) / len(p2(index)))
            if i == 2:
                print('dis_median = ', sum(p3(index)) / len(p3(index)))
    if j != 0 or c == -1:
        index = range(j, int(np.max(average_index)))
        for i in range(0, 3):
            if i == 0:
                print('dis_max = ', sum(p1(index))/len(p1(index)))
            if i == 1:
                print('dis_min = ', sum(p2(index)) / len(p2(index)))
            if i == 2:
                print('dis_median = ', sum(p3(index)) / len(p3(index)))
    return up, down, median, average_index, average

# 得到基点


def Get_Point0(fname, i_roi, num_f=0, area_test=False, AREA=0, n=1):
    """
    得到基准点
    :param fname: 视频名称
    :param i_roi: roi区域
    :param num_f: 当前帧数
    :param area_test: 是否进行面积标定
    :param AREA: 标定面积
    :param n: 选择点数值
    :return: 均值化后的基点
    """
    m = 1
    vc = cv2.VideoCapture(fname)
    ret, frame = vc.read()
    p0 = Get_Feature_Points(frame, i_roi, num_f, area_test, AREA)
    while n > 0:
        ret, frame = vc.read()
        p = Get_Feature_Points(frame, i_roi, -1, area_test, AREA)
        for i in range(0, 3):
            for j in range(0, 2):
                p0[i][j] = p0[i][j] + p[i][j]
        m = m + 1
        n = n - 1
    for i in range(0, 3):
        for j in range(0, 2):
            p0[i][j] = p0[i][j] / m
    return p0


# 频率作图
def f_plot(x, Data, fps, s):
    """
    傅里叶变换作频率图像
    :param x: Data序号
    :param Data: 位移值
    :param fps: 帧率
    :param s: 名称
    :return: 无
    """
    plt.figure()
    Data_fft = nf.fft(Data)
    Data_fft_fqe = nf.fftfreq(len(Data), d=1 / fps)
    Data_fft = np.abs(Data_fft)
    plt.subplot(121), plt.plot(x, Data), plt.scatter(x, Data)
    plt.xlabel('Frames'), plt.ylabel(s)
    plt.subplot(122), plt.plot(Data_fft_fqe, Data_fft)
    # plt.xlim(0, int(fps / 2))
    plt.xlabel('f')

# 振幅绘图


def a_plot(x, Data, j, s):
    """
    绘制振幅图
    :param x: Data序号
    :param Data: 位移值
    :param j: 振幅稳定截取值
    :param s: 名称
    :return: 无
    """
    plt.plot(x, Data), plt.scatter(x, Data)
    up = Data_cope(x, Data, -1, j)
    for sel in range(0, 3):
        # 中间点不需要截断
        if sel != 2:
            plt.plot(x[j:], up[sel], color='r', linestyle='--')
        else:
            plt.plot(x, up[sel], color='r', linestyle='--')
    plt.scatter(up[3], up[4], color='k', zorder=2)
    plt.xlabel('Frames'), plt.ylabel(s)

# 面积标定


def Area_test(fname):
    """
    获取标准面积值，用于轮廓重选
    :param fname: 视频路径名称
    :return: 标定面积值
    """
    vc = cv2.VideoCapture(fname)
    AREA = 0
    n = 4
    for i in range(1, n):
        ret, frame = vc.read()
        AREA = AREA + Area_Test(frame)
    print(AREA / (n - 1))
    return AREA / (n - 1)


# 傅里叶分析测试
def fft_sin():
    """
    傅里叶分析测试
    :return:
    """
    t = np.linspace(start=0, stop=10, num=1000)
    y = 4*np.sin(2*np.pi*t*2)
    ax1 = plt.subplot(121)
    plt.plot(t, y)

    fs = 100
    xf = nf.fft(y)
    xfp = nf.fftfreq(len(y), d=1 / fs)
    xf = np.abs(xf)
    ax1 = plt.subplot(122)
    plt.xlim(0, int(fs/2))
    plt.plot(xfp, xf)
    plt.show()
