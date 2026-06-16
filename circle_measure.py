
import HanShu as H
import rawpy
import cv2
import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
from queue import Queue
import threading
import pandas as pd


# SIZE = (100, 100, 680, 500) # test_pic1.DNG
SIZE = (480, 216, 0, 0) # test-v1.MOV # ROI尺寸
CAL_RADIUS = 25.3 # 标定半径
CENTER = (105.6, 241.9) # 标定圆心
PERIMETER = 162.8 # 标定周长?
START_FRAME = 4500 # 开始帧
END_FRAME = 5100 # 结束帧
ALL_FRAME = 5189
FPS = 30
LW = 1080
LH = 1920
T = 212.20

THREAD_X = 25 # X方向偏移阈值
L_PER_PIXEL = 7/(2 * 27.7) # 每像素的实际尺寸

def read_dng(fn):
    """
    读取dng图片
    :param fn: 图片路径
    :return: img
    """
    with rawpy.imread(fn) as raw:
        rgb = raw.postprocess()  # 得到 RGB numpy 数组
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # 转 OpenCV 的 BGR
    img_show = cv2.resize(bgr, (0, 0), fx=0.2, fy=0.2)
    # print(img_show.shape)
    # cv2.imshow("DNG", img_show)
    # cv2.waitKey(0)
    return img_show

def circle_m(img):
    """
    返回图片中圆形轮廓的圆心坐标
    :param img: 图片
    :return: 圆心坐标
    """
    # i_roi = H.ROI(img.copy(), SIZE)
    # img_roi = img[i_roi[0]:i_roi[1], i_roi[2]:i_roi[3]]
    img_roi = remove_black_border(img)
    img_gay = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY) # 灰度处理

    # 3. 滤波降噪（二选一，根据噪声情况）
    # 高斯模糊：适合高斯噪声、画面整体偏噪（最常用）
    blur = cv2.GaussianBlur(img_gay, (3, 3), 0)
    # 中值滤波：适合椒盐噪声、白点/黑点杂点
    # blur = cv2.medianBlur(img_gay, 3)

    # 4. 二值化（二选一，根据光照）
    # 固定阈值：光照均匀、背景简单
    ret, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    # 自适应阈值：光照不均、明暗差异大（强烈推荐工业/实拍图）
    # binary = cv2.adaptiveThreshold(blur, 255,
    #                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY, 15, 3)

    # 5. 形态学操作（轮廓优化：去小噪点、填补轮廓间隙）
    # 定义结构元素
    kernel = np.ones((3, 3), np.uint8)
    # 开运算：先腐蚀再膨胀 → 去除**小白点噪声、细小干扰轮廓**
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # 闭运算：先膨胀再腐蚀 → 填补**轮廓内部小孔、边缘断裂间隙**
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 可选：反色（目标黑、背景白时使用）
    # binary = cv2.bitwise_not(binary)

    # 可视化每一步结果
    H.cv_show("gray , blur , binary",
              np.hstack((img_gay, blur, binary)))

    cnts_init = cv2.findContours(
        binary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0] # 轮廓检测
    # cnts = sorted(cnts_init, key=gravity_distance, reverse=False) # 重心确定标定面积
    cnts = sorted(cnts_init, key=radius_distance, reverse=False) # 半径确定接近轮廓
    # cnts = sorted(cnts_init, key=perimeter_distance, reverse=False)  # 周长确定接近轮廓
    cnt_all = cv2.drawContours(img_roi.copy(), cnts, -1, (0, 0, 255), 1) # 所有轮廓
    cnt_0 = cv2.drawContours(img_roi.copy(), cnts, 0, (0, 0, 255), 1) # 第一匹配轮廓
    cnt_1 = cv2.drawContours(img_roi.copy(), cnts, 1, (0, 0, 255), 1) # 第二匹配轮廓
    H.cv_show('first_cnt  second_cnt  all_cnt', np.hstack((cnt_0, cnt_1, cnt_all)))
    # for cnt_i in cnts:
    #     print(radius_distance(cnt_i))
    #     print(erro_distance(cnt_i))

    # center0, radius0 = cv2.minEnclosingCircle(cnts[0])
    # perimeter0 = cv2.arcLength(cnts[0], True)
    # center1, radius1 = cv2.minEnclosingCircle(cnts[1])
    # perimeter1 = cv2.arcLength(cnts[1], True)

    min_cnt = 0
    for i in range(len(cnts)):
        if erro_distance(cnts[i]) < THREAD_X:
            center, radius = cv2.minEnclosingCircle(cnts[i])
            perimeter = cv2.arcLength(cnts[i], True)
            scenter =(int(center[0]), int(center[1]))
            sradius = int(radius)
            cv2.circle(img_roi, scenter, sradius, (0, 255, 0), 2)
            cv2.circle(img_roi, scenter, 2, (255, 0, 0), -1)
            H.cv_show("拟合圆", img_roi)
            return center, radius, perimeter
        if erro_distance(cnts[i]) < erro_distance(cnts[min_cnt]):
            min_cnt = i

    print("轮廓错误")
    centerm, radiusm = cv2.minEnclosingCircle(cnts[min_cnt])
    perimeterm = cv2.arcLength(cnts[min_cnt], True)
    scenter = (int(centerm[0]), int(centerm[0]))
    sradius = int(radiusm)
    cv2.circle(img_roi, scenter, sradius, (0, 255, 0), 2)
    cv2.circle(img_roi, scenter, 2, (255, 0, 0), -1)
    print("误差值*******************************")
    for cnt_i in cnts:
        print(radius_distance(cnt_i))
        print(erro_distance(cnt_i))
    print("*******************************误差值")
    H.cv_show('first_cnt  second_cnt  all_cnt', np.hstack((cnt_0, cnt_1, cnt_all)), 1)
    H.cv_show("拟合圆", img_roi, 1)
    return centerm, radiusm, perimeterm

def read_vidio(fn):
    """
    视频测量位移
    :param fn: 视频路径
    :return: None
    """
    global CENTER
    vc = cv2.VideoCapture(fn) # 读取视频
    start_frame = START_FRAME
    end_frame = END_FRAME
    delta_y = [] # y方向位移
    delta_x = [] # x方向位移
    while True:
        if start_frame > 0: # 开始帧
            ret, frame = vc.read()
            center0, *_ = circle_m(frame)
            start_frame -= 1
            end_frame -= 1
            continue
        elif end_frame > 0: # 结束帧
            ret, frame = vc.read()
            end_frame -= 1
            if frame is None:
                break
            center, *_ = circle_m(frame)
            CENTER = center
            delta_y.append(center[1] - center0[1])
            delta_x.append(center[0] - center0[0])
        else:
            break

    delta_y.insert(0, 0)
    delta_x.insert(0, 0)
    delta_y = np.array(delta_y)
    delta_x = np.array(delta_x)
    x = range(0, len(delta_y))

    delta_y *= L_PER_PIXEL
    delta_x *= L_PER_PIXEL
    fps = 30
    plt.subplot(211)
    plt.plot(x, delta_y)
    # plt.scatter(x, delta_y)

    plt.subplot(212)
    plt.plot(x, delta_x)
    # plt.scatter(x, delta_x)

    df = pd.DataFrame({
        "x": list(x),
        "delta_y": delta_y,
        "delta_x": delta_x
    })
    # 导出excel
    df.to_excel("Pictures_File/circular_recognition_pic/data.xlsx", index=False)

    print(f"幅度：{max(delta_y)-min(delta_y)}")
    plt.show()
    vc.release()

def erro_frame(fn, index):
    """
    错误帧查看
    :param fn: 视频路径
    :param index: 错误帧数
    :return: None
    """
    vc = cv2.VideoCapture(fn)
    frame_num = 0
    while True:
        ret, frame = vc.read()
        frame_num += 1
        if frame_num < index:
            continue
        else:
            circle_m(frame)
            break

def gravity_distance(cnt):
    """
    重心坐标距离
    :param cnt: 轮廓
    :return: 距离
    """
    x_sum = 0
    y_sum = 0
    for p in cnt:
        x_sum += p[0][0]
        y_sum += p[0][1]
    P1 = [x_sum / len(cnt), y_sum / len(cnt)]
    P2 = [SIZE[1] / 2, SIZE[0] / 2]
    return dist.euclidean(P2, P1)

def perimeter_distance(cnt):
    """
    周长距离
    :param cnt: 轮廓
    :return: 距离
    """
    per = cv2.arcLength(cnt, True)
    return abs(per - PERIMETER)

def radius_distance(cnt):
    """
    半径距离
    :param cnt: 轮廓
    :return: 距离
    """
    center, radius = cv2.minEnclosingCircle(cnt)
    return abs(radius - CAL_RADIUS)

def erro_distance(cnt):
    """
    错误帧处理
    :param cnt: 轮廓
    :return: 距离
    """
    center, radius = cv2.minEnclosingCircle(cnt)
    return abs(center[0] - CENTER[0])

def area_cal(fn):
    """
    c初始参数标定
    :param fn: 视频路径
    :return: None
    """
    vc = cv2.VideoCapture(fn)
    start_frame = START_FRAME
    # end_frame = START_FRAME + 1 # 单独标定
    end_frame = END_FRAME # 取平均值，用于实际距离和像素转换，视频测试有没问题后执行
    radiuss = []
    perimeters = []
    while True:
        if start_frame > 0:
            ret, frame = vc.read()
            start_frame -= 1
            end_frame -= 1
            continue
        elif end_frame > 0:
            ret, frame = vc.read()
            end_frame -= 1
            # H.ROI(frame, SIZE)
            center, radius, perimeter = circle_m(frame)
            radiuss.append(radius)
            perimeters.append(perimeter)
        else:
            break
    print(f"平均半径：{np.mean(radiuss)} || 平均周长：{np.mean(perimeters)}")
    print(f"圆形坐标：{center}")


def crop_video(input_path, output_path, x, y, w, h):
    """
    视频裁剪
    :param input_path: 视频路径
    :param output_path: 输出路径
    :param x: 位置x
    :param y: 位置y
    :param w: 裁剪宽度
    :param h: 裁剪高度
    :return: None
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception("无法打开视频")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 输出 mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 注意：numpy 是 [y:y+h, x:x+w]（高在前，宽在后）
        crop = frame[y:y+h, x:x+w]
        out.write(crop)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def basic_info(fn):
    cap = cv2.VideoCapture(fn)  # 0 代表摄像头，填路径为本地视频

    # 1. 总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 2. 帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 3. 分辨率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 4. 视频总时长(秒)
    duration = total_frames / fps

    print(f"总帧数: {total_frames}")
    print(f"帧率: {fps:.2f}")
    print(f"分辨率: {width} x {height}")
    print(f"总时长: {duration:.2f} s")

    cap.release()

def read_frame(cap, q):
    while True:
        ret, frame = cap.read()
        q.put((ret, frame))
        if not ret:
            break

def crop_video_fast(input_path, output_path, x, y, w, h):
    cap = cv2.VideoCapture(input_path)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if x + w > src_w or y + h > src_h:
        raise ValueError("裁剪区域越界")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    q = Queue(maxsize=100)
    # 子线程读帧，主线程处理+写入
    t = threading.Thread(target=read_frame, args=(cap, q))
    t.start()

    while True:
        ret, frame = q.get()
        if not ret:
            break
        out.write(frame[y:y+h, x:x+w])

    cap.release()
    out.release()
    t.join()

def remove_black_border(img, flag=0):
    # 转灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化：黑色(0)置0，其余内容置255
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # 获取所有非零点坐标
    coords = np.column_stack(np.where(binary > 0))
    # 有效区域边界
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    # 裁剪原图
    crop_img = img[y_min:y_max+1, x_min:x_max+1]
    if flag:
        print(crop_img.shape)
    return crop_img


if __name__ == '__main__':
    filename = "Pictures_File/circular_recognition_pic/test3000-120/test3000-120.mp4"

    # read_dng(filename) # 读入dng文件
    # circle_m(read_dng(filename)) # 圆形检测

    # basic_info(filename)
    # area_cal(filename) # 初始标定参数*****
    read_vidio(filename) # 视频测量*****
    # erro_frame(filename, START_FRAME+484) # 错误帧查看*****