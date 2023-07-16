import numpy as np
import cv2
import glob
from HanShu import cv_show, resize

# 迭代参数设定
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

# 创建世界坐标
objp = np.zeros((8 * 11, 3), np.float32)
# print(objp)
objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * 1.5
# print(objp[:, :2])
objpoints = []
imgpoints = []
# images = glob.glob('./chess1/*.JPG')
images = glob.glob('./chess/*.DNG')
# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('gray', gray)
#     while cv2.waitKey(100) != 27:
#         if cv2.getWindowProperty('gray', cv2.WND_PROP_VISIBLE) <= 0:
#             break
# cv2.destroyAllWindows()

# 找到棋盘的像素坐标，并对应世界坐标
for fname in images:
    img = cv2.imread(fname)
    # print(img.shape)
    # img = resize(img, width=2016)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
    # print('corners', corners)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, (11, 8), corners2, ret)
        # cv_show('img', img)

# 得到标定参数
ret, mrx, dist, rvecs, tveces = cv2.calibrateCamera(objpoints, imgpoints, (11, 8), None, None)
# print('mrx', mrx)

# 畸变矫正
img = cv2.imread('./chess1/2023_04_11_21_05_IMG_0222.JPG')
# img = resize(img, width=2016)
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mrx, dist, (w, h), 0, (w, h))
dst = cv2.undistort(img, mrx, dist, None, newcameramtx)
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.JPG', dst)
print('mrx')
print(mrx)
print('newcameramtx')
print(newcameramtx)

# 计算重投影误差
total_error = 0
for i in range(len(objpoints)):
    img_points_repro, _ = cv2.projectPoints(objpoints[i], rvecs[i], tveces[i], mrx, dist)
    error = cv2.norm(imgpoints[i], img_points_repro, cv2.NORM_L2) / len(img_points_repro)
    total_error = total_error + error
print('Average: ', total_error / len(objpoints))
print('mrx', mrx)

