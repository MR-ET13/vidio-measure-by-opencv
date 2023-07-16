# 边缘检测
import cv2
import numpy as np
from HanShu import cv_show

# # 背景过滤
img = cv2.imread('.\\Pictures_File\\ShuiPin_text\\10.jpg')
shape = img.shape
img = img[300:750, 200:1100]
# print(shape)
# b, g, r = cv2.split(img)
# cv_show('b', b)
# cv_show('g', g)
# cv_show('r', r)

# cv_show('img', img)
lower = np.uint8([150, 150, 150])
upper = np.uint8([255, 255, 255])
white_mask = cv2.inRange(img, lower, upper)
masked = cv2.bitwise_and(img, img, mask=white_mask)
# 中值滤波
# masked_m = cv2.medianBlur(masked, 5)
# show = np.stack((masked, masked_m))
# cv_show('white_mask', masked)


# 导入图片（灰度）
h_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_o = cv2.imread('.\\Pictures_File\\ShuiPin_text\\9.jpg')
# img = cv2.imread('.\\Pictures_File\\ShuiPin_text\\9.jpg', cv2.IMREAD_GRAYSCALE)
# lower = 90
# upper = 255
# mask = cv2.inRange(img, lower, upper)
# cv_show('img', h_img)

# Canny边缘检测（80，150）  ->相机位处于相对垂直的位置
v1 = cv2.Canny(h_img, 50, 200)
# v2 = cv2.Canny(b, 80, 150)
# v3 = cv2.Canny(g, 80, 150)
# v4 = cv2.Canny(r, 80, 150)
# imge = np.hstack((v2, v3, v4))
# cv_show('name', v1)

# 轮廓检测
cnts = cv2.findContours(v1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
res = cv2.drawContours(img.copy(), cnts, 0, (0, 0, 255), 1)
peri = cv2.arcLength(cnts[0], True)
approx = cv2.approxPolyDP(cnts[0], 0.1*peri, True)
res2 = cv2.drawContours(img.copy(), [approx], -1, (0, 0, 255), 1)
(x, y, w, h) = cv2.boundingRect(approx)
# res3 = cv2.rectangle(img.copy(), (x, y), (x+w, y+h), (255, 0, 0), 2, cv2.LINE_AA)
cv_show('contours', np.hstack((res, res2)))


# 直线提取
# 线段最小长度，线段之间的最大间隔，距离精度，角度精度，阈值
# lines = cv2.HoughLinesP(v1, rho=0.1, theta=np.pi/180, threshold=15, minLineLength=80, maxLineGap=4)
# img_c = img.copy()
# for line in lines:
#     x1, y1, x2, y2 = line[0]
#     cv2.line(img_c, (x1, y1), (x2, y2), (0, 0,  255), 1)
# cv_show('line', img_c)

# 计算距离
