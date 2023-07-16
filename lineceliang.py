# 轮廓提取
import cv2
import numpy as np

# 图像显示
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 读取图片
img = cv2.imread('.\\Pictures_File\\text.jpg')
# cv_show('text', img)

# 转为灰度图像
img_h = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv_show('text_h', img_h)

# 二值化图像
ret, dst = cv2.threshold(img_h, 90, 255, cv2.THRESH_BINARY_INV)
# cv_show('text_threshold', dst)

# 找轮廓
contours, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# 绘制轮廓
draw_img = img.copy()
res = cv2.drawContours(draw_img, contours, 11, (0, 0, 255), 1)
# Draw = np.hstack((res, img))
# cnt = contours[0]
cv_show('lunk', res)
# print(cnt)
print(cv2.arcLength(contours[11], True))