import cv2
import numpy as np

# 读取图像
image = cv2.imread('image.jpg')

# 转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 计算哈里斯角
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# 通过膨胀操作增强角点
dst = cv2.dilate(dst, None)

# 设定阈值
image[dst > 0.01 * dst.max()] = [0, 0, 255]

# 显示图像
cv2.imshow('Harris Corners', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
