import cv2

# 读取图片1和图片2
img1 = cv2.imread('image1.png', 0)  # 以灰度模式读取图片1
img2 = cv2.imread('image2.png', 0)  # 以灰度模式读取图片2

# 初始化SIFT关键点检测器和描述符提取器
sift = cv2.SIFT_create()  # 创建SIFT对象

# 检测关键点和计算描述符
kp1, des1 = sift.detectAndCompute(img1, None)  # 检测图片1的关键点和计算描述符
kp2, des2 = sift.detectAndCompute(img2, None)  # 检测图片2的关键点和计算描述符

# 使用BFMatcher进行特征匹配
bf = cv2.BFMatcher()  # 初始化BFMatcher对象
matches = bf.knnMatch(des1, des2, k=2)  # 对图片1和图片2的描述符进行匹配，k=2表示返回最佳的两个匹配结果

# 应用比例测试以获得良好的匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  # 进行比例测试
        good_matches.append(m)

# 显示匹配结果
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # 绘制匹配结果
cv2.imshow('Matches', img3)  # 显示匹配结果
cv2.waitKey(0)  # 等待按键输入
cv2.destroyAllWindows()  # 关闭所有窗口
