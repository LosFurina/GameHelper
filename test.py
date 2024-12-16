import pyautogui
import cv2
import numpy as np
import time

time.sleep(2)
# 获取屏幕截图
screenshot = pyautogui.screenshot()
frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

# 转换为灰度图
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 阈值分割（检测高亮区域）
_, thresh = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)

# 形态学操作，清除噪声并增强目标区域
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# 设置一个阈值，筛选出符合高宽比的轮廓
contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(contours)

# 遍历每个轮廓
for cnt in contours:
    area = cv2.contourArea(cnt)  # 计算轮廓面积

    if area > 5000 and area < 20000:  # 根据面积过滤小轮廓
        print(f"Contour Area: {area}")
        # 判断轮廓的凸性，非凸的轮廓可能有褶皱
        if cv2.isContourConvex(cnt):  # 如果是凸轮廓，跳过
            continue

        # 利用多边形逼近来判断轮廓是否平滑
        epsilon = 0.04 * cv2.arcLength(cnt, True)  # 计算多边形逼近的精度
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 如果逼近后的轮廓点数比较少，说明是平滑的矩形等，舍去
        if len(approx) > 200:  # 可以根据情况调整这个值
            print(len(approx))
            continue

        # 进一步判断轮廓形状（如通过矩形外接框的长宽比）
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h  # 宽高比
        if aspect_ratio > 0.5 and aspect_ratio < 1.0:  # 宽高比接近 1，可能是树形
            print(aspect_ratio)
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)  # 绘制轮廓

# 显示结果
cv2.imshow("Detected Trees", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

