import cv2
import numpy as np
import matplotlib.pyplot as plt
# 读取图片 与之前的西电校门图片大小同意
image = cv2.imread('rainbow.jpg')
image = cv2.resize(image, (534, 400))
rainbow_image = np.zeros((400, 534, 3), np.uint8)
# 将图像转换为HSV颜色空间 便于颜色的过滤
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# 定义彩虹的颜色范围（最小和最大HSV值）
# RED
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

# ORANGE
lower_orange = np.array([10, 100, 100])
upper_orange = np.array([25, 255, 255])

# YELLOW
lower_yellow = np.array([25, 100, 100])
upper_yellow = np.array([35, 255, 255])

# GREEN
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

# BLUE
lower_blue = np.array([85, 100, 100])
upper_blue = np.array([125, 255, 255])

# PURPLE
lower_purple = np.array([125, 100, 100])
upper_purple = np.array([170, 255, 255])

# 创建掩膜
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask_green = cv2.inRange(hsv, lower_green, upper_green)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

# Combine masks
rainbow_mask = mask_red1 | mask_red2 | mask_orange | mask_yellow | mask_green | mask_blue | mask_purple

kernel = np.ones((5, 5), np.uint8)
rainbow_mask = cv2.morphologyEx(rainbow_mask, cv2.MORPH_CLOSE, kernel)
rainbow_mask = cv2.morphologyEx(rainbow_mask, cv2.MORPH_OPEN, kernel)

# 提取彩虹区域
rainbow_extracted = cv2.bitwise_and(image, image, mask=rainbow_mask)
#把黑色部分对应的原本图像进行保留
for i in range(400) :
    for j in range(534) :
        if np.sum(rainbow_extracted[i][j]) == 0 :
            rainbow_image[i][j] = image[i][j]
        else : rainbow_image[i][j] = [0, 0, 0]


cv2.imwrite('rainbow_extracted.jpg', rainbow_extracted)
cv2.imwrite('final_rainbow_extracted.jpg', rainbow_image)
plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(rainbow_extracted, cmap='gray')
plt.title('Extract Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(rainbow_image, cmap='gray')
plt.title('Final Image'), plt.xticks([]), plt.yticks([])
plt.show()