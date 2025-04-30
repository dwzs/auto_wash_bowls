import cv2
import numpy as np

# 1. 构建一个600 * 600的二值图，背景为黑
image = np.zeros((600, 600), dtype=np.uint8)

# 2. 在二值图中构建一条水平方向的20像素长度单像素宽度的直线
# 放置在图像中央偏上的位置
start_point_h = (290, 200)
end_point_h = (310, 200)
cv2.line(image, start_point_h, end_point_h, 255, 1)

# 3. 在二值图中构建一条斜着45度的20像素长度单像素宽度的直线
# 计算45度角的直线端点（长度约为20像素）
length = 20
dx = int(length * np.cos(np.radians(45)))
dy = int(length * np.sin(np.radians(45)))
start_point_d = (290, 250)
end_point_d = (start_point_d[0] + dx, start_point_d[1] + dy)
cv2.line(image, start_point_d, end_point_d, 255, 1)

# 4. 在二值图中构建一个半径为10像素的填充圆形
center_filled_circle = (300, 350)
radius = 10
cv2.circle(image, center_filled_circle, radius, 255, -1)  # -1表示填充

# 5. 在二值图中构建一个半径为10像素的空心圆形，边界为单像素
center_hollow_circle = (300, 400)
cv2.circle(image, center_hollow_circle, radius, 255, 1)  # 1表示边界宽度为1像素

# 6. 构建类似上面4和5的矩形

# 填充矩形
rect_filled_center = (300, 450)
rect_filled_size = (20, 20)  # 宽20，高20
top_left = (rect_filled_center[0] - rect_filled_size[0]//2, 
           rect_filled_center[1] - rect_filled_size[1]//2)
bottom_right = (rect_filled_center[0] + rect_filled_size[0]//2, 
               rect_filled_center[1] + rect_filled_size[1]//2)
cv2.rectangle(image, top_left, bottom_right, 255, -1)  # -1表示填充

# 空心矩形，边界为单像素
rect_hollow_center = (300, 500)
top_left = (rect_hollow_center[0] - rect_filled_size[0]//2, 
           rect_hollow_center[1] - rect_filled_size[1]//2)
bottom_right = (rect_hollow_center[0] + rect_filled_size[0]//2, 
               rect_hollow_center[1] + rect_filled_size[1]//2)
cv2.rectangle(image, top_left, bottom_right, 255, 1)  # 1表示边界宽度为1像素

# 保存和显示图像
cv2.imwrite('./resources/binary_shapes.png', image)
cv2.imshow('Binary Image with Shapes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
