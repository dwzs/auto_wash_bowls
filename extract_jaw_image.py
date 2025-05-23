import cv2
import numpy as np

def nothing(x):
    pass

# 读取彩色图像
image = cv2.imread('./resources/jaw1.jpg')

# 缩小图像为原来的二分之一
resized_image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))
cv2.imshow('resized_image', resized_image)

# Convert the image to HSV
hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)

# 创建一个窗口
cv2.namedWindow('Trackbars')

# 创建滑块用于调节颜色范围
cv2.createTrackbar('Hue', 'Trackbars', 0, 180, nothing)
cv2.createTrackbar('Saturation', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('Value', 'Trackbars', 100, 255, nothing)

while True:
    # 获取滑块的值
    hue = cv2.getTrackbarPos('Hue', 'Trackbars')
    saturation = cv2.getTrackbarPos('Saturation', 'Trackbars')
    value = cv2.getTrackbarPos('Value', 'Trackbars')

    # Define the range for the selected color in HSV
    lower_bound = np.array([hue - 10, saturation - 50, value - 50])
    upper_bound = np.array([hue + 10, saturation + 50, value + 50])

    # Create a mask for the selected color
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Invert the mask to get the background
    background_mask = cv2.bitwise_not(mask)

    # Create a white background
    white_background = np.full_like(resized_image, 255)

    # Extract the selected color region
    color_region = cv2.bitwise_and(resized_image, resized_image, mask=mask)

    # Combine the color region with the white background
    result = cv2.bitwise_or(color_region, white_background, mask=background_mask)

    # Display the result
    cv2.imshow('Selected Color Region with White Background', result)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
