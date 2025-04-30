import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from extract_ellipse import EllipseExtracter


if __name__ == "__main__":
    # 设置参数
    image_path = "./resources/bowls1.png"
    image_path = "./resources/bowls2.png"
    show_image = True
    save_image = True
    result_path = "./results"

    # 创建椭圆提取器实例
    extractor = EllipseExtracter(
        min_points=300,
        ellipse_params={
            'num_samples': 20,
            'sample_size': 5,
            'tolerance': 2
        },
        confidence_threshold=0.5
    )
    
    # 处理图像
    start_time = time.time()
    ellipses, confidences = extractor.process_image(image_path)
    end_time = time.time()
    print(f"处理时间: {end_time - start_time} 秒")

    for i, (ellipse, confidence) in enumerate(zip(ellipses, confidences)):
        print(f"椭圆 {i+1}: 置信度 = {confidence:.2f}")

    original_image = extractor.get_original_image()
    edge_image = extractor.get_edge_image()
    gray_image = extractor.get_gray_image()

    point_sets_image = extractor.get_point_sets_image()
    edge_ellipses_image = extractor.get_edge_ellipses_image()
    original_ellipses_image = extractor.get_original_ellipses_image()
    # 转换为RGB颜色空间以便正确显示
    rgb_ellipses_image = cv2.cvtColor(original_ellipses_image, cv2.COLOR_BGR2RGB)

    if show_image:
        plt.figure(figsize=(15, 9))
        
        plt.subplot(2, 3, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        
        plt.subplot(2, 3, 2)
        plt.title('Grayscale Image')
        plt.imshow(gray_image, cmap='gray')
        
        plt.subplot(2, 3, 3)
        plt.title('Edge Detection')
        plt.imshow(edge_image, cmap='gray')
        
        plt.subplot(2, 3, 4)
        plt.title(f'Filtered Points')
        plt.imshow(point_sets_image)
        
        plt.subplot(2, 3, 5)
        plt.title('edge_ellipses_image')
        plt.imshow(edge_ellipses_image)
        
        plt.subplot(2, 3, 6)
        plt.title('original_ellipses_image')
        plt.imshow(rgb_ellipses_image)
        
        plt.tight_layout()
        plt.show()

    if save_image:
        # Create the result directory if it doesn't exist
        if not os.path.exists(result_path):
            os.makedirs(result_path)
            
        print(f"保存图像到 {result_path}")
        cv2.imwrite(f"{result_path}/1_gray_image.png", gray_image)
        cv2.imwrite(f"{result_path}/2_edge_image.png", edge_image)
        cv2.imwrite(f"{result_path}/3_point_sets_image.png", point_sets_image)
        cv2.imwrite(f"{result_path}/4_edge_ellipses_image.png", edge_ellipses_image)
        cv2.imwrite(f"{result_path}/5_original_ellipses_image.png", original_ellipses_image)
