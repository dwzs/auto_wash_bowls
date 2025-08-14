import cv2
import numpy as np
import time
import json
import os
from sense.ellipses_extracter import EllipseExtracter

def main():
    # 读取配置
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    camera_id = config.get("camera_id", 0)
    flip_left_right = config.get("flip_left_right", False)
    camera_resolution = config.get("camera_resolution", {"width": 640, "height": 480})
    ellipse_config = config.get("ellipse_extracter", {})
    
    # Initialize webcam
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set resolution from config
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution["height"])
    
    # Print actual resolution being used
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"摄像头分辨率: {int(actual_width)}x{int(actual_height)}")
    print(f"左右翻转: {'启用' if flip_left_right else '禁用'}")
    
    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam.")
        return
    
    # Create ellipse detector instance from config
    detector = EllipseExtracter(
        min_points=ellipse_config.get("min_points", 200),
        ellipse_params=ellipse_config.get("ellipse_params", {
            'num_samples': 20,
            'sample_size': 5,
            'tolerance': 2
        }),
        confidence_threshold=ellipse_config.get("confidence_threshold", 0.5)
    )
    
    print("Starting real-time ellipse detection...")
    print("Press 'q' to quit")
    
    while True:
        start_time = time.time()
        
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # 根据配置进行左右翻转
        if flip_left_right:
            frame = cv2.flip(frame, 1)  # 1表示水平翻转（左右翻转）
        
        # Process the frame using the optimized method
        detected_ellipses, confidences = detector.process_frame(frame)
        original_ellipses_image = detector.get_original_ellipses_image()
        edge_ellipses_image = detector.get_edge_ellipses_image()
        
        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
        cv2.putText(original_ellipses_image, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display detected ellipses count
        high_conf_count = sum(1 for c in confidences if c >= detector.confidence_threshold)
        cv2.putText(original_ellipses_image, f"Detected: {high_conf_count}/{len(detected_ellipses)}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the frame
        cv2.imshow('Ellipse Detection', original_ellipses_image)
        
        # Display edge image for debugging
        cv2.imshow('Edges', edge_ellipses_image)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
