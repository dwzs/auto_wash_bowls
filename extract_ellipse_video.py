import cv2
import numpy as np
import time
from extract_ellipse import EllipseExtracter

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Get frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read from webcam.")
        return
    
    # Create ellipse detector instance
    detector = EllipseExtracter(
        min_points=200,
        ellipse_params={
            'num_samples': 20,
            'sample_size': 5,
            'tolerance': 2
        },
        confidence_threshold=0.5
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
        
        # Process the frame using the optimized method
        detected_ellipses, confidences = detector.process_frame(frame)
        original_ellipses_image = detector.get_original_ellipses_image()
        edge_ellipses_image = detector.get_edge_ellipses_image()
        
        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        # print(f"elapsed_time: {elapsed_time}")

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
