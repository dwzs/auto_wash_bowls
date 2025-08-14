import os
import json
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sense.color_extracter import ColorExtracter

class ColorExtracterNode(Node):
    def __init__(self):
        super().__init__('color_extracter_node')

        # 读取配置
        cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(cfg_path, "r") as f:
            config = json.load(f)

        self.timer_period = config.get("timer_period", 0.1)
        self.flip_left_right = config.get("flip_left_right", False)

        # 初始化颜色提取器
        self.extracter = ColorExtracter(min_area=100, config_path=cfg_path)
        self.cap = self.extracter.create_camera_capture()
        if not self.cap.isOpened():
            self.get_logger().error(f"无法打开摄像头 {self.extracter.camera_id}")
            raise RuntimeError("无法打开摄像头")

        # Publisher - 发布中心点位置
        self.pub_red_center = self.create_publisher(Point, "/sense/colors/red/center", 10)
        self.pub_blue_center = self.create_publisher(Point, "/sense/colors/blue/center", 10)

        # 定时器
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # 颜色阈值
        self.red_low, self.red_high = [176, 20, 20], [184, 255, 255]  # 红色跨零闭环
        self.blue_low, self.blue_high = [100, 150, 100], [130, 255, 255]  # 蓝色

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("读取摄像头帧失败")
            return

        # 使用ColorExtracter的高级接口
        if self.flip_left_right:
            frame = cv2.flip(frame, 1)
        self.extracter.set_original_image(frame)
        
        # 获取红色和蓝色对象图像
        red_image = self.extracter.get_color_objects_image(self.red_low, self.red_high)
        blue_image = self.extracter.get_color_objects_image(self.blue_low, self.blue_high)
        
        # 合并红色和蓝色图像到一个图上
        combined_image = cv2.bitwise_or(red_image, blue_image)
        
        # 获取轮廓边缘图像用于计算中心点
        red_edges = self.extracter.get_color_objects_edage_image(self.red_low, self.red_high)
        blue_edges = self.extracter.get_color_objects_edage_image(self.blue_low, self.blue_high)
        
        # 计算轮廓中心点
        red_center = self._calculate_contour_center(red_edges)
        blue_center = self._calculate_contour_center(blue_edges)
        
        # 发布中心点位置
        if red_center is not None:
            red_point = Point()
            red_point.x = float(red_center[0])
            red_point.y = float(red_center[1])
            red_point.z = 0.0
            self.pub_red_center.publish(red_point)
        
        if blue_center is not None:
            blue_point = Point()
            blue_point.x = float(blue_center[0])
            blue_point.y = float(blue_center[1])
            blue_point.z = 0.0
            self.pub_blue_center.publish(blue_point)
        
        # 在合并图像上标记中心点
        vis_combined = combined_image.copy()
        if red_center is not None:
            cv2.circle(vis_combined, red_center, 10, (0, 0, 255), -1)  # 红色圆点
            cv2.putText(vis_combined, f"Red: {red_center}", (red_center[0]+15, red_center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if blue_center is not None:
            cv2.circle(vis_combined, blue_center, 10, (255, 0, 0), -1)  # 蓝色圆点
            cv2.putText(vis_combined, f"Blue: {blue_center}", (blue_center[0]+15, blue_center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 显示单个合并图像
        cv2.imshow("Red & Blue Objects", vis_combined)
        cv2.waitKey(1)

    def _calculate_contour_center(self, edge_image: np.ndarray) -> tuple:
        """计算轮廓的中心点"""
        contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(largest_contour) < self.extracter.min_area:
            return None
        
        # 计算轮廓的矩
        M = cv2.moments(largest_contour)
        
        if M["m00"] == 0:
            return None
        
        # 计算中心点
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        return (center_x, center_y)

    def destroy_node(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ColorExtracterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
