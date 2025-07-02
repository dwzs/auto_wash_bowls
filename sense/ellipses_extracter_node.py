import json
import os
import rclpy
from rclpy.node import Node
import cv2
from sense.ellipses_extracter import EllipseExtracter

from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class EllipsesExtracterNode(Node):
    def __init__(self):
        super().__init__('ellipses_extracter_node')
        # 读取配置
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        self.camera_id = config.get("camera_id", 0)
        self.ellipse_config = config.get("ellipse_extracter", {})
        self.ellipse_topic = config.get("ellipse_topic", "/sense/ellipses")
        self.ellipse_image_topic = config.get("ellipse_image_topic", "/sense/ellipses/image")
        self.timer_period = config.get("timer_period", 0.1)
        self.publish_image = config.get("publish_image", True)

        self.publisher = self.create_publisher(Detection2DArray, self.ellipse_topic, 10)
        self.image_pub = self.create_publisher(Image, self.ellipse_image_topic, 10)
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            self.get_logger().error(f"无法打开摄像头 {self.camera_id}")
            exit(1)
        self.extracter = EllipseExtracter(
            min_points=self.ellipse_config.get("min_points", 200),
            ellipse_params=self.ellipse_config.get("ellipse_params", {}),
            confidence_threshold=self.ellipse_config.get("confidence_threshold", 0.2)
        )
        self.bridge = CvBridge()
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("未能读取摄像头帧")
            return

        ellipses, confidences = self.extracter.process_frame(frame)
        msg = Detection2DArray()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"

        for ellipse, confidence in zip(ellipses, confidences):
            center, axes, angle = ellipse  # center: (x, y), axes: (major, minor), angle: deg
            detection = Detection2D()
            detection.header = msg.header

            # 填充 bbox
            bbox = BoundingBox2D()
            bbox.center.position.x = float(center[0])
            bbox.center.position.y = float(center[1])
            bbox.center.theta = float(angle) * 3.1415926 / 180.0  # 角度转弧度
            bbox.size_x = float(axes[0])  # 长轴
            bbox.size_y = float(axes[1])  # 短轴
            detection.bbox = bbox

            # 填充置信度
            hypo = ObjectHypothesisWithPose()
            hypo.hypothesis.class_id = "ellipse"
            hypo.hypothesis.score = float(confidence)
            detection.results.append(hypo)

            msg.detections.append(detection)

        self.publisher.publish(msg)

        # 发布带有椭圆标记的原图
        if self.publish_image:
            marked_img = self.extracter.get_original_ellipses_image()
            if marked_img is not None:
                img_msg = self.bridge.cv2_to_imgmsg(marked_img, encoding="bgr8")
                img_msg.header = msg.header
                self.image_pub.publish(img_msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = EllipsesExtracterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
