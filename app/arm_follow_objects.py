#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机械臂跟随目标对象"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import sys
import os

# 添加控制模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from control.arm_controller import Arm

class ArmFollowObjects(Node):
    def __init__(self):
        super().__init__('arm_follow_objects')
        
        # 初始化机械臂
        self.arm = Arm()
        
        # 订阅红色和蓝色中心点
        self.sub_red = self.create_subscription(
            Point, '/sense/colors/red/center', self.red_center_callback, 10)
        self.sub_blue = self.create_subscription(
            Point, '/sense/colors/blue/center', self.blue_center_callback, 10)
        
        # 存储中心点位置
        self.red_center = None
        self.blue_center = None
        
        # 控制参数
        self.pixel_to_meter = 0.001  # 像素到米的转换比例 (1像素 = 1mm)
        self.dead_zone = 20  # 死区像素，避免震荡
        self.max_move_distance = 0.05  # 最大单次移动距离(m)
        
        # 定时器进行跟踪控制
        self.timer = self.create_timer(0.2, self.follow_control)  # 5Hz控制频率
        
        self.get_logger().info("机械臂跟随对象节点启动")

    def red_center_callback(self, msg):
        """红色中心点回调（夹爪位置）"""
        self.red_center = (int(msg.x), int(msg.y))

    def blue_center_callback(self, msg):
        """蓝色中心点回调（目标对象位置）"""
        self.blue_center = (int(msg.x), int(msg.y))

    def follow_control(self):
        """跟踪控制主循环"""
        # 检查是否有有效的目标点
        if self.red_center is None or self.blue_center is None:
            return
        
        # 计算像素差异 (目标 - 当前位置)
        dx_pixel = self.blue_center[0] - self.red_center[0]
        dy_pixel = self.blue_center[1] - self.red_center[1]
        
        # 检查是否在死区内
        if abs(dx_pixel) < self.dead_zone and abs(dy_pixel) < self.dead_zone:
            self.get_logger().debug("目标在死区内，不移动")
            return
        
        # 转换像素差异为机械臂移动向量
        # 注意：图像坐标系Y轴向下，机械臂坐标系Y轴向上
        dx_meter = dx_pixel * self.pixel_to_meter
        dy_meter = -dy_pixel * self.pixel_to_meter  # 反向Y轴
        dz_meter = 0.0  # 不改变Z轴
        
        # 限制最大移动距离
        dx_meter = max(-self.max_move_distance, min(self.max_move_distance, dx_meter))
        dy_meter = max(-self.max_move_distance, min(self.max_move_distance, dy_meter))
        
        move_vector = [dx_meter, dy_meter, dz_meter]
        
        self.get_logger().info(
            f"红色中心: {self.red_center}, 蓝色中心: {self.blue_center}, "
            f"像素差异: ({dx_pixel}, {dy_pixel}), 移动向量: {move_vector}")
        
        # 执行移动
        try:
            # success = self.arm.move_to_direction_abs(
            #     move_vector, timeout=1.0, tolerance=0.01, tcp=True)
            # if not success:
            #     self.get_logger().warn("机械臂移动失败")
            print(f"move_vector: {move_vector}")
        except Exception as e:
            self.get_logger().error(f"移动控制错误: {e}")

    def destroy_node(self):
        """安全销毁节点"""
        if hasattr(self, 'arm'):
            self.arm.destroy_node()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ArmFollowObjects()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("用户中断")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
