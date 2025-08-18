#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""机械臂跟随目标对象"""

import numpy as np
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
        self.dead_zone = 20  # 死区像素，避免震荡
        self.move_distance = 0.02  # 固定移动距离(m)
        
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
        du_pixel = self.blue_center[0] - self.red_center[0]
        dv_pixel = self.blue_center[1] - self.red_center[1]
        
        # 检查是否在死区内
        if abs(du_pixel) < self.dead_zone and abs(dv_pixel) < self.dead_zone:
            self.get_logger().debug("目标在死区内，不移动")
            return
        
        # 计算方向向量并归一化，然后乘以固定距离
        # u -> -x, v -> -z (图像坐标系到机械臂坐标系的映射)
        direction_vector = np.array([-du_pixel, 0, -dv_pixel])
        
        # 计算向量长度
        vector_length = np.linalg.norm(direction_vector)
        if vector_length == 0:
            return
        
        # 归一化并乘以固定移动距离
        unit_vector = direction_vector / vector_length
        move_vector = unit_vector * self.move_distance
        
        self.get_logger().info(
            f"红色中心: {self.red_center}, 蓝色中心: {self.blue_center}, "
            f"像素差异: ({du_pixel}, {dv_pixel}), 移动向量: {move_vector}")
        
        # 执行移动
        try:
            success = self.arm.move_to_direction_abs(
                move_vector, timeout=1.0, tolerance=0.01, tcp=True)
            if not success:
                self.get_logger().warn("机械臂移动失败")
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
