#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-100机器人硬件驱动ROS2节点"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import threading
import time
import sys
import os
import json
import math

# 导入硬件底层接口
from control.so100_driver import So100Driver

class So100DriverNode(Node):
    def __init__(self):
        # 直接加载配置文件
        config_path = os.path.join(os.path.dirname(__file__), "config", "so100_config.json")
        with open(config_path, "r") as f:
            self.cfg = json.load(f)
        
        # 从配置读取参数到成员变量
        self.node_name = self.cfg["ros"]["node_name"]
        self.joint_state_topic = self.cfg["ros"]["topics"]["joint_states"]
        self.joint_cmd_topic = self.cfg["ros"]["topics"]["joint_commands"]
        self.pub_rate = self.cfg["ros"]["pub_rate"]
        self.queue_size = self.cfg["ros"]["queue_size"]
        self.servo_ids = self.cfg["robot"]["servo_ids"]
        
        # 初始化ROS节点
        super().__init__(self.node_name)
        
        # 初始化硬件驱动
        self.driver = So100Driver()
        if not hasattr(self.driver, 'initialized') or not self.driver.initialized:
            self.get_logger().error('硬件驱动初始化失败')
            raise RuntimeError('硬件驱动初始化失败')
        
        # ROS2 通信接口
        self.joint_state_pub = self.create_publisher(
            JointState, 
            self.joint_state_topic,
            self.queue_size
        )
        
        self.joint_cmd_sub = self.create_subscription(
            JointState,
            self.joint_cmd_topic,
            self.joint_cmd_callback,
            self.queue_size
        )
        
        # 状态发布定时器
        self.state_timer = self.create_timer(
            1.0/self.pub_rate, 
            self.publish_joint_state
        )
        
        # 关节名称（与URDF保持一致）
        self.joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        
        # 线程锁
        self.cmd_lock = threading.Lock()
        
        self.get_logger().info(f'{self.node_name} 已启动')
        self.get_logger().info(f'发布话题: {self.joint_state_topic}')
        self.get_logger().info(f'订阅话题: {self.joint_cmd_topic}')
        self.get_logger().info(f'发布频率: {self.pub_rate} Hz')
    
    def joint_cmd_callback(self, msg: JointState):
        """接收关节控制命令 - 支持NaN跳过"""
        try:
            if len(msg.position) != len(self.servo_ids):
                self.get_logger().error(f'关节数量不匹配')
                return
            
            # 直接按顺序处理，NaN表示跳过该关节
            for i, position in enumerate(msg.position):
                if not math.isnan(position):  # 只控制非NaN的关节
                    servo_id = self.servo_ids[i]
                    success = self.driver.write_joint(servo_id, position)
                    if not success:
                        self.get_logger().error(f'关节{servo_id}命令写入失败')
                
        except Exception as e:
            self.get_logger().error(f'处理关节命令出错: {e}')
    
    def publish_joint_state(self):
        """发布当前关节状态"""
        try:
            # 读取当前关节位置
            current_positions = self.driver.read_joints()
            
            if current_positions is None:
                self.get_logger().warn('读取关节状态失败')
                return
            
            # 构造JointState消息
            joint_state = JointState()
            joint_state.header.stamp = self.get_clock().now().to_msg()
            joint_state.header.frame_id = ''
            joint_state.name = self.joint_names
            joint_state.position = current_positions
            joint_state.velocity = []  # 可选：如果有速度反馈
            joint_state.effort = []    # 可选：如果有力矩反馈
            
            # 发布状态
            self.joint_state_pub.publish(joint_state)
            
        except Exception as e:
            self.get_logger().error(f'发布关节状态出错: {e}')
    
    def destroy_node(self):
        """节点销毁时的清理工作"""
        try:
            self.get_logger().info('正在关闭SO-100硬件驱动节点...')
            
            # 停止硬件驱动
            if hasattr(self, 'driver'):
                self.driver.stop()
            
            # 调用父类销毁方法
            super().destroy_node()
            
            self.get_logger().info('SO-100硬件驱动节点已关闭')
            
        except Exception as e:
            self.get_logger().error(f'关闭节点出错: {e}')

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        # 创建节点
        node = So100DriverNode()
        
        # 运行节点
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print('用户中断')
    except Exception as e:
        print(f'节点运行出错: {e}')
    finally:
        # 清理资源
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
