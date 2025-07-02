#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-100夹爪控制器 - 完整功能版本"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import json
import time
import os
from typing import Optional, List
import threading
import math

class GripperController(Node):
    """夹爪控制器 - 完整功能，包含ROS通信和控制逻辑"""
    
    def __init__(self):
        super().__init__('gripper_controller')
        
        # 加载配置
        config_path = os.path.join(os.path.dirname(__file__), "config", "gripper_config.json")
        self.config = self._load_config(config_path)
        
        # 夹爪参数
        self.joint_index = self.config.get("joint_index", 5)  # 夹爪在关节数组中的索引
        self.open_position = self.config.get("open_position", 1.2)    # 打开位置
        self.close_position = self.config.get("close_position", -0.15)  # 关闭位置
        self.tolerance = self.config.get("tolerance", 0.05)  # 位置容差
        
        # ROS通信配置
        ros_config = self.config.get("ros", {})
        self.cmd_topic = ros_config.get("cmd_topic", "/so100/cmd/joints")
        self.state_topic = ros_config.get("state_topic", "/so100/state/joints")
        
        # ROS发布器和订阅器
        self.joint_cmd_pub = self.create_publisher(JointState, self.cmd_topic, 10)
        self.current_joints = None
        self.lock = threading.Lock()
        
        self.joint_state_sub = self.create_subscription(
            JointState, 
            self.state_topic, 
            self._joint_state_callback, 
            10
        )
        
        # 启动后台线程处理ROS消息
        self.spin_thread = threading.Thread(target=self._spin_thread, daemon=True)
        self.spin_thread.start()
        
        print(f"夹爪控制器初始化完成 - 关节索引:{self.joint_index}, 开:{self.open_position}, 关:{self.close_position}")
        self.get_logger().info("夹爪控制器就绪")
    
    def _load_config(self, config_file: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"警告: 配置文件未找到 {config_file}, 使用默认配置")
            return {}
        except json.JSONDecodeError as e:
            print(f"警告: 配置文件格式错误 {e}, 使用默认配置")
            return {}
    
    def _joint_state_callback(self, msg: JointState):
        """关节状态回调"""
        with self.lock:
            # print(f"DEBUG: 收到关节状态数据: {msg.position}")
            self.current_joints = list(msg.position)
        # self.current_joints = list(msg.position)
        # print(f"DEBUG: 收到关节状态数据: {msg.position}")

    def _get_current_joints(self) -> Optional[List[float]]:
        """获取当前关节状态"""
        with self.lock:
            return self.current_joints.copy() if self.current_joints else None
    
    def _send_joint_command(self, joint_positions: List[float]):
        """发送关节命令"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = joint_positions
        self.joint_cmd_pub.publish(msg)
    
    def _send_gripper_command(self, position: float):
        """发送夹爪命令 - 其他关节用NaN跳过"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = [
            math.nan,  # joint_1 - 跳过
            math.nan,  # joint_2 - 跳过  
            math.nan,  # joint_3 - 跳过
            math.nan,  # joint_4 - 跳过
            math.nan,  # joint_5 - 跳过
            position   # joint_6 - 夹爪位置
        ]
        self.joint_cmd_pub.publish(msg)
    
    def _wait_for_position(self, target_pos: float, timeout: float) -> bool:
        """等待到达目标位置"""
        if timeout <= 0:
            return True
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            current_pos = self.get_joint()
            if current_pos and abs(current_pos - target_pos) < self.tolerance:
                return True
            time.sleep(0.05)  # 50ms检查一次
        return False
    
    def _spin_thread(self):
        """后台线程处理ROS消息"""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
    
    # ==================== 对外接口 ====================
    
    def open(self, timeout: float = 5.0) -> bool:
        """打开夹爪
        
        Args:
            timeout: 超时时间，>0时等待到位，<=0时不等待
            
        Returns:
            成功返回True，失败或超时返回False
        """
        self._send_gripper_command(self.open_position)
        self.get_logger().info("打开夹爪")
        return self._wait_for_position(self.open_position, timeout)
    
    def close(self, timeout: float = 5.0) -> bool:
        """关闭夹爪
        
        Args:
            timeout: 超时时间，>0时等待到位，<=0时不等待
            
        Returns:
            成功返回True，失败或超时返回False
        """
        self._send_gripper_command(self.close_position)
        self.get_logger().info("关闭夹爪")
        return self._wait_for_position(self.close_position, timeout)
    
    def set_joint(self, position: float, timeout: float = 5.0) -> bool:
        """设置夹爪关节位置
        
        Args:
            position: 目标位置
            timeout: 超时时间，>0时等待到位，<=0时不等待
            
        Returns:
            成功返回True，失败或超时返回False
        """
        # 限制位置范围
        min_pos = min(self.open_position, self.close_position)
        max_pos = max(self.open_position, self.close_position)
        position = max(min_pos, min(max_pos, position))
        
        self._send_gripper_command(position)
        self.get_logger().info(f"设置夹爪位置: {position:.3f}")
        return self._wait_for_position(position, timeout)
    
    def get_joint(self) -> Optional[float]:
        """获取夹爪关节位置
        
        Returns:
            当前夹爪位置，如果无法获取则返回None
        """
        current_joints = self._get_current_joints()
        if not current_joints or len(current_joints) <= self.joint_index:
            return None
        return current_joints[self.joint_index]
    
    def set_opening_percentage(self, percent: float, timeout: float = 5.0) -> bool:
        """按百分比设置夹爪开度
        
        Args:
            percent: 开度百分比 (0-100)，0为完全关闭，100为完全打开
            timeout: 超时时间，>0时等待到位，<=0时不等待
            
        Returns:
            成功返回True，失败或超时返回False
        """
        # 限制百分比范围
        percent = max(0.0, min(100.0, percent))
        
        # 计算目标位置
        range_total = self.open_position - self.close_position
        target_pos = self.close_position + (percent / 100.0) * range_total
        
        self._send_gripper_command(target_pos)
        self.get_logger().info(f"设置夹爪开度: {percent:.1f}%")
        
        return self._wait_for_position(target_pos, timeout)
    
    def get_opening_percentage(self) -> Optional[float]:
        """获取夹爪开度百分比
        
        Returns:
            开度百分比 (0-100)，如果无法获取则返回None
        """
        position = self.get_joint()
        if position is None:
            return None
        
        # 计算百分比
        range_total = self.open_position - self.close_position
        if abs(range_total) < 1e-6:  # 避免除零
            return 0.0
        
        percentage = (position - self.close_position) / range_total * 100.0
        return max(0.0, min(100.0, percentage))
    

def main():
    """测试函数"""
    rclpy.init()
    gripper = None
    
    try:
        gripper = GripperController()
        
        # # 现在回调会在后台自动更新
        # while True:
        #     joints = gripper.get_joint()
        #     if joints is not None:
        #         print(f"当前位置: {joints:.3f}")
        #     time.sleep(0.01)  # 可以设置更长的间隔

        joint_max = 1.2
        joint_min = -0.15
        step = 0.001
        going_up = True
        joint = joint_min

        while True:
            gripper.set_joint(joint, timeout=0)
            print(f"关节位置: {joint + step * (1 if going_up else -1):.3f} {'↑' if going_up else '↓'}")
            
            if going_up:
                joint += step
                if joint >= joint_max:
                    going_up = False
            else:
                joint -= step
                if joint <= joint_min:
                    going_up = True
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("用户中断")
    finally:
        if gripper:
            gripper.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
