#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-100机器人控制接口"""

import rclpy
import numpy as np
import time
from typing import List, Union, Optional, Dict, Any, Tuple
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation


def format_to_2dp(value: Any) -> Any:
    """将数字或包含数字的列表/数组格式化为小数点后两位
    
    参数:
        value: 要格式化的值，可以是数字、列表或NumPy数组
        
    返回:
        格式化后的值，保持原始类型
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        if isinstance(value, np.ndarray):
            # 对于numpy数组，创建一个新的格式化数组
            formatted = np.round(value, 2)
            if isinstance(value, np.ndarray):
                return formatted
            else:
                return type(value)(formatted)  # 转换回原始类型
        else:
            # 对于列表或元组，迭代处理每个元素
            return type(value)([format_to_2dp(x) for x in value])
    elif isinstance(value, (int, float, np.number)):
        # 对于单个数字，直接四舍五入到两位小数
        return round(float(value), 2)
    else:
        # 对于其他类型，原样返回
        return value


class Gripper:
    """夹爪控制类"""
    
    def __init__(self, robot_instance):
        """初始化夹爪控制器
        
        参数:
            robot_instance: 父机器人实例，用于访问ROS接口
        """
        self._robot = robot_instance
        self._joint_index = 5  # 夹爪关节索引
        self.close_position = 1.0  # 夹爪关闭位置值
        self.open_position = 0.0   # 夹爪打开位置值
    
    def get_joint(self) -> Optional[float]:
        """获取夹爪位置"""
        if not self._robot.is_connected():
            self._robot.get_logger().warn("机器人未连接")
            return None
        
        joints = self._robot.get_joints()
        if joints is None:
            return None
            
        return joints[self._joint_index]
    
    def set_joint(self, position: float, wait=True, timeout=5.0, tolerance=0.01) -> bool:
        # 获取当前机械臂关节位置
        current_joints = self._robot.get_joints()
        if current_joints is None:
            return False
        
        # 构建完整的关节位置数组 (保留前5个机械臂关节，更新第6个夹爪关节)
        full_joint_positions = current_joints.copy()
        full_joint_positions[self._joint_index] = position
        
        # 发送完整的关节命令
        return self._robot.set_joints(full_joint_positions, wait, timeout, tolerance)
    
    def open(self, wait=True, timeout=5.0) -> bool:
        return self.set_joint(self.open_position, wait, timeout)
    
    def close(self, wait=True, timeout=5.0) -> bool:
        return self.set_joint(self.close_position, wait, timeout)


class Arm:
    """机械臂控制类"""
    
    def __init__(self, robot_instance):
        """初始化机械臂控制器
        
        参数:
            robot_instance: 父机器人实例，用于访问ROS接口
        """
        self._robot = robot_instance
        self.joint_index = [0, 1, 2, 3, 4]  # 机械臂关节索引
        # self.chain = self._build_chain_manually()
        
        self.joints_data = [
            {"translation": [0, 0, 0],            "orientation": [0, 0, 0], "rotate_axis": [0, 0, 1]},
            {"translation": [0, -0.0452, 0.0165], "orientation": [1.5708, 0, 0], "rotate_axis": [0, 1, 0]},
            {"translation": [0, 0.1025, 0.0306],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0]},
            {"translation": [0, 0.11257, 0.028],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0]},
            {"translation": [0, 0.0052, 0.1349],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0]},
            {"translation": [0, -0.0601, 0],      "orientation": [0, 0, 0], "rotate_axis": [0, 1, 0]},
            {"translation": [0, -0.1, 0],         "orientation": [0, 0, 0], "rotate_axis": [0, 0, 1]}
        ]

    def get_joints(self) -> Optional[np.ndarray]:
        """获取机械臂关节角度（不包括夹爪）"""
        if not self._robot.is_connected():
            self._robot.get_logger().warn("机器人未连接")
            return None
        
        joints = self._robot.get_joints()
        if joints is None:
            return None
            
        # Use list comprehension to get the joint angles at specified indices
        # return np.array([joints[i] for i in self.joint_index])
        return [joints[i] for i in self.joint_index]
    
    def forward_kinematics(self, joint_angles):
        """Compute the forward kinematics for the SO-100 robot arm.
        
        Parameters:
            joint_angles (list): A list of joint angles [j1, j2, j3, j4, j5].
            
        Returns:
            np.ndarray: The position and orientation (as a quaternion) of the TCP.
        """
        def transformation_matrix(translation, origin_orientation, rotation, angle):
            """Create a transformation matrix for a joint."""
            rot_matrix = Rotation.from_rotvec(np.array(rotation) * angle).as_matrix()
            origin_rot_matrix = Rotation.from_euler('xyz', origin_orientation).as_matrix()
            combined_rot_matrix = origin_rot_matrix @ rot_matrix
            transform = np.eye(4)
            transform[:3, :3] = combined_rot_matrix
            transform[:3, 3] = translation
            return transform
        
        joint_angles.insert(0, 0)  # 基座关节转角为0
        transformi = np.eye(4)
        for i, angle_i in enumerate(joint_angles):
            joint_i = self.joints_data[i]
            transformi = transformi @ transformation_matrix(
                joint_i["translation"], joint_i["orientation"], joint_i["rotate_axis"], angle_i
            )
        
        position = transformi[:3, 3]
        orientation = Rotation.from_matrix(transformi[:3, :3]).as_quat()
        
        return np.concatenate((position, orientation))

    def get_end_pose(self) -> Optional[Dict[str, np.ndarray]]:
        """获取末端执行器位姿"""
        if not self._robot.is_connected():
            self._robot.get_logger().warn("机器人未连接")
            return None
            
        arm_joint_angles = self.get_joints()

        # Use the forward_kinematics method
        tcp_pose = self.forward_kinematics(arm_joint_angles)
        
        return format_to_2dp(tcp_pose).tolist()

    def get_tcp_pose(self) -> Optional[Dict[str, np.ndarray]]:
        """获取末端执行器位姿"""
        if not self._robot.is_connected():
            self._robot.get_logger().warn("机器人未连接")
            return None
            
        arm_joint_angles = self.get_joints()
        tcp_joint_angles = arm_joint_angles + [0]

        # Use the forward_kinematics method
        tcp_pose = self.forward_kinematics(tcp_joint_angles)
        
        return format_to_2dp(tcp_pose).tolist()
    
    def set_joints(self, positions: np.ndarray, wait=True, timeout=10.0, tolerance=0.01) -> bool:
        if not self._robot.is_connected():
            self._robot.get_logger().warn("机器人未连接")
            return False
        
        # 确保关节数量正确
        if len(positions) != len(self.joint_index):
            self._robot.get_logger().error(f"机械臂关节数量错误: 需要{len(self.joint_index)}个关节位置")
            return False
        
        # 获取当前夹爪位置
        current_gripper_pos = self._robot.gripper.get_joint()
        if current_gripper_pos is None:
            return False
        
        # 构建完整的关节位置数组（包括夹爪）
        full_joint_positions = np.append(positions, current_gripper_pos)
        
        # 发送完整的关节命令
        return self._robot.set_joints(full_joint_positions, wait, timeout, tolerance)

    def move_home(self, wait=True) -> bool:
        """将机械臂移动到初始位置（所有关节角度为0）"""
        home_positions = np.zeros(len(self.joint_index))
        return self.set_joints(home_positions, wait=wait)
    


class So100Robot(Node):
    """SO-100机器人控制接口主类"""
    
    def __init__(self):
        # 初始化ROS
        if not rclpy.ok():
            rclpy.init()
            
        super().__init__('So100Robot')
        self.is_robot_connected = False
        
        # 总关节数量
        self.TOTAL_JOINTS_COUNT = 6  # 总关节数量（5个机械臂关节+1个夹爪）
        self.current_joint_positions = np.zeros(self.TOTAL_JOINTS_COUNT)
        
        # 创建发布器和订阅器
        self.simple_command_publisher = self.create_publisher(
            Float64MultiArray, 'so100_position_commands', 10)
        
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'so100_joint_states', self._joint_state_callback, 10)
        
        # 创建机械臂和夹爪实例
        self.arm = Arm(self)
        self.gripper = Gripper(self)
        
        # 等待机器人就绪
        self._wait_for_robot(10)
    
    def _joint_state_callback(self, msg):
        """关节状态回调函数"""
        self.current_joint_positions = msg.position.tolist()
        if not self.is_robot_connected:
            self.is_robot_connected = True
            # self.get_logger().info(f'机器人已连接，当前位置: {format_to_2dp(self.current_joint_positions)}')
    
    def _wait_for_robot(self, timeout: float = 10.0) -> bool:
        """等待机器人连接就绪"""
        start_time = time.time()
        self.get_logger().info('等待机器人连接...')
        
        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.is_robot_connected:
                self.get_logger().info('机器人已准备就绪')
                return True
            time.sleep(0.1)
        
        self.get_logger().warn('等待机器人连接超时')
        return False
    
    def is_connected(self) -> bool:
        """检查机器人是否就绪"""
        return self.is_robot_connected
    
    def get_joints(self) -> Optional[np.ndarray]:
        """获取所有关节位置"""
        if not self.is_robot_connected:
            self.get_logger().warn("机器人未连接")
            return None
        
        rclpy.spin_once(self, timeout_sec=0.1)
        return self.current_joint_positions
    
    def set_joints(self, joint_positions, wait=True, timeout=10.0, tolerance=0.01) -> bool:
        if not self.is_robot_connected:
            self.get_logger().warn("机器人未连接")
            return False
        
        # 确保关节数量正确
        if len(joint_positions) != self.TOTAL_JOINTS_COUNT:
            self.get_logger().error(f"关节数量错误: 需要{self.TOTAL_JOINTS_COUNT}个关节位置")
            return False
        
        # 发送命令
        msg = Float64MultiArray()
        msg.data = np.array(joint_positions).tolist()
        self.simple_command_publisher.publish(msg)
        self.get_logger().info(f'发送移动命令: {format_to_2dp(joint_positions)}')
        
        if not wait:
            return True
        
        # 等待运动完成
        start_time = time.time()
        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if np.all(np.abs(self.current_joint_positions - joint_positions) < tolerance):
                return True
            time.sleep(0.1)
        
        self.get_logger().warn("到达目标位置超时")
        return False
    
    def close(self):
        """关闭节点"""
        self.destroy_node()
        

def main():
    """示例用法"""
    so100_robot = So100Robot()
    
    try:
        while True:
            # 获取机械臂关节角度
            arm_angles = so100_robot.arm.get_joints()
            # if arm_angles is not None:
            #     print(f"机械臂关节角度: {format_to_2dp(arm_angles)}")
            
            # # 获取夹爪位置
            # gripper_pos = so100_robot.gripper.get_joint()
            # if gripper_pos is not None:
            #     print(f"夹爪位置: {format_to_2dp(gripper_pos)}")
                
            # 获取末端执行器位姿
            pose = so100_robot.arm.get_tcp_pose()
            if pose is not None:
                print(f"末端执行器位置: {format_to_2dp(pose[0:3])}")
                # print(f"末端执行器方向: {format_to_2dp(pose[3:])}")
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("用户中断，程序结束")
    finally:
        so100_robot.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
