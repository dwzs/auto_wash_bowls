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
from ikpy.chain import Chain
from ikpy.link import Link, URDFLink

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
        if not self._robot.is_ready():
            self._robot.get_logger().warn("机器人未连接")
            return None
        
        joints = self._robot.get_joints()
        if joints is None:
            return None
            
        return joints[self._joint_index]
    
    def set_joint(self, position: float, wait=True, timeout=5.0, tolerance=0.01) -> bool:
        """控制夹爪位置，保持机械臂关节不变
        
        参数:
            position: 夹爪位置（0.0-1.0，0为完全打开，1为完全闭合）
            wait: 是否等待运动完成
            timeout: 等待超时时间（秒）
            tolerance: 位置容差
            
        返回:
            bool: 操作是否成功
        """
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
        """打开夹爪
        
        参数:
            wait: 是否等待运动完成
            timeout: 等待超时时间（秒）
            
        返回:
            bool: 操作是否成功
        """
        return self.set_joint(self.open_position, wait, timeout)
    
    def close(self, wait=True, timeout=5.0) -> bool:
        """关闭夹爪
        
        参数:
            wait: 是否等待运动完成
            timeout: 等待超时时间（秒）
            
        返回:
            bool: 操作是否成功
        """
        return self.set_joint(self.close_position, wait, timeout)


class Arm:
    """机械臂控制类"""
    
    def __init__(self, robot_instance):
        """初始化机械臂控制器
        
        参数:
            robot_instance: 父机器人实例，用于访问ROS接口
        """
        self._robot = robot_instance
        self.joints_count = 5  # 机械臂关节数量
        self.chain = self._build_chain_manually()
        
        # 添加关节角度限制
        self.joint_limits = [
            (-0.5, 0.5),     # 关节1 (Rotation) 范围
            (-np.pi, np.pi), # 关节2 (Pitch) 范围
            (-np.pi, np.pi), # 关节3 (Elbow) 范围
            (-np.pi, np.pi), # 关节4 (Wrist_Pitch) 范围
            (-np.pi, np.pi)  # 关节5 (Wrist_Roll) 范围
        ]
    
    def _inverse_kinematics(self, target_transform) -> Optional[np.ndarray]:
        """计算逆运动学，获取机械臂关节角度
        
        参数:
            target_transform: 目标4x4变换矩阵
            
        返回:
            np.ndarray: 5个机械臂关节角度，如果计算失败则返回None
        """
        try:
            # 提取目标位置和方向
            target_position = target_transform[:3, 3]
            target_orientation = target_transform[:3, :3]
            
            # 获取初始猜测值
            current_angles = self.get_joints()
            # 使用零向量作为备选
            if current_angles is None:
                initial_position = np.zeros(6)  # 5个机械臂关节+基座关节
            else:
                initial_position = np.insert(current_angles, 0, 0)  # 在开头添加基座关节
            
            ik_solution = self.chain.inverse_kinematics(
                target_position,
                target_orientation,
                initial_position=initial_position,
            )
            
            # 校验解的质量
            fk_result = self.chain.forward_kinematics(ik_solution)
            error = np.linalg.norm(fk_result[:3, 3] - target_position)
            
            if error > 0.01:
                self._robot.get_logger().warn(f"IK解精度不足: 误差={format_to_2dp(error)}米")
                return None
                
            return ik_solution[1:6]  # 只返回机械臂的5个关节角度
                
        except Exception as e:
            self._robot.get_logger().error(f"IK计算错误: {str(e)}")
            return None
    
    def _build_chain_manually(self) -> Chain:
        """手动构建SO-100机器人的运动链
        
        当无法从URDF文件加载模型时，使用此方法手动构建运动链。
        基于SO_5DOF_ARM100_8j_URDF.SLDASM.urdf文件中的参数。
        注意：此链仅包含机械臂5个关节，不包括夹爪关节。
        
        返回:
            Chain: 构建好的ikpy运动链
        """
        # 创建链，首先添加一个固定的基座链接
        chain = Chain(name="so100_arm", links=[
            # 基座链接 (固定)
            URDFLink(
                name="base_link",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],  # Z轴，但这个链接是固定的
                bounds=(-0.01, 0.01)  # 几乎为零的范围表示固定链接
            ),
            
            # Rotation关节 - 连接base_link到Rotation_Pitch
            URDFLink(
                name="Rotation",
                origin_translation=[0, -0.0452, 0.0165],
                origin_orientation=[1.5708, 0, 0],  # X轴旋转90度
                rotation=[0, 1, 0],  # Y轴旋转
                # bounds=(-np.pi, np.pi)
                bounds=(-0.5, 0.5)
            ),
            
            # Pitch关节 - 连接Rotation_Pitch到Upper_Arm
            URDFLink(
                name="Pitch",
                origin_translation=[0, 0.1025, 0.0306],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],  # X轴旋转
                bounds=(-np.pi, np.pi)
            ),
            
            # Elbow关节 - 连接Upper_Arm到Lower_Arm
            URDFLink(
                name="Elbow",
                origin_translation=[0, 0.11257, 0.028],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],  # X轴旋转
                bounds=(-np.pi, np.pi)
            ),
            
            # Wrist_Pitch关节 - 连接Lower_Arm到Wrist_Pitch_Roll
            URDFLink(
                name="Wrist_Pitch",
                origin_translation=[0, 0.0052, 0.1349],
                origin_orientation=[0, 0, 0],
                rotation=[1, 0, 0],  # X轴旋转
                bounds=(-np.pi, np.pi)
            ),
            
            # Wrist_Roll关节 - 连接Wrist_Pitch_Roll到Fixed_Jaw
            URDFLink(
                name="Wrist_Roll",
                origin_translation=[0, -0.0601, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],  # Y轴旋转
                bounds=(-np.pi, np.pi)
            ),
        ])
        
        return chain
    
    def _execute_pose_trajectory(self, pose_waypoints, wait=True, timeout=10.0, tolerance=0.01) -> bool:
        """执行一系列位姿路径点
        
        参数:
            pose_waypoints: 路径点列表，每个路径点为(position, orientation)元组
            wait: 是否等待每个路径点移动完成
            timeout: 总等待超时时间（秒）
            tolerance: 位置容差（米）
            
        返回:
            bool: 操作是否成功
        """
        if not pose_waypoints:
            return True
        
        # 平均分配超时时间
        point_timeout = timeout / len(pose_waypoints) if wait else timeout
        
        for i, (pos, orient) in enumerate(pose_waypoints):
            success = self.move_to_pose(pos, orient, wait, point_timeout, tolerance)
            if not success:
                self._robot.get_logger().error(f"执行位姿路径点 {i+1} 失败")
                return False
            
            # 如果不等待完成整个轨迹，只执行第一个点后返回
            if not wait and i == 0:
                return True
        
        return True

    def get_joints(self) -> Optional[np.ndarray]:
        """获取机械臂关节角度（不包括夹爪）"""
        if not self._robot.is_ready():
            self._robot.get_logger().warn("机器人未连接")
            return None
        
        joints = self._robot.get_joints()
        if joints is None:
            return None
            
        return joints[:self.joints_count]
    
    def get_end_pose(self) -> Optional[Dict[str, np.ndarray]]:
        """获取末端执行器位姿"""
        if not self._robot.is_ready():
            self._robot.get_logger().warn("机器人未连接")
            return None
            
        # 计算正向运动学 (只使用机械臂关节)
        arm_joint_angles = self.get_joints()
        if arm_joint_angles is None:
            return None
            
        full_joint_angles = np.insert(arm_joint_angles, 0, 0)
        transform_matrix = self.chain.forward_kinematics(full_joint_angles)
        
        # 提取位置和姿态
        position = transform_matrix[:3, 3]
        orientation = Rotation.from_matrix(transform_matrix[:3, :3]).as_quat()
        
        return {
            'position': position,
            'orientation': orientation
        }
    
    def set_joints(self, positions: np.ndarray, wait=True, timeout=10.0, tolerance=0.01) -> bool:
        """移动机械臂关节到指定位置
        
        参数:
            positions: 机械臂关节位置 [j1, j2, j3, j4, j5]
            wait: 是否等待运动完成
            timeout: 等待超时时间（秒）
            tolerance: 位置容差（弧度）
            
        返回:
            bool: 操作是否成功
        """
        if not self._robot.is_ready():
            self._robot.get_logger().warn("机器人未连接")
            return False
        
        # 确保关节数量正确
        if len(positions) != self.joints_count:
            self._robot.get_logger().error(f"机械臂关节数量错误: 需要{self.joints_count}个关节位置")
            return False
        
        # 检查关节角度是否在限制范围内
        for i, (pos, (lower, upper)) in enumerate(zip(positions, self.joint_limits)):
            if pos < lower or pos > upper:
                self._robot.get_logger().error(
                    f"关节 {i+1} 超出限制范围: 位置 {format_to_2dp(pos)}, 范围 [{format_to_2dp(lower)}, {format_to_2dp(upper)}]")
                return False
        
        # 获取当前夹爪位置
        current_gripper_pos = self._robot.gripper.get_joint()
        if current_gripper_pos is None:
            return False
        
        # 构建完整的关节位置数组（包括夹爪）
        full_joint_positions = np.append(positions, current_gripper_pos)
        
        # 发送完整的关节命令
        return self._robot.set_joints(full_joint_positions, wait, timeout, tolerance)
    
    def move_to_position(self, position, wait=True, timeout=10.0, tolerance=0.01) -> bool:
        """移动机械臂末端到指定的笛卡尔空间位置
        
        参数:
            position: 目标位置 [x, y, z]，单位为米
            wait: 是否等待运动完成
            timeout: 等待超时时间（秒）
            tolerance: 位置容差（米）
            
        返回:
            bool: 操作是否成功
        """
        if not self._robot.is_ready():
            self._robot.get_logger().warn("机器人未连接")
            return False
            
        # 获取当前姿态，保持方向不变，只改变位置
        current_pose = self.get_end_pose()
        if current_pose is None:
            self._robot.get_logger().error("无法获取当前位姿")
            return False
            
        # 使用当前的方向
        orientation = current_pose['orientation']
        
        # 调用move_to_pose进行运动
        return self.move_to_pose(position, orientation, wait, timeout, tolerance)
    
    def move_to_pose(self, position, orientation, wait=True, timeout=10.0, tolerance=0.01) -> bool:
        """移动机械臂末端到指定的笛卡尔空间位姿
        
        参数:
            position: 目标位置 [x, y, z]，单位为米
            orientation: 目标方向，四元数 [x, y, z, w]
            wait: 是否等待运动完成
            timeout: 等待超时时间（秒）
            tolerance: 位置容差（米）
            
        返回:
            bool: 操作是否成功
        """
        if not self._robot.is_ready():
            self._robot.get_logger().warn("机器人未连接")
            return False
            
        # 创建目标变换矩阵
        target_transform = np.eye(4)
        target_transform[:3, :3] = Rotation.from_quat(orientation).as_matrix()
        target_transform[:3, 3] = position
        
        # 逆运动学计算关节角度
        arm_joint_angles = self._inverse_kinematics(target_transform)
        
        if arm_joint_angles is None:
            self._robot.get_logger().error("无法计算逆运动学解")
            return False
            
        # 使用move_joints移动机械臂
        return self.set_joints(arm_joint_angles, wait, timeout, tolerance)
    

    def move_home(self, wait=True) -> bool:
        """将机械臂移动到初始位置（所有关节角度为0）"""
        home_positions = np.zeros(self.joints_count)
        return self.set_joints(home_positions, wait=wait)
    

    def follow_path(self, waypoints: List[Union[List[float], np.ndarray]],
                    wait=True, timeout=None, tolerance=0.01) -> bool:
        """使机械臂沿笛卡尔空间路径点移动
        
        参数:
            waypoints: 笛卡尔空间位置路径点列表，每个点为 [x, y, z]，单位为米
            wait: 是否等待每个路径点移动完成
            timeout: 每个路径点的等待超时时间（秒），若为None则使用默认值(10.0)
            tolerance: 位置容差（米）
            
        返回:
            bool: 操作是否成功
        """
        if not self._robot.is_ready():
            self._robot.get_logger().warn("机器人未连接")
            return False
        
        if not waypoints:
            self._robot.get_logger().warn("路径点列表为空")
            return False
        
        # 设置默认超时时间
        if timeout is None:
            timeout = 10.0
        
        # 按顺序执行每个路径点
        for i, point in enumerate(waypoints):
            self._robot.get_logger().info(f"执行笛卡尔路径点 {i+1}/{len(waypoints)}")
            
            # 笛卡尔空间点 - 只改变位置，保持方向不变
            success = self.move_to_position(point, wait, timeout, tolerance)
            
            if not success:
                self._robot.get_logger().error(f"执行笛卡尔路径点 {i+1} 失败")
                return False
            
            # 如果不等待完成整个轨迹，只执行第一个点后返回
            if not wait and i == 0:
                return True
        
        return True
    
    def follow_joints(self, waypoints: List[Union[List[float], np.ndarray]],
                      wait=True, timeout=None, tolerance=0.01) -> bool:
        """使机械臂沿关节空间路径点移动
        
        参数:
            waypoints: 关节空间路径点列表，每个点为 [j1, j2, j3, j4, j5]
            wait: 是否等待每个路径点移动完成
            timeout: 每个路径点的等待超时时间（秒），若为None则使用默认值(10.0)
            tolerance: 关节角度容差（弧度）
            
        返回:
            bool: 操作是否成功
        """
        if not self._robot.is_ready():
            self._robot.get_logger().warn("机器人未连接")
            return False
        
        if not waypoints:
            self._robot.get_logger().warn("路径点列表为空")
            return False
        
        # 设置默认超时时间
        if timeout is None:
            timeout = 10.0
        
        # 按顺序执行每个路径点
        for i, point in enumerate(waypoints):
            self._robot.get_logger().info(f"执行关节路径点 {i+1}/{len(waypoints)}")
            
            # 关节空间点
            success = self.set_joints(point, wait, timeout, tolerance)
            
            if not success:
                self._robot.get_logger().error(f"执行关节路径点 {i+1} 失败")
                return False
            
            # 如果不等待完成整个轨迹，只执行第一个点后返回
            if not wait and i == 0:
                return True
        
        return True
    
    def move_linear_to_position(self, position, segments=10, wait=True, timeout=10.0, tolerance=0.01) -> bool:
        """以线性路径移动机械臂末端到指定位置
        
        参数:
            position: 目标位置 [x, y, z]，单位为米
            segments: 将路径分成的线段数量，越多越接近直线
            wait: 是否等待运动完成
            timeout: 每个路径点的等待超时时间（秒）
            tolerance: 位置容差（米）
            
        返回:
            bool: 操作是否成功
        """
        if not self._robot.is_ready():
            self._robot.get_logger().warn("机器人未连接")
            return False
        
        # 获取当前位姿
        current_pose = self.get_end_pose()
        if current_pose is None:
            self._robot.get_logger().error("无法获取当前位姿")
            return False
        
        start_position = current_pose['position']
        current_orientation = current_pose['orientation']
        
        # 确保输入是numpy数组
        target_position = np.array(position)
        
        # 生成中间路径点
        waypoints = []
        for i in range(1, segments + 1):
            ratio = i / segments
            # 线性插值计算中间点
            intermediate_pos = start_position + ratio * (target_position - start_position)
            waypoints.append(intermediate_pos)
        
        # 创建包含位置和方向的完整路径点
        full_waypoints = []
        for pos in waypoints:
            full_waypoints.append((pos, current_orientation))
        
        # 执行轨迹
        total_timeout = timeout * segments if wait else timeout
        
        return self._execute_pose_trajectory(full_waypoints, wait, total_timeout, tolerance)
    


class So100Robot(Node):
    """SO-100机器人控制接口主类"""
    
    def __init__(self):
        # 初始化ROS
        if not rclpy.ok():
            rclpy.init()
            
        super().__init__('So100Robot')
        self.is_robot_ready = False
        
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
        self._wait_for_robot()
    
    def _joint_state_callback(self, msg):
        """关节状态回调函数"""
        self.current_joint_positions = np.array(msg.position)
        if not self.is_robot_ready:
            self.is_robot_ready = True
            # self.get_logger().info(f'机器人已连接，当前位置: {format_to_2dp(self.current_joint_positions)}')
    
    def _wait_for_robot(self, timeout: float = 10.0) -> bool:
        """等待机器人连接就绪"""
        start_time = time.time()
        self.get_logger().info('等待机器人连接...')
        
        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.is_robot_ready:
                self.get_logger().info('机器人已准备就绪')
                return True
            time.sleep(0.1)
        
        self.get_logger().warn('等待机器人连接超时')
        return False
    
    def is_ready(self) -> bool:
        """检查机器人是否就绪"""
        return self.is_robot_ready
    
    def get_joints(self) -> Optional[np.ndarray]:
        """获取所有关节位置"""
        if not self.is_robot_ready:
            self.get_logger().warn("机器人未连接")
            return None
        
        rclpy.spin_once(self, timeout_sec=0.1)
        return self.current_joint_positions
    
    def set_joints(self, joint_positions, wait=True, timeout=10.0, tolerance=0.01) -> bool:
        """移动所有关节到指定位置
        
        参数:
            joint_positions: 关节位置 [j1, j2, j3, j4, j5, j6]（包括夹爪）
            wait: 是否等待运动完成
            timeout: 等待超时时间（秒）
            tolerance: 位置容差（弧度）
            
        返回:
            bool: 操作是否成功
        """
        if not self.is_robot_ready:
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
            if arm_angles is not None:
                print(f"机械臂关节角度: {format_to_2dp(arm_angles)}")
            
            # 获取夹爪位置
            gripper_pos = so100_robot.gripper.get_joint()
            if gripper_pos is not None:
                print(f"夹爪位置: {format_to_2dp(gripper_pos)}")
                
            # 获取末端执行器位姿
            pose = so100_robot.arm.get_end_pose()
            if pose is not None:
                print(f"末端执行器位置: {format_to_2dp(pose['position'])}")
                # print(f"末端执行器方向: {format_to_2dp(pose['orientation'])}")
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("用户中断，程序结束")
    finally:
        so100_robot.close()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
