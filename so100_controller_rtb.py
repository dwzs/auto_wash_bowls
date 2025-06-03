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
from roboticstoolbox import ERobot
import os
from spatialmath import SE3


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
        
        self.offset_tcp_2_end = [0, -0.1, 0] # Translation [x, y, z] from flange to TCP, in flange frame
        # self.offset_tcp_2_end = [0, -0.0, 0] # Translation [x, y, z] from flange to TCP, in flange frame
        self.tcp_transform = SE3(self.offset_tcp_2_end[0], self.offset_tcp_2_end[1], self.offset_tcp_2_end[2])

        # 加载机器人模型用于逆解
        self._load_robot_model()
        
        # self.joints_data = [
        #     {"translation": [0, 0, 0],            "orientation": [0, 0, 0], "rotate_axis": [0, 0, 1], "joint_limit": [-0, 0]}, # 基座虚拟关节
        #     {"translation": [0, -0.0452, 0.0165], "orientation": [1.5708, 0, 0], "rotate_axis": [0, 1, 0], "joint_limit": [-2, 2]},
        #     {"translation": [0, 0.1025, 0.0306],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0], "joint_limit": [-1.8, 1.8]},
        #     {"translation": [0, 0.11257, 0.028],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0], "joint_limit": [-1.8, 1.57]},
        #     {"translation": [0, 0.0052, 0.1349],  "orientation": [0, 0, 0], "rotate_axis": [1, 0, 0], "joint_limit": [-3.6, 0.3] },
        #     {"translation": [0, -0.0601, 0],      "orientation": [0, 0, 0], "rotate_axis": [0, 1, 0], "joint_limit": [-1.57, 1.57]},
        #     {"translation": [0, -0.1, 0],         "orientation": [0, 0, 0], "rotate_axis": [0, 0, 1], "joint_limit": [0, 0]} # tcp 虚拟关节
        # ]
        self.joints_limits = [
            [-2, 2],
            [-1.8, 1.8],
            [-1.8, 1.57],
            [-3.6, 0.3],
            [-1.57, 1.57]
        ]

    def _load_robot_model(self):
        """加载机器人URDF模型用于逆解计算"""
        try:
            # 获取当前脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            urdf_file = os.path.join(script_dir, 'resources/SO_5DOF_ARM100_8j_URDF.SLDASM.urdf')
            
            if os.path.exists(urdf_file):
                self.robot_model = ERobot.URDF(urdf_file)
                print("robot_model: ", self.robot_model)
                # Set the TCP transform for the robot model
                if self.robot_model is not None:
                    self.robot_model.tool = self.tcp_transform
                    self._robot.get_logger().info(f"机器人URDF模型加载成功, TCP设置为: {self.tcp_transform.t}")
                    # print("robot_model with tool: ", self.robot_model.tool)
            else:
                self.robot_model = None
                self._robot.get_logger().warn(f"URDF文件不存在: {urdf_file}")
        except Exception as e:
            self.robot_model = None
            self._robot.get_logger().error(f"加载URDF模型失败: {e}")

    def _create_pose_matrix(self, pose_components):
        """
        将位姿列表转换为4x4齐次变换矩阵
        
        参数:
            pose_components (list): 位姿列表 [x, y, z, qx, qy, qz, qw]
            
        返回:
            np.ndarray: 4x4齐次变换矩阵
        """
        position = pose_components[:3]
        quaternion = pose_components[3:]  # [qx, qy, qz, qw]

        # 将四元数转换为旋转矩阵
        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

        # 创建齐次变换矩阵
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = position
        return pose_matrix

    def _forward_kinematics(self, joint_angles, use_tool=True):
        """
        使用roboticstoolbox计算正运动学 (默认计算TCP位姿)
        
        Parameters:
            joint_angles (list): A list of joint angles [j1, j2, j3, j4, j5].
            use_tool (bool): If True, use the configured TCP. If False, get flange pose.
            
        Returns:
            list: The position and orientation (as a quaternion) [x, y, z, qx, qy, qz, qw].
        """
        if self.robot_model is None:
            self._robot.get_logger().error("机器人模型未加载，无法进行正运动学计算")
            return None
            
        try:
            tool_to_use = self.robot_model.tool if use_tool else SE3() # Use identity for flange

            fk_pose_matrix = self.robot_model.fkine(np.array(joint_angles), tool=tool_to_use)
            
            # 提取位置
            position = fk_pose_matrix.t
            
            # 提取旋转并转换为四元数 [qx, qy, qz, qw]
            # scipy.spatial.transform.Rotation.as_quat() returns [x, y, z, w]
            quaternion_xyzw = Rotation.from_matrix(fk_pose_matrix.R).as_quat()
            
            # 组合位置和四元数
            pose_list = list(position) + list(quaternion_xyzw)
            return pose_list
            
        except Exception as e:
            self._robot.get_logger().error(f"roboticstoolbox正运动学计算出错: {e}")
            return None

    def _inverse_kinematics(self, target_pose_list, initial_joint_guess=None, mask=None):
        """
        计算逆运动学解 (目标位姿为TCP位姿)
        
        参数:
            target_pose_list (list): 目标位姿 [x, y, z, qx, qy, qz, qw]
            initial_joint_guess (list, optional): 关节角度初始猜测值
            mask (list, optional): 位姿约束掩码 [x, y, z, rx, ry, rz]，1表示约束该自由度
            
        返回:
            list: 关节角度列表，如果求解失败返回None
        """
        try:
            target_pose_matrix = self._create_pose_matrix(target_pose_list) 
            
            # 设置初始猜测值
            if initial_joint_guess is None:
                # 使用当前关节角度作为初始猜测
                current_joints = self.get_joints()
                if current_joints is None: # Check if get_joints failed
                    self._robot.get_logger().error("无法获取当前关节角度作为IK初始猜测值")
                    return None
                q_guess = np.array(current_joints)
            else:
                q_guess = np.array(initial_joint_guess)
            
            # 设置默认掩码（参考ik_roboticstoolbox.py）
            if mask is None:
                mask = [1, 1, 1, 0, 1, 1]  # 约束位置和部分旋转
            
            # ikine_LM 会隐式地使用 self.robot_model.tool。
            # 它求解的关节角度 q 会满足：
            #   robot.fkine(q, tool=None) * robot.tool == target_pose_matrix
            solution = self.robot_model.ikine_LM(
                target_pose_matrix, 
                q0=q_guess, 
                tool=self.robot_model.tool,
                mask=mask, 
                tol = 0.001,
                joint_limits=True,
                ilimit=1000,  # Increased iteration limit per search
                slimit=150    # Increased search limit (restarts)
            )
            
            if solution.success:
                self._robot.get_logger().info(f"逆解成功: {format_to_2dp(solution.q.tolist())}")
                return solution.q.tolist()
            else:
                self._robot.get_logger().warn(f"逆解失败，目标位姿: {target_pose_list}")
                iteration = solution.iterations # 如果需要，可以用于调试
                search_count = solution.searches
                residual = solution.residual
                reason = solution.reason
                self._robot.get_logger().warn(f"逆解失败，迭代次数: {iteration}, 搜索次数: {search_count}, 残差: {residual}, 原因: {reason}")
                return None
                
        except Exception as e:
            self._robot.get_logger().error(f"逆解计算出错: {e}")
            return None

    def move_to_pose(self, target_pose_list, initial_joint_guess=None, mask=None, wait=True, timeout=10.0, tolerance=0.01):
        """
        移动到指定位姿
        
        参数:
            target_pose_list (list): 目标位姿 [x, y, z, qx, qy, qz, qw]
            initial_joint_guess (list, optional): 关节角度初始猜测值
            mask (list, optional): 位姿约束掩码
            wait (bool): 是否等待运动完成
            timeout (float): 超时时间
            tolerance (float): 到达精度
            
        返回:
            bool: 是否成功到达目标位姿
        """
        # 计算逆解
        joint_solution = self._inverse_kinematics(target_pose_list, initial_joint_guess, mask)
        
        if joint_solution is None:
            return False
        
        # 执行关节运动
        return self.set_joints(joint_solution, wait, timeout, tolerance)

    def get_joints(self) -> Optional[np.ndarray]:
        """获取机械臂关节角度（不包括夹爪）"""
        if not self._robot.is_connected():
            self._robot.get_logger().warn("机器人未连接")
            return None
        
        joints = self._robot.get_joints()
        if joints is None:
            return None
            
        joints_list = [joints[i] for i in self.joint_index]
        return format_to_2dp(joints_list)
    

    def get_flange_pose(self) -> Optional[List[float]]:
        """获取末端执行器法兰(flange)位姿 [x, y, z, qx, qy, qz, qw]"""
        arm_joint_angles = self.get_joints()
        flange_pose = self._forward_kinematics(arm_joint_angles, use_tool=False)
        return format_to_2dp(flange_pose)

    def get_tcp_pose(self) -> Optional[List[float]]:
        """获取TCP位姿 [x, y, z, qx, qy, qz, qw]"""
        arm_joint_angles = self.get_joints()
        tcp_pose = self._forward_kinematics(arm_joint_angles, use_tool=True)
        return format_to_2dp(tcp_pose)
    
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
    
    def move_up(self, wait=True) -> bool:
        """将机械臂移动到初始位置（所有关节角度为0）"""
        up_positions = [0, 0, -np.pi/2, -np.pi/2, 0]
        return self.set_joints(up_positions, wait=wait)

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
        

def test_joints(joints, arm):
    print(" ")
    print("--------------------------------")
    print("original joints: ", joints)
    arm.set_joints(joints)
    time.sleep(2)
    flange_pose = arm.get_flange_pose()
    print("flange pose: ", flange_pose)

    tcp_pose = arm.get_tcp_pose()
    print("tcp pose: ", tcp_pose)
    # joints_ik = arm._inverse_kinematics(tcp_pose)
    # print("inverse kinematics joints: ", joints_ik)
    print("move to pose: ", flange_pose)
    arm.move_to_pose(flange_pose)
    # arm.move_to_pose(tcp_pose)




def main():
    # """示例用法"""
    arm = So100Robot().arm
    joints0 = [0, 0, 0, 0, 0]
    joints_up = [0, 0, -np.pi/2, -np.pi/2, 0]
    

    # arm.set_joints(joints0)
    # time.sleep(2)
    # arm.set_joints(joints_up)
    # time.sleep(2)

    test_joints(joints0, arm)
    # time.sleep(3)
    # test_joints(joints_up, arm)
    # joints = so100_robot.arm.get_joints()
    # end_pose = so100_robot.arm.get_flange_pose()
    # tcp_pose = so100_robot.arm.get_tcp_pose()
    # print("joints: ", joints)
    # print("end_pose: ", end_pose)
    # print("tcp_pose: ", tcp_pose)

    # so100_robot.arm.move_up()
    # joints = so100_robot.arm.get_joints()
    # end_pose = so100_robot.arm.get_flange_pose()
    # tcp_pose = so100_robot.arm.get_tcp_pose()
    # print("joints: ", joints)
    # print("end_pose: ", end_pose)
    # print("tcp_pose: ", tcp_pose)

    # time.sleep(5)

    # so100_robot.arm.move_home()
    # joints = so100_robot.arm.get_joints()
    # end_pose = so100_robot.arm.get_flange_pose()
    # tcp_pose = so100_robot.arm.get_tcp_pose()
    # print("joints: ", joints)
    # print("end_pose: ", end_pose)
    # print("tcp_pose: ", tcp_pose)

    # pose1_components = [0.25, -0.1, 0.2, 0, 0, 0, 1] # x, y, z, qx, qy, qz, qw
    # solution = so100_robot.arm._inverse_kinematics(pose1_components)
    # # print("solution: ", solution)
    # so100_robot.arm.move_to_pose(pose1_components)
    # end_pose = so100_robot.arm.get_flange_pose()
    # tcp_pose = so100_robot.arm.get_tcp_pose()
    # print("end_pose: ", end_pose)
    # print("tcp_pose: ", tcp_pose)




if __name__ == '__main__':
    main()
