#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-100机器人控制接口"""

import json
import rclpy
import numpy as np
import time
import logging
from typing import List, Union, Optional, Dict, Any, Tuple
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation
from roboticstoolbox import ERobot
import os
from spatialmath import SE3
import colorlog


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
        
        # 配置日志
        self.logger = logging.getLogger('so100_gripper')
    
    def get_joint(self) -> Optional[float]:
        """获取夹爪位置"""
        if not self._robot.is_connected():
            self.logger.warning("机器人未连接")
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
        self.logger.info("打开夹爪")
        return self.set_joint(self.open_position, wait, timeout)
    
    def close(self, wait=True, timeout=5.0) -> bool:
        self.logger.info("关闭夹爪")
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
        
        # 配置日志
        self.logger = logging.getLogger('so100_arm')

        # 加载机器人模型用于逆解
        self._load_robot_model()
        # self._offset_calibration_data()
        self.line_joints = []

    # def _offset_calibration_data(self):
    #     """根据标定数据计算零位偏移"""

    #     self.offest_zero = self._robot.zero_pose_joints[:5]  
    #     self.zero_joints = [0, 0, 0, 0, 0]
    #     self.home_joints = [self._robot.home_pose_joints[i] - self.offest_zero[i] for i in range(5)]
    #     self.up_joints = [self._robot.up_pose_joints[i] - self.offest_zero[i] for i in range(5)]
        
    #     # 计算偏移后的关节限制
    #     raw_limits = self._robot.joints_limits
    #     self.joints_limits = []
    #     for i in range(5):
    #         joint_key = str(i+1)
    #         if joint_key in raw_limits:
    #             joint_limit_dict = raw_limits[joint_key]
    #             # 从嵌套字典中提取min和max值
    #             try:
    #                 raw_min = float(joint_limit_dict["min"])
    #                 raw_max = float(joint_limit_dict["max"])
    #             except (ValueError, TypeError, KeyError) as e:
    #                 self.logger.error(f"关节{i+1}的限制值无法解析: {joint_limit_dict}")
    #                 self.joints_limits.append(None)
    #                 continue
                
    #             # 应用零位偏移：逻辑限制 = 原始限制 - 零位偏移
    #             offset_min = raw_min - self.offest_zero[i]
    #             offset_max = raw_max - self.offest_zero[i]
    #             self.joints_limits.append([offset_min, offset_max])
    #         else:
    #             self.logger.warning(f"关节{i+1}的限制未找到")
    #             self.joints_limits.append(None)

    #     print("self.zero_joints: ", self.zero_joints)
    #     print("self.home_joints: ", self.home_joints)
    #     print("self.up_joints: ", self.up_joints)
    #     print("self.joints_limits: ", self.joints_limits)

    def _load_robot_model(self):
        """加载机器人URDF模型用于逆解计算"""
        try:
            # 获取当前脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            urdf_file = os.path.join(script_dir, 'resources/so100_tcp.urdf')
            
            if os.path.exists(urdf_file):
                self.robot_model = ERobot.URDF(urdf_file)
                self.logger.info(f"成功加载URDF模型: {urdf_file}")
                # print("self.robot_model: ", self.robot_model)
            else:
                self.robot_model = None
                self.logger.warning(f"URDF文件不存在: {urdf_file}")
        except Exception as e:
            self.robot_model = None
            self.logger.error(f"加载URDF模型失败: {e}")

        self.zero_joints = self._robot.zero_pose_joints[:5]
        self.home_joints = self._robot.home_pose_joints[:5]
        self.up_joints = self._robot.up_pose_joints[:5]
        
        limits = self._robot.joints_limits
        # print("limits: ", limits)
        self.joints_limits = [limits[i] for i in self.joint_index]

        # print("self.zero_joints: ", self.zero_joints)
        # print("self.home_pose: ", self.home_pose)
        # print("self.up_pose: ", self.up_pose)
        # print("self.joints_limits: ", self.joints_limits)
        self.line_joints = []

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

    def _forward_kinematics(self, joint_angles, use_tool=False):
        """
        使用roboticstoolbox计算正运动学 (默认计算TCP位姿)
        
        Parameters:
            joint_angles (list): A list of joint angles [j1, j2, j3, j4, j5].
            use_tool (bool): If True, use the configured TCP. If False, get flange pose.
            
        Returns:
            list: The position and orientation (as a quaternion) [x, y, z, qx, qy, qz, qw].
        """
        if use_tool:
            end_link = "tcp"
        else:
            end_link = "Fixed_Jaw"

        try:
            # tool_to_use = self.robot_model.tool if use_tool else SE3() # Use identity for flange

            fk_pose_matrix = self.robot_model.fkine(np.array(joint_angles), end=end_link)
            
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

    def _inverse_kinematics(self, target_pose_list, initial_joint_guess=None, mask=None, use_tool=False):
        """
        计算逆运动学解 (目标位姿为TCP位姿)
        
        参数:
            target_pose_list (list): 目标位姿 [x, y, z, qx, qy, qz, qw]
            initial_joint_guess (list, optional): 关节角度初始猜测值
            mask (list, optional): 位姿约束掩码 [x, y, z, rx, ry, rz]，1表示约束该自由度
            
        返回:
            list: 关节角度列表，如果求解失败返回None
        """
        if use_tool:
            end_link = "tcp"
        else:
            end_link = "Fixed_Jaw"

        try:
            target_pose_matrix = self._create_pose_matrix(target_pose_list) 
            
            # 设置初始猜测值
            if initial_joint_guess is None:
                # 使用当前关节角度作为初始猜测
                current_joints = self.get_joints()
                if current_joints is None: # Check if get_joints failed
                    self.logger.error("无法获取当前关节角度作为IK初始猜测值")
                    return None
                q_guess = np.array(current_joints)
            else:
                q_guess = np.array(initial_joint_guess)
            
            # 设置默认掩码（参考ik_roboticstoolbox.py）
            if mask is None:
                mask = [1, 1, 1, 0, 1, 1]  # 约束位置和部分旋转
            
            # 使用 ik_LM 函数，返回值是元组格式
            result = self.robot_model.ik_LM(
                target_pose_matrix, 
                q0=q_guess, 
                mask=mask, 
                end=end_link,
                tol = 0.001,
                joint_limits=True,
                ilimit=1000,  # Increased iteration limit per search
                slimit=150    # Increased search limit (restarts)
            )
            
            # ik_LM 返回 (q, success, iterations, searches, residual)
            if len(result) >= 2:
                q_solution, success = result[0], result[1]
                
                if success:
                    self.logger.debug(f"逆解成功: {format_to_2dp(q_solution.tolist())}")
                    return q_solution.tolist()
                else:
                    # 提取更多调试信息
                    iterations = result[2] if len(result) > 2 else "未知"
                    searches = result[3] if len(result) > 3 else "未知"
                    residual = result[4] if len(result) > 4 else "未知"
                    
                    self.logger.warning(f"逆解失败，目标位姿: {target_pose_list}")
                    self.logger.warning(f"逆解失败，迭代次数: {iterations}, 搜索次数: {searches}, 残差: {residual}")
                    return None
            else:
                self.logger.error(f"ik_LM返回值格式异常: {result}")
                return None
                
        except Exception as e:
            self.logger.error(f"逆解计算出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def flange_move_to_pose(self, target_pose_list, initial_joint_guess=None, mask=None, wait=True, timeout=10.0, tolerance=0.01):
        """
        移动法兰到目标位姿
        
        参数:
            target_pose_list: 目标位姿，可以是：
                - [x, y, z] - 只移动位置，保持当前姿态
                - [x, y, z, qx, qy, qz, qw] - 完整位姿
            initial_joint_guess: 关节角度初始猜测值
            mask: 位姿约束掩码
            wait: 是否等待运动完成
            timeout: 超时时间
            tolerance: 到达精度
            
        返回:
            bool: 是否成功
        """
        # 处理输入参数，确保包含完整的位姿信息
        if len(target_pose_list) == 3:
            # 如果只有位置信息，获取当前姿态
            current_flange = self.get_flange_pose()
            if current_flange is None:
                self.logger.error("无法获取当前法兰姿态")
                return False
            
            # 组合新位置和当前姿态
            full_target_pose = list(target_pose_list) + current_flange[3:]  # [x,y,z] + [qx,qy,qz,qw]
            self.logger.info(f"移动法兰到位置: {format_to_2dp(target_pose_list)}, 保持当前姿态")
            
        elif len(target_pose_list) == 7:
            # 完整位姿信息
            full_target_pose = list(target_pose_list)
            self.logger.info(f"移动法兰到完整位姿: 位置{format_to_2dp(target_pose_list[:3])}, 姿态{format_to_2dp(target_pose_list[3:])}")
            
        else:
            self.logger.error(f"目标位姿参数长度错误: {len(target_pose_list)}, 应为3或7")
            return False
        
        # 计算逆解
        joint_solution = self._inverse_kinematics(full_target_pose, initial_joint_guess, mask, use_tool=False)
        
        if joint_solution is None:
            return False
        
        # 执行关节运动
        return self.set_joints(joint_solution, wait, timeout, tolerance)

    def tcp_move_to_pose(self, target_pose_list, initial_joint_guess=None, mask=None, wait=True, timeout=20.0, tolerance=0.01):
        """
        移动TCP到目标位姿
        
        参数:
            target_pose_list: 目标位姿，可以是：
                - [x, y, z] - 只移动位置，保持当前姿态
                - [x, y, z, qx, qy, qz, qw] - 完整位姿
            initial_joint_guess: 关节角度初始猜测值
            mask: 位姿约束掩码
            wait: 是否等待运动完成
            timeout: 超时时间
            tolerance: 到达精度
            
        返回:
            bool: 是否成功
        """
        # 处理输入参数，确保包含完整的位姿信息
        if len(target_pose_list) == 3:
            # 如果只有位置信息，获取当前姿态
            current_tcp = self.get_tcp_pose()
            if current_tcp is None:
                self.logger.error("无法获取当前TCP姿态")
                return False
            
            # 组合新位置和当前姿态
            full_target_pose = list(target_pose_list) + current_tcp[3:]  # [x,y,z] + [qx,qy,qz,qw]
            self.logger.info(f"移动到位置: {format_to_2dp(target_pose_list)}, 保持当前姿态")
            
        elif len(target_pose_list) == 7:
            # 完整位姿信息
            full_target_pose = list(target_pose_list)
            self.logger.debug(f"移动到完整位姿: 位置{format_to_2dp(target_pose_list[:3])}, 姿态{format_to_2dp(target_pose_list[3:])}")
            
        else:
            self.logger.error(f"目标位姿参数长度错误: {len(target_pose_list)}, 应为3或7")
            return False
        
        # 计算逆解
        start_time = time.time()
        initial_joint_guess = self.get_joints()
        end_time = time.time()
        self.logger.debug(f"获取关节角度耗时: {end_time - start_time:.4f} 秒")

        start_time = time.time()
        joint_solution = self._inverse_kinematics(full_target_pose, initial_joint_guess, mask, use_tool=True)
        end_time = time.time()
        self.logger.debug(f"逆运动学计算耗时: {end_time - start_time:.4f} 秒")

        if joint_solution is None:
            return False
        
        # 执行关节运动
        self.line_joints.append(joint_solution)
        return self.set_joints(joint_solution, wait, timeout, tolerance)

    def get_joints(self) -> Optional[List[float]]:
        """获取机械臂关节角度（不包括夹爪），应用零位偏移"""
        joints = self._robot.get_joints()
        # self.logger.debug(f"获取到的所有关节: {joints}")
        if joints is None:
            return None
        
        # # 获取机械臂关节原始值
        # raw_joints = [joints[i] for i in self.joint_index]
        
        # # 应用零位偏移
        # joints_offseted = [raw - zero for raw, zero in zip(raw_joints, self.offest_zero)]
        
        # # print(f"原始机械臂关节: {raw_joints}")
        # # print(f"零位偏移: {self.zero_joints}")
        # # print(f"偏移后关节: {offset_joints}")
        
        # return format_to_2dp(joints_offseted)
        return format_to_2dp(joints)
    

    def get_flange_pose(self) -> Optional[List[float]]:
        """获取末端执行器法兰(flange)位姿 [x, y, z, qx, qy, qz, qw]"""
        arm_joint_angles = self.get_joints()
        print("arm_joint_angles: ", arm_joint_angles)
        flange_pose = self._forward_kinematics(arm_joint_angles, use_tool=False)
        return format_to_2dp(flange_pose)

    def get_tcp_pose(self) -> Optional[List[float]]:
        """获取TCP位姿 [x, y, z, qx, qy, qz, qw]"""
        arm_joint_angles = self.get_joints()
        print("arm_joint_angles: ", arm_joint_angles)
        tcp_pose = self._forward_kinematics(arm_joint_angles, use_tool=True)
        return format_to_2dp(tcp_pose)
    
    def set_joints(self, joints: List[float], wait=True, timeout=10.0, tolerance=0.1) -> bool:
        # 确保关节数量正确
        if len(joints) != len(self.joint_index):
            self.logger.error(f"机械臂关节数量错误: 需要{len(self.joint_index)}个关节位置, 实际关节数量: {len(joints)}")
            return False
        
        # # 应用零位偏移，将逻辑关节角度转换为原始关节角度
        # raw_positions = [pos + zero for pos, zero in zip(positions, self.offest_zero)]
        
        # # print(f"输入的逻辑关节角度: {format_to_2dp(positions)}")
        # # print(f"转换后的原始关节角度: {format_to_2dp(raw_positions)}")
        
        # 获取当前夹爪位置
        current_gripper_pos = self._robot.gripper.get_joint()
        if current_gripper_pos is None:
            return False
        
        # 构建完整的关节位置数组（包括夹爪）
        full_joint_positions = joints + [current_gripper_pos]
        
        self.logger.debug(f"发送给机器人的完整关节位置: {format_to_2dp(full_joint_positions)}")
        
        # 发送完整的关节命令
        # print("full_joint_positions: ", full_joint_positions)
        return self._robot.set_joints(full_joint_positions, wait, timeout, tolerance)

    def move_home(self, wait=True) -> bool:
        """将机械臂移动到初始位置（所有关节角度为0）"""
        self.logger.info("移动到初始位置")
        return self.set_joints(self.home_joints, wait=wait)
    
    def move_up(self, wait=True) -> bool:
        """将机械臂移动到初始位置（所有关节角度为0）"""
        self.logger.info("移动到抬起位置")
        return self.set_joints(self.up_joints, wait=wait)
    
    def move_zero(self, wait=True) -> bool:
        """将机械臂移动到初始位置（所有关节角度为0）"""
        self.logger.info("移动到初始位置")
        return self.set_joints(self.zero_joints, wait=wait)
    
    def move_line(self, position_a, position_b, wait=True, timeout=10.0, tolerance=0.01):
        """
        控制机械臂TCP从A点直线运动到B点，保持当前姿态
        
        参数:
            position_a: 起始位置 [x, y, z]
            position_b: 终止位置 [x, y, z]
            wait: 是否等待运动完成
            timeout: 超时时间
            tolerance: 到达精度
            
        返回:
            bool: 是否成功完成直线运动
        """
        try:
            # 验证输入参数
            if len(position_a) != 3 or len(position_b) != 3:
                self.logger.error(f"位置参数长度错误: A点长度{len(position_a)}, B点长度{len(position_b)}, 应为3")
                return False
            
            # 获取当前TCP姿态
            current_tcp = self.get_tcp_pose()
            if current_tcp is None:
                self.logger.error("无法获取当前TCP姿态")
                return False
            
            # 使用当前姿态构建完整位姿
            current_orientation = current_tcp[3:]  # [qx, qy, qz, qw]
            pose_a = list(position_a) + current_orientation
            pose_b = list(position_b) + current_orientation
            
            self.logger.info(f"开始直线运动: A{format_to_2dp(position_a)} -> B{format_to_2dp(position_b)}, 保持当前姿态")
            
            # 计算直线距离和插值点数量
            distance = np.linalg.norm(np.array(position_b) - np.array(position_a))
            
            # 根据距离确定插值点数量 (每1cm一个点)
            num_points = max(int(distance * 100), 2)  # 至少2个点
            num_points = min(num_points, 50)  # 最多50个点
            
            self.logger.info(f"直线距离: {distance:.3f}m, 插值点数: {num_points}")
            
            # 生成直线轨迹点
            waypoints = []
            for i in range(num_points + 1):
                t = i / num_points  # 插值参数 0 到 1
                
                # 位置线性插值
                pos_interp = np.array(position_a) + t * (np.array(position_b) - np.array(position_a))
                
                # 组合位置和固定姿态
                waypoint = list(pos_interp) + current_orientation
                waypoints.append(waypoint)
            
            # 输出直线运动关节轨迹，每行一个轨迹点
            self.logger.debug("直线运动关节轨迹:")
            for i, waypoint in enumerate(waypoints):
                self.logger.debug(f"  轨迹点 {i+1}: {format_to_2dp(waypoint)}")

            # 执行轨迹运动
            for i, waypoint in enumerate(waypoints):
                self.logger.debug(f"移动到轨迹点 {i+1}/{len(waypoints)}: {format_to_2dp(waypoint[:3])}")
                
                start_time = time.time()
                if not self.tcp_move_to_pose(waypoint, wait=wait, tolerance=tolerance):
                    self.logger.error(f"移动到轨迹点 {i+1} 失败")
                    return False
                end_time = time.time()
                self.logger.debug(f"移动到轨迹点 {i+1} 耗时: {end_time - start_time:.4f} 秒")
            

            
            self.logger.info("直线运动完成")
            return True
            
        except Exception as e:
            self.logger.error(f"直线运动过程中发生错误: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False


class So100Robot(Node):
    """SO-100机器人控制接口主类"""
    
    def __init__(self):
        # 配置Python logging with colors (minimal changes)
        colorlog.basicConfig(
            # level=logging.DEBUG,
            level=logging.INFO,
            format='%(log_color)s[%(levelname)s][%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('so100_controller')
        
        # 初始化ROS
        if not rclpy.ok():
            rclpy.init()
            
        super().__init__('So100Robot')
        self.is_robot_connected = False
        self.servo_ids = [1, 2, 3, 4, 5, 6]
        
        # 总关节数量
        self.TOTAL_JOINTS_COUNT = 6  # 总关节数量（5个机械臂关节+1个夹爪）
        self.current_joint_positions = [0.0] * self.TOTAL_JOINTS_COUNT
        self.calibration_file = "so100_calibration.json"


        # # calibration_data = self._load_joints_limits()
        calibration_data = self.load_calibration_data()
        # self.joints_limits = calibration_data["joint_limits_offseted"]
        # # self.zero_pose_joints = calibration_data["poses_joints"]["zero_pose_joints"]
        # # self.home_pose_joints = calibration_data["poses_joints_offseted"]["home_pose_joints"]
        # # self.up_pose_joints = calibration_data["poses_joints_offseted"]["up_pose_joints"]
        
        # self.zero_pose_joints = calibration_data["poses_joints"]["zero_pose_joints"]
        # self.home_pose_joints = calibration_data["poses_joints"]["home_pose_joints"]
        # self.up_pose_joints = calibration_data["poses_joints"]["up_pose_joints"]

        # self.zero_pose_joints_offseted = calibration_data["poses_joints_offseted"]["zero_pose_joints"]
        # self.home_pose_joints_offseted = calibration_data["poses_joints_offseted"]["home_pose_joints"]
        # self.up_pose_joints_offseted = calibration_data["poses_joints_offseted"]["up_pose_joints"]


        # 创建发布器和订阅器
        self.pub_rate = 1000
        self.simple_command_publisher = self.create_publisher(
            JointState, 'so100_position_commands', self.pub_rate)
        
        self.joint_state_subscriber = self.create_subscription(
            JointState, 'so100_joint_states', self._joint_state_callback, 10)
        
        # 创建机械臂和夹爪实例
        self.arm = Arm(self)
        self.gripper = Gripper(self)
        
        # 等待机器人就绪
        self._wait_for_robot(10)
        
    # def _load_joints_limits(self):
    #     """从配置文件加载关节限制"""
    #     try:
    #         with open(self.calibration_file, 'r', encoding='utf-8') as f:
    #             calibration_data = json.load(f)
    #         return calibration_data
    #     except FileNotFoundError:
    #         self.logger.error(f'关节限制配置文件未找到: {self.calibration_file}')
    #         raise
    #     except json.JSONDecodeError as e:
    #         self.logger.error(f'关节限制配置文件格式错误: {e}')
    #         raise
    #     except Exception as e:
    #         self.logger.error(f'关节限制配置文件加载失败: {e}')
    #         raise

    def _check_and_limit_joint(self, servo_id: int, joint_rad: int):
        """检查关节是否超出限制"""
        if joint_rad < self.joints_limits[servo_id-1][0]:
            self.logger.warning(f'关节{servo_id}({joint_rad})超出最小限制({self.joints_limits[servo_id-1][0]})')
            joint_rad = self.joints_limits[servo_id-1][0]
        elif joint_rad > self.joints_limits[servo_id-1][1]:
            self.logger.warning(f'关节{servo_id}({joint_rad})超出最大限制({self.joints_limits[servo_id-1][1]})')
            joint_rad = self.joints_limits[servo_id-1][1]
        return joint_rad

    def load_calibration_data(self):
        """从配置文件加载标定数据"""
        try:
            with open(self.calibration_file, 'r', encoding='utf-8') as f:
                calibration_data = json.load(f)
            
            # 加载关节限制
            self.joints_limits = []
            joint_limits_dict = calibration_data.get('joint_limits_offseted', {})
            for id in self.servo_ids:  
                jointi_limit = joint_limits_dict.get(str(id))
                if jointi_limit is None:
                    self.logger.error(f'关节{id}的限制配置未找到')
                    return False
                self.joints_limits.append([jointi_limit["min"], jointi_limit["max"]])
            
            self.zero_pose_joints = calibration_data["poses_joints_offseted"]["zero_pose_joints"]
            self.home_pose_joints = calibration_data["poses_joints_offseted"]["home_pose_joints"]
            self.up_pose_joints = calibration_data["poses_joints_offseted"]["up_pose_joints"]

            # self.zero_pose_joints_offseted = calibration_data["poses_joints_offseted"]["zero_pose_joints"]
            # self.home_pose_joints_offseted = calibration_data["poses_joints_offseted"]["home_pose_joints"]
            # self.up_pose_joints_offseted = calibration_data["poses_joints_offseted"]["up_pose_joints"]

            self.logger.info(f'成功加载标定数据: {self.calibration_file}')
            self.logger.info(f'关节限制: {self.joints_limits}')
            self.logger.info(f'zero_pose关节位置: {self.zero_pose_joints}')
            self.logger.info(f'home_pose关节位置: {self.home_pose_joints}')
            self.logger.info(f'up_pose关节位置: {self.up_pose_joints}')
            # self.logger.info(f'zero_pose关节位置(offseted): {self.zero_pose_joints_offseted}')
            # self.logger.info(f'home_pose关节位置(offseted): {self.home_pose_joints_offseted}')
            # self.logger.info(f'up_pose关节位置(offseted): {self.up_pose_joints_offseted}')
            return True
        
        except FileNotFoundError:
            self.logger.error(f'标定文件未找到: {self.calibration_file}')
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f'标定文件格式错误: {e}')
            raise
        except Exception as e:
            self.logger.error(f'加载标定数据失败: {e}')
            raise



    def _joint_state_callback(self, msg):
        """关节状态回调函数"""
        self.current_joint_positions = msg.position.tolist()
        # self.logger.debug(f'关节状态: {format_to_2dp(self.current_joint_positions)}')
        if not self.is_robot_connected:
            self.is_robot_connected = True
            self.logger.info(f'机器人已连接，当前位置: {format_to_2dp(self.current_joint_positions)}')
    
    def _wait_for_robot(self, timeout: float = 10.0) -> bool:
        """等待机器人连接就绪"""
        start_time = time.time()
        self.logger.info('等待机器人连接...')
        
        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.is_robot_connected:
                self.logger.info('机器人已准备就绪')
                return True
            time.sleep(0.1)
        
        self.logger.warning('等待机器人连接超时')
        return False
    
    def is_connected(self) -> bool:
        """检查机器人是否就绪"""
        return self.is_robot_connected
    
    def get_joints(self) -> Optional[List[float]]:
        """获取所有关节位置"""
        rclpy.spin_once(self, timeout_sec=0.1)
        return self.current_joint_positions
    
    def set_joints(self, joint_positions, wait=True, timeout=10.0, tolerance=0.01) -> bool:
        if not self.is_robot_connected:
            self.logger.warning("机器人未连接")
            return False
        
        # 确保关节数量正确
        if len(joint_positions) != self.TOTAL_JOINTS_COUNT:
            self.logger.error(f"关节数量错误: 需要{self.TOTAL_JOINTS_COUNT}个关节位置")
            return False
        
        # 检查关节是否超出限制
        for i, joint_position in enumerate(joint_positions):
            joint_positions[i] = self._check_and_limit_joint(i+1, joint_position)

        # 发送命令
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        msg.position = joint_positions
        self.simple_command_publisher.publish(msg)
        self.logger.debug(f'发送移动命令: {format_to_2dp(joint_positions)}')
        
        if not wait:
            return True
        
        # 等待运动完成
        start_time = time.time()
        i = 0
        joints_diffs = []
        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.01)
            current_joint_positions = self.get_joints()
            joints_diffs = [abs(current - target) for current, target in zip(current_joint_positions, joint_positions)]
            if all(diff < tolerance for diff in joints_diffs):
                return True
            # else:
            #     self.logger.warning("到达目标位置超时")
            #     self.logger.debug(f"关节误差: {joints_diffs}")
            time.sleep(0.01)
            i += 1
        self.logger.warning("到达目标位置超时")
        self.logger.debug(f"target_joint_positions: {joint_positions}")
        self.logger.debug(f"current_joint_positions: {format_to_2dp(self.current_joint_positions)}")
        self.logger.debug(f"有关节误差超过{tolerance}的关节，当前误差: {joints_diffs}")
        
        return False
    
    def close(self):
        """关闭节点"""
        self.logger.info("关闭SO-100控制节点")
        self.destroy_node()
        
    def print_joints_loop(self):
        """打印关节位置"""
        while True:
            joints = self.get_joints()
            rclpy.spin_once(self, timeout_sec=0.01)
            self.logger.debug(f'关节状态: {format_to_2dp(joints)}')
            time.sleep(0.01)

def main():
    """示例用法"""

    ## so100 test
    init_joints = [0, 0, 0, 0, 0]

    tcp_init_pose = [0.0, -0.24, 0.08, 0.71, 0.0, 0.0, 0.71]
    tcp_init_pose1 = [0.1, -0.24, 0.28, 0.71, 0.0, 0.0, 0.71]

    tcp_position_a = [ 0.1, -0.15, 0.05]    
    tcp_position_b = [-0.1, -0.15, 0.1]
    tcp_position_c = [-0.15, -0.25, 0.05]
    tcp_position_d = [-0.15, -0.25, 0.1]
    so100_robot = So100Robot()
    # # so100_robot.
    # arm_joints = so100_robot.arm.get_joints()
    # print("arm_joints: ", arm_joints)
    # so100_robot.arm.move_zero()
    # so100_robot.arm.move_home()
    so100_robot.arm.move_up()

    # joints = so100_robot.arm.get_joints()
    # print("joints: ", joints)
    # joints[0] += 0.2
    # so100_robot.arm.set_joints(joints, wait=True, timeout=20, tolerance=0.1)

    # so100_robot.arm.move_zero(wait=True)
    # so100_robot.arm.move_up(wait=True)
    # so100_robot.arm.move_home(wait=True)

    # gripper_joint = so100_robot.gripper.get_joint()
    # print("gripper_joint: ", gripper_joint)
    # so100_robot.gripper.set_joint(1.6, wait=True, timeout=20, tolerance=0.1)
    # gripper_joint = so100_robot.gripper.get_joint()
    # print("gripper_joint: ", gripper_joint)

    # tcp_init_pose[0] += 0.1
    # so100_robot.arm.tcp_move_to_pose(tcp_init_pose, wait=True, timeout=20, tolerance=0.1)
    # tcp_init_pose[2] -= 0.05
    # so100_robot.arm.tcp_move_to_pose(tcp_init_pose, wait=True, timeout=20, tolerance=0.1)

    # so100_robot.gripper.set_joint(0.5, wait=True, timeout=20, tolerance=0.1)


if __name__ == '__main__':
    main()
