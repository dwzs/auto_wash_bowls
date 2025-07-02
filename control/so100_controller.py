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
import traceback


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
    
    def __init__(self, robot_instance, config):
        """初始化夹爪控制器"""
        self._robot = robot_instance
        gripper_config = config.get("gripper", {})
        
        self._joint_index = gripper_config.get("joint_index", 5)
        self.close_position = gripper_config.get("close_position", 1.0)
        self.open_position = gripper_config.get("open_position", 0.0)
        
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
    
    def set_joint(self, position: float, wait=True, timeout=None, tolerance=None) -> bool:
        # 使用配置的默认值
        if timeout is None:
            timeout = self._robot.motion_config.get("default_timeout", 5.0)
        if tolerance is None:
            tolerance = self._robot.motion_config.get("default_tolerance", 0.01)
            
        current_joints = self._robot.get_joints()
        if current_joints is None:
            return False
        
        full_joint_positions = current_joints.copy()
        full_joint_positions[self._joint_index] = position
        
        return self._robot.set_joints(full_joint_positions, wait, timeout, tolerance)
    
    def open(self, wait=True, timeout=None) -> bool:
        self.logger.info("打开夹爪")
        return self.set_joint(self.open_position, wait, timeout)
    
    def close(self, wait=True, timeout=None) -> bool:
        self.logger.info("关闭夹爪")
        return self.set_joint(self.close_position, wait, timeout)


class Arm:
    """机械臂控制类"""
    
    def __init__(self, robot_instance, config):
        """初始化机械臂控制器"""
        self._robot = robot_instance
        self.config = config
        
        arm_config = config.get("arm", {})
        self.joint_index = arm_config.get("joint_indices", [0, 1, 2, 3, 4])
        self.urdf_file = arm_config.get("urdf_file", "resources/so100_tcp.urdf")
        
        # 运动学配置
        self.kinematics_config = config.get("kinematics", {})
        self.motion_config = config.get("motion", {})
        
        self.logger = logging.getLogger('so100_arm')
        
        self._load_robot_model()
        self.line_joints = []

    def _load_robot_model(self):
        """加载机器人URDF模型用于逆解计算"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            urdf_path = os.path.join(script_dir, self.urdf_file)
            
            if os.path.exists(urdf_path):
                self.robot_model = ERobot.URDF(urdf_path)
                self.logger.info(f"成功加载URDF模型: {urdf_path}")
            else:
                self.robot_model = None
                self.logger.warning(f"URDF文件不存在: {urdf_path}")
        except Exception as e:
            self.robot_model = None
            self.logger.error(f"加载URDF模型失败: {e}")

        # 从机器人实例获取预设位置
        self.zero_joints = self._robot.zero_pose_joints[:5]
        self.home_joints = self._robot.home_pose_joints[:5]
        self.up_joints = self._robot.up_pose_joints[:5]
        
        limits = self._robot.joints_limits
        self.joints_limits = [limits[i] for i in self.joint_index]
        self.line_joints = []

    def _create_pose_matrix(self, pose_components):
        """将位姿列表转换为4x4齐次变换矩阵"""
        position = pose_components[:3]
        quaternion = pose_components[3:]  # [qx, qy, qz, qw]

        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()

        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = position
        return pose_matrix

    def _forward_kinematics(self, joint_angles, use_tool=False):
        """使用roboticstoolbox计算正运动学"""
        end_link = "tcp" if use_tool else "Fixed_Jaw"

        try:
            fk_pose_matrix = self.robot_model.fkine(np.array(joint_angles), end=end_link)
            
            position = fk_pose_matrix.t
            quaternion_xyzw = Rotation.from_matrix(fk_pose_matrix.R).as_quat()
            
            pose_list = list(position) + list(quaternion_xyzw)
            return pose_list
            
        except Exception as e:
            self._robot.get_logger().error(f"roboticstoolbox正运动学计算出错: {e}")
            return None

    def _inverse_kinematics(self, target_pose_list, initial_joint_guess=None, mask=None, use_tool=False):
        """计算逆运动学解"""
        end_link = "tcp" if use_tool else "Fixed_Jaw"

        try:
            target_pose_matrix = self._create_pose_matrix(target_pose_list) 
            
            if initial_joint_guess is None:
                current_joints = self.get_joints()
                if current_joints is None:
                    self.logger.error("无法获取当前关节角度作为IK初始猜测值")
                    return None
                q_guess = np.array(current_joints)
            else:
                q_guess = np.array(initial_joint_guess)
            
            if mask is None:
                mask = self.kinematics_config.get("default_mask", [1, 1, 1, 0, 1, 1])
            
            # 使用配置参数
            result = self.robot_model.ik_LM(
                target_pose_matrix, 
                q0=q_guess, 
                mask=mask, 
                end=end_link,
                tol=self.kinematics_config.get("ik_tolerance", 0.001),
                joint_limits=self.kinematics_config.get("ik_joint_limits", True),
                ilimit=self.kinematics_config.get("ik_iteration_limit", 1000),
                slimit=self.kinematics_config.get("ik_search_limit", 150),
                joint_limits=True
            )
            
            if len(result) >= 2:
                q_solution, success = result[0], result[1]
                
                if success:
                    self.logger.debug(f"逆解成功: {format_to_2dp(q_solution.tolist())}")
                    return q_solution.tolist()
                else:
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
            self.logger.error(traceback.format_exc())
            return None

    def flange_move_to_pose(self, target_pose_list, initial_joint_guess=None, mask=None, 
                           wait=True, timeout=None, tolerance=None):
        """移动法兰到目标位姿"""
        # 使用配置的默认值
        if timeout is None:
            timeout = self.motion_config.get("default_timeout", 10.0)
        if tolerance is None:
            tolerance = self.motion_config.get("default_tolerance", 0.01)
            
        if len(target_pose_list) == 3:
            current_flange = self.get_flange_pose()
            if current_flange is None:
                self.logger.error("无法获取当前法兰姿态")
                return False
            
            full_target_pose = list(target_pose_list) + current_flange[3:]
            self.logger.info(f"移动法兰到位置: {format_to_2dp(target_pose_list)}, 保持当前姿态")
            
        elif len(target_pose_list) == 7:
            full_target_pose = list(target_pose_list)
            self.logger.info(f"移动法兰到完整位姿: 位置{format_to_2dp(target_pose_list[:3])}, 姿态{format_to_2dp(target_pose_list[3:])}")
            
        else:
            self.logger.error(f"目标位姿参数长度错误: {len(target_pose_list)}, 应为3或7")
            return False
        
        joint_solution = self._inverse_kinematics(full_target_pose, initial_joint_guess, mask, use_tool=False)
        
        if joint_solution is None:
            return False
        
        return self.set_joints(joint_solution, wait, timeout, tolerance)

    def tcp_move_to_pose(self, target_pose_list, initial_joint_guess=None, mask=None, 
                        wait=True, timeout=None, tolerance=None):
        """移动TCP到目标位姿"""
        # 使用配置的默认值
        if timeout is None:
            timeout = self.motion_config.get("tcp_timeout", 20.0)
        if tolerance is None:
            tolerance = self.motion_config.get("tcp_tolerance", 0.01)
            
        if len(target_pose_list) == 3:
            current_tcp = self.get_tcp_pose()
            if current_tcp is None:
                self.logger.error("无法获取当前TCP姿态")
                return False
            
            full_target_pose = list(target_pose_list) + current_tcp[3:]
            self.logger.info(f"移动到位置: {format_to_2dp(target_pose_list)}, 保持当前姿态")
            
        elif len(target_pose_list) == 7:
            full_target_pose = list(target_pose_list)
            self.logger.debug(f"移动到位姿: 位置{format_to_2dp(target_pose_list[:3])}, 姿态{format_to_2dp(target_pose_list[3:])}")
            
        else:
            self.logger.error(f"目标位姿参数长度错误: {len(target_pose_list)}, 应为3或7")
            return False
        
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
        
        self.line_joints.append(joint_solution)
        return self.set_joints(joint_solution, wait, timeout, tolerance)

    def get_joints(self) -> Optional[List[float]]:
        """获取机械臂关节角度（不包括夹爪）"""
        joints = self._robot.get_joints()
        if joints is None:
            return None
        
        return format_to_2dp(joints)

    def get_flange_pose(self) -> Optional[List[float]]:
        """获取末端执行器法兰(flange)位姿"""
        arm_joint_angles = self.get_joints()
        flange_pose = self._forward_kinematics(arm_joint_angles, use_tool=False)
        return format_to_2dp(flange_pose)

    def get_tcp_pose(self) -> Optional[List[float]]:
        """获取TCP位姿"""
        arm_joint_angles = self.get_joints()
        tcp_pose = self._forward_kinematics(arm_joint_angles, use_tool=True)
        return format_to_2dp(tcp_pose)
    
    def set_joints(self, joints: List[float], wait=True, timeout=None, tolerance=None) -> bool:
        # 使用配置的默认值
        if timeout is None:
            timeout = self.motion_config.get("default_timeout", 10.0)
        if tolerance is None:
            tolerance = self.motion_config.get("default_tolerance", 0.01)
            
        if len(joints) != len(self.joint_index):
            self.logger.error(f"机械臂关节数量错误: 需要{len(self.joint_index)}个关节位置, 实际关节数量: {len(joints)}")
            return False
        
        current_gripper_pos = self._robot.gripper.get_joint()
        if current_gripper_pos is None:
            return False
        
        full_joint_positions = joints + [current_gripper_pos]
        
        self.logger.debug(f"发送给机器人的完整关节位置: {format_to_2dp(full_joint_positions)}")
        
        return self._robot.set_joints(full_joint_positions, wait, timeout, tolerance)

    def move_home(self, wait=True) -> bool:
        """将机械臂移动到home位置"""
        self.logger.info("移动到home位置")
        return self.set_joints(self.home_joints, wait=wait)
    
    def move_up(self, wait=True) -> bool:
        """将机械臂移动到up位置"""
        self.logger.info("移动到抬起位置")
        return self.set_joints(self.up_joints, wait=wait)
    
    def move_zero(self, wait=True) -> bool:
        """将机械臂移动到zero位置"""
        self.logger.info("移动到zero位置")
        return self.set_joints(self.zero_joints, wait=wait)
    
    def move_line(self, position_a, position_b, wait=True, timeout=None, tolerance=None):
        """控制机械臂TCP从A点直线运动到B点"""
        # 使用配置的默认值
        if timeout is None:
            timeout = self.motion_config.get("default_timeout", 10.0)
        if tolerance is None:
            tolerance = self.motion_config.get("default_tolerance", 0.01)
            
        try:
            if len(position_a) != 3 or len(position_b) != 3:
                self.logger.error(f"位置参数长度错误: A点长度{len(position_a)}, B点长度{len(position_b)}, 应为3")
                return False
            
            current_tcp = self.get_tcp_pose()
            if current_tcp is None:
                self.logger.error("无法获取当前TCP姿态")
                return False
            
            current_orientation = current_tcp[3:]
            pose_a = list(position_a) + current_orientation
            pose_b = list(position_b) + current_orientation
            
            self.logger.info(f"开始直线运动: A{format_to_2dp(position_a)} -> B{format_to_2dp(position_b)}, 保持当前姿态")
            
            distance = np.linalg.norm(np.array(position_b) - np.array(position_a))
            
            # 使用配置参数
            line_config = self.motion_config.get("line_motion", {})
            max_num_points = line_config.get("max_points", 100)
            min_num_points = line_config.get("min_points", 2)
            point_per_meter = line_config.get("points_per_meter", 100)
            
            num_points = max(int(distance * point_per_meter), min_num_points)
            num_points = min(num_points, max_num_points)
            
            self.logger.info(f"直线距离: {distance:.3f}m, 插值点数: {num_points}")
            
            waypoints = []
            for i in range(num_points + 1):
                t = i / num_points
                pos_interp = np.array(position_a) + t * (np.array(position_b) - np.array(position_a))
                waypoint = list(pos_interp) + current_orientation
                waypoints.append(waypoint)
            
            self.logger.debug("直线运动关节轨迹:")
            for i, waypoint in enumerate(waypoints):
                self.logger.debug(f"  轨迹点 {i+1}: {format_to_2dp(waypoint)}")

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
            self.logger.error(traceback.format_exc())
            return False

    def move_direction_abs(self, direction: list[float], wait=True, timeout=None, tolerance=None):
        """控制机械臂TCP从当前位置沿指定向量运动（绝对坐标系）"""
        # 使用配置的默认值
        if timeout is None:
            timeout = self.motion_config.get("default_timeout", 10.0)
        if tolerance is None:
            tolerance = self.motion_config.get("default_tolerance", 0.01)
            
        try:
            current_tcp = self.get_tcp_pose()   
            if current_tcp is None:
                self.logger.error("无法获取当前TCP姿态")
                return False
            
            new_position = np.array(current_tcp[:3]) + direction
            position_target = list(new_position)
            
            self.logger.info(f"开始沿方向移动: {format_to_2dp(direction)}")

            return self.move_line(current_tcp[:3], position_target, wait, timeout, tolerance)
        
        except Exception as e:
            self.logger.error(f"沿方向移动过程中发生错误: {e}")
            self.logger.error(traceback.format_exc())

    def move_direction_relative(self, direction: list[float], distance: float, 
                              wait=True, timeout=None, tolerance=None):
        """控制机械臂TCP从当前位置沿指定方向移动指定距离（相对坐标系）"""
        # 使用配置的默认值
        if timeout is None:
            timeout = self.motion_config.get("default_timeout", 10.0)
        if tolerance is None:
            tolerance = self.motion_config.get("default_tolerance", 0.01)
            
        try:
            current_tcp = self.get_tcp_pose()   
            if current_tcp is None:
                self.logger.error("无法获取当前TCP姿态")
                return False
            
            tcp_quat = current_tcp[3:]
            rot = Rotation.from_quat(tcp_quat).as_matrix()
            world_direction = rot @ np.array(direction)
            new_position = np.array(current_tcp[:3]) + world_direction * distance

            current_orientation = current_tcp[3:]
            pose_target = list(new_position) + current_orientation

            self.logger.info(f"开始沿相对方向移动: {format_to_2dp(direction)} 距离 {distance:.3f}m")

            return self.move_line(current_tcp[:3], pose_target, wait, timeout, tolerance)
        
        except Exception as e:
            self.logger.error(f"沿方向移动过程中发生错误: {e}")
            self.logger.error(traceback.format_exc())


class So100Robot(Node):
    """SO-100机器人控制接口主类"""
    
    def __init__(self, config_file=None):
        # 加载配置
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), "config", "so100_config.json")
        
        self.config = self._load_config(config_file)
        
        # 配置日志
        logging_config = self.config.get("logging", {})
        colorlog.basicConfig(
            level=getattr(logging, logging_config.get("level", "INFO")),
            format=logging_config.get("format", '%(log_color)s[%(levelname)s][%(filename)s:%(lineno)d] %(message)s'),
            datefmt=logging_config.get("date_format", '%Y-%m-%d %H:%M:%S')
        )
        self.logger = logging.getLogger('so100_controller')
        
        # 初始化ROS
        if not rclpy.ok():
            rclpy.init()
            
        # ROS配置
        ros_config = self.config.get("ros", {})
        node_name = ros_config.get("node_name", "So100Robot")
        super().__init__(node_name)
        
        # 机器人配置
        robot_config = self.config.get("robot", {})
        self.is_robot_connected = False
        self.servo_ids = robot_config.get("servo_ids", [1, 2, 3, 4, 5, 6])
        self.TOTAL_JOINTS_COUNT = robot_config.get("total_joints_count", 6)
        self.pub_rate = robot_config.get("pub_rate", 1000)
        self.connection_timeout = robot_config.get("connection_timeout", 10.0)
        
        self.current_joint_positions = [0.0] * self.TOTAL_JOINTS_COUNT
        
        # 标定配置
        calibration_config = self.config.get("calibration", {})
        self.calibration_file = calibration_config.get("calibration_file", "so100_calibration.json")

        # 运动配置（供子类使用）
        self.motion_config = self.config.get("motion", {})

        # 加载标定数据
        calibration_data = self.load_calibration_data()

        # 创建发布器和订阅器
        topics_config = ros_config.get("topics", {})
        joint_commands_topic = topics_config.get("joint_commands", "so100_joints_commands")
        joint_states_topic = topics_config.get("joint_states", "so100_joints_states")
        
        self.simple_command_publisher = self.create_publisher(
            JointState, joint_commands_topic, self.pub_rate)
        
        self.joint_state_subscriber = self.create_subscription(
            JointState, joint_states_topic, self._joint_state_callback, 10)
        
        # 创建机械臂和夹爪实例
        self.arm = Arm(self, self.config)
        self.gripper = Gripper(self, self.config)
        
        # 等待机器人就绪
        self._wait_for_robot(self.connection_timeout)
    
    def _load_config(self, config_file: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"成功加载配置文件: {config_file}")
                return config
        except FileNotFoundError:
            print(f"警告: 配置文件未找到 {config_file}, 使用默认配置")
            return {}
        except json.JSONDecodeError as e:
            print(f"警告: 配置文件格式错误 {e}, 使用默认配置")
            return {}

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

            self.logger.info(f'成功加载标定数据: {self.calibration_file}')
            self.logger.info(f'关节限制: {self.joints_limits}')
            self.logger.info(f'zero_pose关节位置: {self.zero_pose_joints}')
            self.logger.info(f'home_pose关节位置: {self.home_pose_joints}')
            self.logger.info(f'up_pose关节位置: {self.up_pose_joints}')
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
    
    def set_joints(self, joint_positions, wait=True, timeout=None, tolerance=None) -> bool:
        # 使用配置的默认值
        if timeout is None:
            timeout = self.motion_config.get("default_timeout", 10.0)
        if tolerance is None:
            tolerance = self.motion_config.get("default_tolerance", 0.01)
            
        if not self.is_robot_connected:
            self.logger.warning("机器人未连接")
            return False
        
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
        joints_diffs = []
        while time.time() - start_time < timeout:
            rclpy.spin_once(self, timeout_sec=0.01)
            current_joint_positions = self.get_joints()
            joints_diffs = [abs(current - target) for current, target in zip(current_joint_positions, joint_positions)]
            if all(diff < tolerance for diff in joints_diffs):
                return True
            time.sleep(0.01)
            
        self.logger.warning("到达目标位置超时")
        self.logger.warning(f"target_joint_positions: {joint_positions}")
        self.logger.warning(f"current_joint_positions: {format_to_2dp(self.current_joint_positions)}")
        self.logger.warning(f"有关节误差超过{tolerance}的关节，当前误差: {joints_diffs}")
        
        return False
    
    def close(self):
        """关闭节点"""
        self.logger.info("关闭SO-100控制节点")
        self.destroy_node()


def main():
    """示例用法"""
    tcp_position_a = [0.15, -0.25, 0.05]    
    tcp_position_b = [-0.15, -0.25, 0.05]
    
    so100_robot = So100Robot()
    so100_robot.arm.move_line(tcp_position_a, tcp_position_b, wait=True, timeout=20, tolerance=0.04)


if __name__ == '__main__':
    main()
