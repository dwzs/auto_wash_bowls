#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-100机械臂控制器 - 最简配置版本"""

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import json
import time
import os
import numpy as np
from typing import List, Optional
import threading
from scipy.spatial.transform import Rotation
from roboticstoolbox import ERobot
from loguru import logger

TARGET_JOINTS = []

def format_to_2dp(value):
    """将数字或包含数字的列表/数组格式化为小数点后两位"""
    if isinstance(value, (list, tuple, np.ndarray)):
        if isinstance(value, np.ndarray):
            return np.round(value, 2)
        else:
            return type(value)([format_to_2dp(x) for x in value])
    elif isinstance(value, (int, float, np.number)):
        return round(float(value), 2)
    else:
        return value

class Arm(Node):
    """机械臂控制类"""
    
    def __init__(self, config_file=None):
        super().__init__('arm_controller')
        
        # 直接加载配置文件
        config_path = config_file or os.path.join(os.path.dirname(__file__), "config", "arm_config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)
        logger.info(f"成功加载配置文件: {config_path}")

        # 创建发布器和订阅器
        self.joint_cmd_pub = self.create_publisher(JointState, self.cfg["ros"]["cmd_topic"], 10)
        self.joint_state_sub = self.create_subscription(
            JointState, self.cfg["ros"]["state_topic"], self._joint_state_callback, 10)
        
        # 状态变量
        self.current_all_joints = None
        self.current_arm_joints = None
        self.lock = threading.Lock()
        self._shutdown = False
        
        # 启动后台线程处理ROS消息
        self.spin_thread = threading.Thread(target=self._spin_thread, daemon=True)
        self.spin_thread.start()
        
        # 加载标定数据
        with open(self.cfg["calibration_file"], 'r', encoding='utf-8') as f:
            self.cal = json.load(f)
        logger.info(f"成功加载标定文件: {self.cfg['calibration_file']}")
        
        # 预设位置
        poses = self.cal["poses_joints_offseted"]
        self.zero_joints = poses["zero_pose_joints"][:5]
        self.home_joints = poses["home_pose_joints"][:5]
        self.up_joints = poses["up_pose_joints"][:5]
        
        # 关节限制
        limits = self.cal["joint_limits_offseted"]
        self.joints_limits = [[limits[str(i+1)]["min"], limits[str(i+1)]["max"]] for i in range(5)]
        
        # 加载机器人模型
        self._load_robot_model()
        
        # 等待连接
        if not self._wait_for_connection():
            logger.error("Arm __init__ failed: _wait_for_connection failed")
            raise RuntimeError("Failed to wait for connection")

        logger.info("Arm __init__ success")
    
    def _spin_thread(self):
        """后台线程处理ROS消息"""
        while rclpy.ok() and not self._shutdown:
            try:
                rclpy.spin_once(self, timeout_sec=0.01)
            except Exception as e:
                logger.error(f"Spin thread error: {e}")
                break
    
    def _load_robot_model(self):
        """加载机器人URDF模型"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            urdf_path = os.path.join(script_dir, "..", self.cfg["urdf_file"])
            
            if not os.path.exists(urdf_path):
                logger.error(f"URDF file not found: {urdf_path}")
                return False

            self.robot_model = ERobot.URDF(urdf_path)
            if self.robot_model is None:
                logger.error(f"failed to load URDF file: {urdf_path}")
                return False

            logger.info(f"load_robot_model success: {urdf_path}")
            return True
        
        except Exception as e:
            logger.error(f"load_robot_model failed: {e}")
            return False
    
    def _joint_state_callback(self, msg):
        """关节状态回调"""
        if self._shutdown:  # 添加shutdown检查
            return
        
        with self.lock:
            self.current_all_joints = list(msg.position)
            if len(self.current_all_joints) >= len(self.cfg["joint_indices"]):
                self.current_arm_joints = [self.current_all_joints[i] for i in self.cfg["joint_indices"]]
            
            if self.current_arm_joints is None:
                logger.error("current_arm_joints is None")

    def _wait_for_connection(self, timeout=10.0):
        """等待关节状态数据"""
        start_time = time.time()
        logger.info("等待关节状态数据...")
        
        while time.time() - start_time < timeout:
            wait_time = time.time() - start_time
            if self.current_arm_joints is None:
                logger.info(f"waiting {wait_time} seconds for joint state data...")
            else:
                logger.info(f"wait for connection success: current_arm_joints is {self.current_arm_joints}")
                return True
            time.sleep(0.1)
        
        logger.error(f"wait for connection timeout({timeout}) seconds")
        return False
    
    def _send_arm_joints(self, arm_joints: List[float]) -> bool:
        """发送机械臂命令"""
        if len(arm_joints) != len(self.cfg["joint_indices"]):
            logger.error(f"send arm command failed: actual joints length({len(arm_joints)}) != expected joints length({len(self.cfg['joint_indices'])})")
            return False
        
        # 确保所有值都是float类型
        arm_joints_float = [float(j) for j in arm_joints]

        try:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.position = arm_joints_float + [math.nan]  # 夹爪位置用NaN跳过
            self.joint_cmd_pub.publish(msg)
            return True
        except Exception as e:
            logger.error(f"send arm command failed: {e}")
            return False

    
    def _out_joint_limits(self, joints: List[float]) -> bool:
        """检查关节是否超出限制"""
        joints_limit = self.joints_limits

        for i, joint_val in enumerate(joints):  
            if joint_val < joints_limit[i][0]:
                logger.error(f"joint{i}({joint_val}) < lower limit({joints_limit[i][0]})")
                return True
            elif joint_val > joints_limit[i][1]:
                logger.error(f"joint{i}({joint_val}) > upper limit({joints_limit[i][1]})")
                return True

        return False
    
    # def _reduce_joints_diff(self, target_joints: List[float], tolerance: float = 0.01) -> List[float]:
    #     """减少关节差值"""
    #     for i in range(len(current_joints)):
    #         current_joints = self.get_joints()
    #         joint_diff = current_joints[i] - target_joints[i]
    #         if abs(joint_diff) > tolerance:
    #             self._send_arm_joints(target_joints)
    #             logger.info(f"reduce joint diff: {current_joints[i]} -> {target_joints[i]}")
    #         else:
    #             logger.info(f"joint{i} diff is small enough: {current_joints[i]} -> {target_joints[i]}")
        
    #     return joint_diff

    # def _wait_for_joints(self, target_joints: List[float], timeout: float = 100, tolerance: float = 0.04) -> bool:
    #     """等待到达目标位置"""
    #     if timeout <= 0:
    #         return True
        
    #     start_time = time.time()

    #     # 等待关节到达目标位置
    #     while time.time() - start_time < timeout:
    #         current = self.get_joints()
    #         if current and len(current) == len(target_joints):
    #             if all(abs(c - t) < tolerance for c, t in zip(current, target_joints)):
    #                 return True
    #         time.sleep(0.01)
        
    #     # 超时，打印当前关节和目标关节
    #     current = self.get_joints()
    #     joint_diff = [current[i] - target_joints[i] for i in range(len(target_joints))]
    #     logger.error(f"timeout({timeout}), current_joints({current}), target_joints({target_joints}), joint_diff({joint_diff})")
    #     return False
    def _wait_for_joints(self, target_joints: List[float], timeout: float = 100, tolerance: float = 0.04) -> bool:
        """等待到达目标位置"""
        if timeout <= 0:
            return True
        
        start_time = time.time()
        # 上一次误差记录：每个关节保存上一次的误差值
        last_errors = [float('inf')] * len(target_joints)
        adjusted_targets = target_joints.copy()


        current = self.get_joints()
        last_errors = [abs(current[i] - target_joints[i]) for i in range(len(current))]
        error_direction = 1

        # 等待关节到达目标位置
        while time.time() - start_time < timeout:
            current = self.get_joints()

            # 计算当前误差
            current_errors = [target_joints[i] - current[i] for i in range(len(current))]
            diff_errors = [current_errors[i] - last_errors[i] for i in range(len(current_errors))]
            print(f"last_errors   : {last_errors}")
            print(f"\ncurrent_errors: {current_errors}")
            print(f"diff_errors   : {diff_errors}") 
            # 检查是否已到达容差范围
            if all(abs(err) < tolerance for err in current_errors):
                print(f"all(err < tolerance for err in current_errors)")
                print(f"current_errors: {current_errors}")
                return True
            
            # 检查误差是否没有减小，如果没有减小则调整目标
            for i in range(len(current_errors)):
                if abs(current_errors[i]) > tolerance:
                    # print(f"current_errors[i]: {current_errors[i]}")
                    if abs(current_errors[i]) >= abs(last_errors[i]):
                        print("joint i: ", i, " stop moving !")
                        # 误差没有减小，调整目标值
                        error_direction = 1 if current_errors[i] > 0 else -1
                        # adjustment = tolerance * 0.5 * error_direction
                        adjustment = tolerance * error_direction
                        print(f"adjustment: {adjustment}")
                        adjusted_targets[i] += adjustment

            # 更新上一次误差
            last_errors = current_errors.copy()

            key = 20
            for i in range(len(adjusted_targets)):
                if abs(adjusted_targets[i] - target_joints[i]) > tolerance * key:
                    print(f"joint{i} > tolerance * {key}")
                    adjusted_targets[i] = target_joints[i] + tolerance * key * error_direction

            # 发送新的目标位置
            print(f"original_targets: {target_joints}")
            print(f"adjusted_targets: {adjusted_targets}")
            if not self._out_joint_limits(adjusted_targets):
                self._send_arm_joints(adjusted_targets)

            time.sleep(0.1)
            # input("press enter to continue..........")
        
        # 超时，打印当前关节和目标关节
        current = self.get_joints()
        if current:
            joint_diff = [current[i] - target_joints[i] for i in range(len(target_joints))]
            for i, diff in enumerate(joint_diff):
                if abs(diff) > tolerance:
                    logger.error(f"timeout({timeout}), current_joints({current}), target_joints({target_joints}), errors({joint_diff})")
        
        return False
    
    def _create_pose_matrix(self, pose_components):
        """将位姿列表转换为4x4齐次变换矩阵"""
        if len(pose_components) != 7:
            logger.error(f"pose_components length({len(pose_components)}) != 7")
            return None

        position = pose_components[:3]
        quaternion = pose_components[3:]  # [qx, qy, qz, qw]
        
        rotation_matrix = Rotation.from_quat(quaternion).as_matrix()
        
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = position
        return pose_matrix
    
    def _forward_kinematics(self, joint_angles, use_tool=True):
        """计算正运动学"""
        if self.robot_model is None:
            logger.error("机器人模型未加载")
            return None
        
        end_link = "tcp" if use_tool else "Fixed_Jaw"
        
        try:
            fk_pose_matrix = self.robot_model.fkine(np.array(joint_angles), end=end_link)
            
            position = fk_pose_matrix.t
            quaternion_xyzw = Rotation.from_matrix(fk_pose_matrix.R).as_quat()
            
            pose_list = list(position) + list(quaternion_xyzw)
            return pose_list
            
        except Exception as e:
            logger.error(f"正运动学计算失败: {e}")
            return None
    
    def _inverse_kinematics(self, target_pose_list, ik_init_pose, mask, use_tool=True):
        """计算逆运动学"""
        if self.robot_model is None:
            logger.error("机器人模型未加载")
            return None
        
        end_link = "tcp" if use_tool else "Fixed_Jaw"
        
        try:
            target_pose_matrix = self._create_pose_matrix(target_pose_list)
            
            q_guess = np.array(ik_init_pose)
            
            result = self.robot_model.ik_LM(
                target_pose_matrix,
                q0=q_guess,
                mask=mask,
                end=end_link,
                tol=self.cfg["kinematics"]["ik_tolerance"],
                joint_limits=True,
                ilimit=self.cfg["kinematics"]["ik_max_iterations"],
                slimit=self.cfg["kinematics"]["ik_max_searches"]
            )
            
            if len(result) >= 2:
                q_solution, success = result[0], result[1]
                
                if success:
                    logger.debug(f"逆解成功: {format_to_2dp(q_solution.tolist())}")
                    return q_solution.tolist()
                else:
                    logger.warning(f"逆解失败，目标位姿: {target_pose_list}")
                    return None
            else:
                logger.error(f"ik_LM返回值格式异常: {result}")
                return None
                
        except Exception as e:
            logger.error(f"逆解计算失败: {e}")
            return None
    

    def cacul_ik_joints_within_limits_from_pose(self, target_pose_list, use_tool=True):
        # 1. 使用当前关节角度逆解
        current_joints = self.get_joints()
        if current_joints is None:
            logger.error("get_joints failed !")
            return None

        mask = self.cfg["kinematics"]["default_mask"]
        joint_solution = self._inverse_kinematics(target_pose_list, current_joints, mask, use_tool=use_tool)
        if joint_solution is None:
            logger.error(f"_inverse_kinematics failed !")
            logger.info(f"target_pose_list: {target_pose_list}")
            return None

        # 2. 如果关节超出限制，使用零初始位姿逆解
        if self._out_joint_limits(joint_solution):
            logger.warning("joints out of limits, try zero init pose for IK !")
            init_ik_pose = [0.0, 0.0, 0.0, 0.0, 0.0]
            joint_solution = self._inverse_kinematics(target_pose_list, init_ik_pose, mask, use_tool=use_tool)

        if joint_solution is None:
            logger.error(f"joint_solution is None !")
            return None
        
        # 3. 检查关节是否超出限制
        if self._out_joint_limits(joint_solution):
            logger.error("joints out of limits !")
            return None

        return joint_solution

    # ==================== 基础接口 ====================
    
    def get_joints(self) -> Optional[List[float]]:
        """获取机械臂关节角度"""
        with self.lock:
            if self.current_arm_joints is not None:
                return self.current_arm_joints.copy()
            else:
                logger.error("get_joints failed !")
                return None
    
    def get_pose(self, tcp: bool = True) -> Optional[List[float]]:
        """获取位姿
        
        Returns:
            位姿列表 [x, y, z, roll, pitch, yaw] 或 None (欧拉角格式)
        """
        joints = self.get_joints()
        if joints is None:
            logger.error("get_joints failed !")
            return None

        pose = self._forward_kinematics(joints, use_tool=tcp)
        if pose is None:
            logger.error("forward_kinematics failed !")
            return None
        
        # 转换四元数为欧拉角
        position = pose[:3]
        quaternion = pose[3:]
        rotation = Rotation.from_quat(quaternion)
        euler_angles = rotation.as_euler('xyz')  # [roll, pitch, yaw]
        
        return list(position) + list(euler_angles)

    
    # ==================== 运动控制接口 ====================
    
    def move_to_joints(self, joints: List[float], timeout: float = 100, tolerance: float = 0.04) -> bool:
        """设置机械臂关节位置
        
        Args:
            joints: 目标关节位置 (5个关节)
            timeout: 超时时间，>0时等待到位，<=0时不等待
            tolerance: 位置容差
            
        Returns:
            成功返回True，失败或超时返回False
        """

        global TARGET_JOINTS
        TARGET_JOINTS = joints

        if len(joints) != len(self.cfg["joint_indices"]):
            logger.error(f"actual joints length({len(joints)}) != expected joints length({len(self.cfg['joint_indices'])}) !")
            return False
        
        # 检查关节范围
        if self._out_joint_limits(joints):
            logger.error("joints out of limits !")
            return False

        if not self._send_arm_joints(joints):
            logger.error("send_arm_command failed !")
            return False
        
        if not self._wait_for_joints(joints, timeout, tolerance):
            logger.error("wait for joints failed !")
            return False

        return True
    

    def move_to_pose(self, target_pose_list, timeout: float = 100, tolerance: float = 0.004, tcp: bool = True) -> bool:
        """移动到目标位姿
        
        Args:
            target_pose_list: 目标位姿，可以是：
                - 3个位置参数: [x, y, z] (保持当前姿态)
                - 6个位姿参数: [x, y, z, roll, pitch, yaw] (欧拉角，弧度)
            timeout: 超时时间
            tolerance: 位置容差
            tcp: True为TCP移动，False为flange移动，默认为True
        """
        
        if len(target_pose_list) == 3:
            current_pose = self.get_pose(tcp=tcp)
            if current_pose is None:
                pose_type = "TCP" if tcp else "flange"
                logger.error(f"get_pose {pose_type} failed: !")
                return False
            
            target_position = target_pose_list
            target_rotation_euler = list(current_pose[3:])

        elif len(target_pose_list) == 6:
            # 将欧拉角转换为四元数
            target_position = target_pose_list[:3]
            target_rotation_euler = target_pose_list[3:]  # [roll, pitch, yaw]
            
        else:
            logger.error(f"target pose length({len(target_pose_list)}) != 3 or 6 !")
            return False
        
        # 转换为四元数
        target_rotation = Rotation.from_euler('xyz', target_rotation_euler)
        target_rotation_quaternion = target_rotation.as_quat()  # [qx, qy, qz, qw]
        
        target_pose_quat = list(target_position) + list(target_rotation_quaternion)
        target_pose_quat_reachable = self._tuning_rotation_reachable(target_pose_quat)

        ik_joints = self.cacul_ik_joints_within_limits_from_pose(target_pose_quat_reachable, use_tool=tcp)
        if ik_joints is None:
            logger.error(f"cacul_ik_joints_within_limits_from_pose failed !")
            return False

        if not self.move_to_joints(ik_joints, timeout, tolerance):
            logger.error("move_to_joints failed !")
            return False

        return True

    def move_line(self, start_position, end_position, step=0.01, timeout=100, tolerance=0.05, tcp: bool = True) -> bool:
        """沿直线运动 - 先计算所有轨迹点，再依次运动"""
        start_position = np.array(start_position)
        end_position = np.array(end_position)
        direction = end_position - start_position
        distance = np.linalg.norm(direction)
        
        if distance < 1e-6:  # 距离太小，直接返回
            logger.warning("start and end position too close, skip motion !")
            return True
        
        # 计算步数和单位方向向量
        steps = max(1, int(distance / step))
        
        # 预先计算所有轨迹点
        trajectory_points = []
        for i in range(steps + 1):  # 包括起点和终点
            alpha = i / steps
            point = start_position + alpha * direction
            trajectory_points.append(point.tolist())
        
        logger.info(f"distance({distance:.3f}) m, steps({steps}), trajectory_points({len(trajectory_points)})")
        print(f"trajectory_points: {trajectory_points}")
        
        # 依次运动到每个轨迹点
        for i, point in enumerate(trajectory_points):
            success = self.move_to_pose(point, timeout=timeout, tolerance=tolerance, tcp=tcp)
            if not success:
                logger.error(f"move_to_pose failed at point({i+1}) !")
                return False
            
            if i % 10 == 0:  # 每10个点打印一次进度
                logger.info(f"progress({i+1}/{len(trajectory_points)})")      
        
        logger.info("move_line: motion completed !")
        return True

    # ==================== 预设位置 ====================
    
    def move_home(self, timeout: Optional[float] = None) -> bool:
        """移动到home位置"""
        logger.info("移动到home位置")
        if timeout is None:
            timeout = 100
        return self.move_to_joints(self.home_joints, timeout=timeout)
    
    def move_to_up_pose(self, timeout: Optional[float] = None) -> bool:
        """移动到up位置"""
        logger.info("移动到up位置")
        if timeout is None:
            timeout = 100
        return self.move_to_joints(self.up_joints, timeout=timeout)
    
    def move_to_joint_zero_pose(self, timeout: Optional[float] = None) -> bool:
        """移动到zero位置"""
        logger.info("移动到zero位置")
        if timeout is None:
            timeout = 100
        return self.move_to_joints(self.zero_joints, timeout=timeout)

    def destroy_node(self):
        """安全销毁节点"""
        self._shutdown = True
        if hasattr(self, 'spin_thread'):
            self.spin_thread.join(timeout=1.0)  # 等待线程结束
        super().destroy_node()


    def _get_theta_from_position(self, position):
        """根据位置获取角度，范围 [0, 2π]"""
        x, y = position[0], position[1]
        theta = np.arctan2(y, x)
        if theta < 0:
            theta += 2 * np.pi
        return theta

    def _get_theta_from_rotation(self, rotation):
        """根据旋转获取角度，范围 [0, 2π]"""
        if len(rotation) != 3:
            logger.error(f"rotation length({len(rotation)}) != 3")
            return 0.0
            
        # 获取旋转后的Y轴向量
        rot = Rotation.from_euler('xyz', rotation)
        vy = rot.apply([0, 1, 0])
        
        # 在XY平面投影
        x, y = vy[0], vy[1]
        theta = np.arctan2(y, x)
        
        if theta < 0:
            theta += 2 * np.pi
        
        # 针对so100夹爪，需要把夹爪y轴方向转180度
        if theta >= np.pi:
            theta = theta - np.pi
        else:
            theta = np.pi + theta
        
        return theta

    def _roll_rotation(self, rotation, angle, axis=[0, 0, 1]):
        """绕轴旋转欧拉角"""
        if len(rotation) != 3:
            logger.error(f"rotation length({len(rotation)}) != 3")
            return rotation
            
        original_rotation = Rotation.from_euler('xyz', rotation)
        axis = np.array(axis) / np.linalg.norm(axis)
        axis_rotation = Rotation.from_rotvec(angle * axis)
        combined_rotation = axis_rotation * original_rotation
        return combined_rotation.as_euler('xyz').tolist()

    def _is_rotation_reachable(self, pose):
        """检查旋转是否可达"""
        if len(pose) < 6:
            logger.error(f"pose length({len(pose)}) < 6")
            return True  # 默认可达
            
        position = pose[:3]
        rotation = pose[3:6]  # 只取前3个旋转分量
        
        theta_position = self._get_theta_from_position(position)
        theta_rotation = self._get_theta_from_rotation(rotation)
        
        return abs(theta_position - theta_rotation) < 0.01  # 1度容差

    def _tuning_rotation_reachable(self, pose):
        """调整位姿使旋转可达
        
        Args:
            pose: [x, y, z, qx, qy, qz, qw] 四元数格式
        
        Returns:
            调整后的位姿或原位姿
        """
        if len(pose) != 7:
            logger.warning(f"pose length({len(pose)}) != 7, skip rotation tuning")
            return pose
        
        try:
            # 转换四元数为欧拉角
            position = pose[:3]
            quaternion = pose[3:]
            
            # 验证四元数
            if len(quaternion) != 4:
                logger.error(f"quaternion length({len(quaternion)}) != 4")
                return pose
                
            rotation = Rotation.from_quat(quaternion).as_euler('xyz')
            
            # 构建6元素位姿用于检查
            pose_6d = list(position) + list(rotation)
            
            # 检查是否需要调整
            if self._is_rotation_reachable(pose_6d):
                return pose  # 已经可达，不需要调整
            
            # 计算调整角度
            theta_position = self._get_theta_from_position(position)
            theta_rotation = self._get_theta_from_rotation(rotation)
            theta_diff = theta_position - theta_rotation
            
            # 调整旋转
            new_rotation = self._roll_rotation(rotation, theta_diff)
            
            # 转换回四元数
            new_quat = Rotation.from_euler('xyz', new_rotation).as_quat()
            new_pose = list(position) + list(new_quat)
            
            # 验证调整结果
            new_pose_6d = list(position) + new_rotation
            if self._is_rotation_reachable(new_pose_6d):
                logger.debug(f"Rotation tuned: {np.degrees(theta_diff):.1f}°")
                return new_pose
            else:
                logger.warning("Rotation tuning failed, using original pose")
                return pose
                
        except Exception as e:
            logger.error(f"Rotation tuning error: {e}")
            return pose

    def move_to_direction_abs(self, vector, timeout: float = 100, tolerance: float = 0.004, tcp: bool = True) -> bool:
        """沿绝对方向移动
        
        Args:
            vector: 移动向量 [dx, dy, dz] (世界坐标系)
            timeout: 超时时间
            tolerance: 位置容差
            tcp: True为TCP移动，False为flange移动
            
        Returns:
            成功返回True，失败返回False
        """
        current_pose = self.get_pose(tcp=tcp)
        if current_pose is None:
            logger.error("get_pose failed!")
            return False
        
        # 计算目标位置 = 当前位置 + 移动向量
        target_position = [current_pose[i] + vector[i] for i in range(3)]
        
        return self.move_to_pose(target_position, timeout, tolerance, tcp)
    
    def move_to_direction_relative(self, vector, timeout: float = 100, tolerance: float = 0.004, tcp: bool = True) -> bool:
        """沿相对方向移动（相对于当前姿态）
        
        Args:
            vector: 移动向量 [dx, dy, dz] (当前姿态坐标系)
            timeout: 超时时间  
            tolerance: 位置容差
            tcp: True为TCP移动，False为flange移动
            
        Returns:
            成功返回True，失败返回False
        """
        current_pose = self.get_pose(tcp=tcp)
        if current_pose is None:
            logger.error("get_pose failed!")
            return False
        
        # 获取当前姿态的旋转矩阵
        current_rotation = Rotation.from_euler('xyz', current_pose[3:])
        rotation_matrix = current_rotation.as_matrix()
        
        # 将相对向量转换到世界坐标系
        vector_world = rotation_matrix @ np.array(vector)
        
        # 计算目标位置
        target_position = [current_pose[i] + vector_world[i] for i in range(3)]
        
        return self.move_to_pose(target_position, timeout, tolerance, tcp)


def main():
    """简单的测试函数"""
    rclpy.init()
    arm = None
    
    try:
        arm = Arm()
        print(f"机械臂初始化成功")
        print(f"当前关节: {arm.get_joints()}")
        print(f"当前TCP位姿: {arm.get_pose()}")
        
        # 保持运行状态
        print("机械臂就绪，按 Ctrl+C 退出")
        while True:
            time.sleep(1.0)

        # # direction = [0.05, 0.0, 0.0]
        # direction = [0.0, -0.1, 0.0]
        # # direction = [0.0, 0.0, 0.0]
        # arm.move_to_direction_abs(direction, timeout=10, tolerance=0.004, tcp=True)

    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if arm:
            try:
                arm.destroy_node()
            except Exception as e:
                logger.error(f"销毁节点错误: {e}")
        
        try:
            rclpy.shutdown()
        except Exception as e:
            logger.error(f"关闭rclpy错误: {e}")

if __name__ == '__main__':
    logger.remove()
    logger.add(
        os.sys.stdout,
        format="<level>{level: <4}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    main()




