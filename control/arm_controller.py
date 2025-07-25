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
    
    def _send_arm_command(self, arm_joints: List[float]) -> bool:
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
    
    def _wait_for_joints(self, target_joints: List[float], timeout: float = 100, tolerance: float = 0.04) -> bool:
        """等待到达目标位置"""
        if timeout <= 0:
            return True
        
        start_time = time.time()

        # 等待关节到达目标位置
        while time.time() - start_time < timeout:
            current = self.get_joints()
            if current and len(current) == len(target_joints):
                if all(abs(c - t) < tolerance for c, t in zip(current, target_joints)):
                    return True
            time.sleep(0.01)
        
        # 超时，打印当前关节和目标关节
        current = self.get_joints()
        joint_diff = [current[i] - target_joints[i] for i in range(len(target_joints))]
        logger.error(f"timeout({timeout}), current_joints({current}), target_joints({target_joints}), joint_diff({joint_diff})")
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
    

    def cacul_valid_ik_joints_from_pose(self, target_pose_list, use_tool=True):
        # 1. 使用当前关节角度逆解
        current_joints = self.get_joints()
        if current_joints is None:
            logger.error("get_joints failed !")
            return None

        mask = self.cfg["kinematics"]["default_mask"]
        joint_solution = self._inverse_kinematics(target_pose_list, current_joints, mask, use_tool=use_tool)
        if joint_solution is None:
            logger.error(f"_inverse_kinematics failed !")
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

        if not self._send_arm_command(joints):
            logger.error("send_arm_command failed !")
            return False
        
        if not self._wait_for_joints(joints, timeout, tolerance):
            logger.error("wait for joints failed !")
            return False

        return True
    

    def move_to_pose(self, target_pose_list, timeout: float = 100, tolerance: float = 0.04, tcp: bool = True) -> bool:
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
            
            full_target_pose = list(target_pose_list) + current_pose[3:]

        elif len(target_pose_list) == 6:
            # 将欧拉角转换为四元数
            position = target_pose_list[:3]
            euler_angles = target_pose_list[3:]  # [roll, pitch, yaw]
            
            # 转换为四元数
            rotation = Rotation.from_euler('xyz', euler_angles)
            quaternion = rotation.as_quat()  # [qx, qy, qz, qw]
            
            full_target_pose = list(position) + list(quaternion)

        else:
            logger.error(f"target pose length({len(target_pose_list)}) != 3 or 6 !")
            return False
        
        full_target_pose = self._tuning_rotation_reachable(full_target_pose)
        ik_joints = self.cacul_valid_ik_joints_from_pose(full_target_pose, use_tool=tcp)
        if ik_joints is None:
            logger.error(f"cacul_valid_ik_joints_from_pose failed !")
            return False

        if not self.move_to_joints(ik_joints, timeout, tolerance):
            logger.error("move_to_joints failed !")
            return False

        return True

    def move_line(self, start_position, end_position, step=0.01, timeout=100, tcp: bool = True) -> bool:
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
        
        # 依次运动到每个轨迹点
        for i, point in enumerate(trajectory_points):
            success = self.move_to_pose(point, timeout=timeout, tcp=tcp)
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
    
    def move_up(self, timeout: Optional[float] = None) -> bool:
        """移动到up位置"""
        logger.info("移动到up位置")
        if timeout is None:
            timeout = 100
        return self.move_to_joints(self.up_joints, timeout=timeout)
    
    def move_zero(self, timeout: Optional[float] = None) -> bool:
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


def main():
    """测试函数"""
    rclpy.init()
    arm = None
    
    try:
        arm = Arm()
        # print(f"当前关节: {arm.get_joints()}")
        # print(f"当前TCP位姿: {arm.get_pose()}")
        
        # # 1. 测试实时获取关节角度
        # while True:
        #     joints = arm.get_joints()
        #     if joints is not None:
        #         print(f"joints: ", joints)
        #     time.sleep(0.01)  # 每秒打印一次

        # #2. 测试高频控制某一关节
        # joint_max = 1
        # joint_min = -1
        # step = 0.001
        # going_up = True
        # joint = joint_min
        # while True:
        #     # 只传递5个关节值，不是6个
        #     arm.move_to_joints([joint, 0.0, 0.0, 0.0, 0.0], timeout=0)
        #     print(f"关节位置: {joint:.3f} {'↑' if going_up else '↓'}")
            
        #     if going_up:
        #         joint += step
        #         if joint >= joint_max:
        #             going_up = False
        #     else:
        #         joint -= step
        #         if joint <= joint_min:
        #             going_up = True
        #     time.sleep(0.01)

        # # 3. 测试笛卡尔运动
        # position = [-0.1, -0.2, 0.1]
        # arm.move_to_pose(position)

        # position = [0.1, -0.25, 0.1]
        # arm.move_to_pose(position)

        # # 4. 矩形运动
        # time_to_sleep = 0
        # pose1 = [-0.1, -0.2, 0.05, 0, 0, 0]
        # pose2 = [0.1, -0.2, 0.05, 0, 0, 0]
        # pose3 = [0.1, -0.2, 0.15, 0, 0, 0]
        # pose4 = [-0.1, -0.2, 0.11, 0, 0, 0]

        # print(f"\nmoving to pose1..........")
        # arm.move_to_pose(pose1, timeout=100)
        # print(f"pose1_target:  {pose1}")
        # print(f"pose1_actual: {arm.get_pose()}")
        # current_joints = arm.get_joints()
        # print(f"target_joints: {TARGET_JOINTS}")
        # print(f"current_joints: {current_joints}")
        # joint_diff = [TARGET_JOINTS[i] - current_joints[i] for i in range(5)]
        # print(f"joint_diff: {joint_diff}")
        # print(f"sleep {time_to_sleep} seconds..........")
        # time.sleep(time_to_sleep)
        # input("press enter to continue..........")

        # print(f"\nmoving to pose2..........")
        # arm.move_to_pose(pose2, timeout=100)
        # print(f"pose2_target:  {pose2}")
        # print(f"pose2_actual: {arm.get_pose()}")
        # current_joints = arm.get_joints()
        # joint_diff = [TARGET_JOINTS[i] - current_joints[i] for i in range(5)]
        # print(f"target_joints: {TARGET_JOINTS}")
        # print(f"current_joints: {current_joints}")
        # print(f"joint_diff: {joint_diff}")
        # print(f"sleep {time_to_sleep} seconds..........")
        # time.sleep(time_to_sleep)
        # input("press enter to continue..........")

        # print(f"\nmoving to pose3..........")
        # arm.move_to_pose(pose3, timeout=100)
        # print(f"pose3_target:  {pose3}")
        # print(f"pose3_actual: {arm.get_pose()}")
        # current_joints = arm.get_joints()
        # joint_diff = [TARGET_JOINTS[i] - current_joints[i] for i in range(5)]
        # print(f"target_joints: {TARGET_JOINTS}")
        # print(f"current_joints: {current_joints}")
        # print(f"joint_diff: {joint_diff}")
        # print(f"sleep {time_to_sleep} seconds..........")
        # time.sleep(time_to_sleep)
        # input("press enter to continue..........")  

        # print(f"\nmoving to pose4..........")
        # arm.move_to_pose(pose4, timeout=100)
        # print(f"pose4_target:  {pose4}")
        # print(f"pose4_actual: {arm.get_pose()}")
        # current_joints = arm.get_joints()
        # joint_diff = [TARGET_JOINTS[i] - current_joints[i] for i in range(5)]
        # print(f"target_joints: {TARGET_JOINTS}")
        # print(f"current_joints: {current_joints}")
        # print(f"joint_diff: {joint_diff}")
        # print(f"sleep {time_to_sleep} seconds..........")
        # time.sleep(time_to_sleep)   
        # input("press enter to continue..........")

        # # 直线运动
        # pose1 = [-0.1, -0.2, 0.05, 0, 0, 0, 1]
        # pose2 = [0.1, -0.2, 0.05, 0, 0, 0, 1]
        # pose3 = [0.1, -0.2, 0.11, 0, 0, 0, 1]
        # pose4 = [-0.1, -0.2, 0.11, 0, 0, 0, 1]
        # arm.move_line(pose1, pose2)
        # input("press enter to continue..........")
        # arm.move_line(pose2, pose3)
        # input("press enter to continue..........")
        # arm.move_line(pose3, pose4)
        # input("press enter to continue..........")
        # arm.move_line(pose4, pose1)

        # # 测试关节控制精度
        # # target_joints = [0.0,0.0,0.0,0.0,0.0]
        # target_joints = [0.5701169177893028, -0.12349406347192682, 0.025298338928624098, 0.1070100633657809, -0.5715415433504774]
        # arm.move_to_joints(target_joints, timeout=10, tolerance=0.01)
        # current_joints = arm.get_joints()
        # diff = [target_joints[i] - current_joints[i] for i in range(5)]
        # print(f"target_joints: {target_joints}")
        # print(f"current_joints: {current_joints}")
        # print(f"diff: {diff}")


        # [0.47191824960873996, 0.6268051355951858, -0.6912450679514204, 0.18219341869152395, -0.4411214371731673]
        # [0.47191824960873996, 0.6268051355951858, -0.6912450679514204, 0.18219341869152395, -0.4411214371731673]
        # [0.4703838954184185, 0.6360112607371136, -0.6881763595707775, 0.18219341869152395, -0.4411214371731673]
        # [0.4703838954184185, 0.6436830316887203, -0.6851076511901351, 0.18219341869152395, -0.4411214371731673]
        # [0.4703838954184185, 0.6513548026403266, -0.6820389428094926, 0.18219341869152395, -0.4395870829828459]
        # [0.4703838954184185, 0.6513548026403266, -0.6820389428094926, 0.18219341869152395, -0.4395870829828459]

        # current_joints = arm.get_joints()
        # # current_joints = [0.4703838954184185, 0.6513548026403266, -0.6820389428094926, 0.18219341869152395, -0.4395870829828459]
        # #                  [0.4703838954184185, 0.659026573591933, -0.6789702344288497, 0.18219341869152395, -0.4395870829828459]
        # # print(f"current_joints: {current_joints}")
        # arm.move_to_joints(current_joints, timeout=10, tolerance=0.01)
        # current_joints = arm.get_joints()
        # print(f"current_joints: {current_joints}")


        # while True:
        #     current_joints = arm.get_joints()
        #     print(f"current_joints: {current_joints}")
        #     time.sleep(0.1)

        # while True:
        #     current_joints = arm.get_joints()
        #     arm.move_to_joints(current_joints, timeout=10, tolerance=0.01)
        #     print(current_joints)
        #     time.sleep(0.1)

        zero_joints = [0.0, 0.0, 0.0, 0.0, 0.0]
        pose1 = [0.2, -0.2, 0.3, np.pi/6, np.pi/6, 0.0]
        # print(f"moving to zero_joints..........")
        # arm.move_to_joints(zero_joints, timeout=10, tolerance=0.04)

        # input(f"press enter moving to pose1:{pose1}")
        arm.move_to_pose(pose1, timeout=10, tolerance=0.04)
        current_pose = arm.get_pose()
        print(f"current_pose: {current_pose}")

    except KeyboardInterrupt:
        logger.info("user interrupt")
    except Exception as e:
        logger.error(f"error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 按正确顺序清理资源
        if arm:
            try:
                arm.destroy_node()
            except Exception as e:
                logger.error(f"Error destroying node: {e}")
        
        try:
            rclpy.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down rclpy: {e}")


if __name__ == '__main__':
    logger.remove()
    logger.add(
        os.sys.stdout,
        format="<level>{level: <4}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    main()




