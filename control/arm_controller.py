#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-100机械臂控制器 - 最简配置版本"""

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
        self.get_logger().info(f"成功加载配置文件: {config_path}")
        
        # 创建发布器和订阅器
        self.joint_cmd_pub = self.create_publisher(JointState, self.cfg["ros"]["cmd_topic"], 10)
        self.joint_state_sub = self.create_subscription(
            JointState, self.cfg["ros"]["state_topic"], self._joint_state_callback, 10)
        
        # 状态变量
        self.current_all_joints = None
        self.current_arm_joints = None
        self.lock = threading.Lock()
        
        # 启动后台线程处理ROS消息
        self.spin_thread = threading.Thread(target=self._spin_thread, daemon=True)
        self.spin_thread.start()
        
        # 加载标定数据
        with open(self.cfg["calibration_file"], 'r', encoding='utf-8') as f:
            self.cal = json.load(f)
        self.get_logger().info(f"成功加载标定文件: {self.cfg['calibration_file']}")
        
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
        self._wait_for_connection()
        self.get_logger().info("机械臂控制器初始化完成")
    
    def _spin_thread(self):
        """后台线程处理ROS消息"""
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
    
    def _load_robot_model(self):
        """加载机器人URDF模型"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            urdf_path = os.path.join(script_dir, "..", self.cfg["urdf_file"])
            
            if os.path.exists(urdf_path):
                self.robot_model = ERobot.URDF(urdf_path)
                self.get_logger().info(f"成功加载URDF模型: {urdf_path}")
            else:
                self.robot_model = None
                self.get_logger().warning(f"URDF文件不存在: {urdf_path}")
        except Exception as e:
            self.robot_model = None
            self.get_logger().error(f"加载URDF模型失败: {e}")
    
    def _joint_state_callback(self, msg):
        """关节状态回调"""
        with self.lock:
            self.current_all_joints = list(msg.position)
            if len(self.current_all_joints) >= len(self.cfg["joint_indices"]):
                self.current_arm_joints = [self.current_all_joints[i] for i in self.cfg["joint_indices"]]
    
    def _wait_for_connection(self, timeout=10.0):
        """等待关节状态数据"""
        start_time = time.time()
        self.get_logger().info("等待关节状态数据...")
        
        while time.time() - start_time < timeout:
            if self.current_arm_joints is not None:
                self.get_logger().info("已接收到关节状态数据")
                return True
            time.sleep(0.1)
        
        self.get_logger().warning("等待关节状态数据超时")
        return False
    
    def _send_arm_command(self, arm_joints: List[float]) -> bool:
        """发送机械臂命令"""
        if len(arm_joints) != len(self.cfg["joint_indices"]):
            self.get_logger().error(f"关节数量不匹配: 期望{len(self.cfg['joint_indices'])}, 实际{len(arm_joints)}")
            return False
        
        # 获取当前夹爪位置
        with self.lock:
            if self.current_all_joints is None or len(self.current_all_joints) <= self.cfg["gripper_joint_index"]:
                self.get_logger().error("无法获取当前夹爪位置")
                return False
            gripper_pos = self.current_all_joints[self.cfg["gripper_joint_index"]]
        
        # 确保所有值都是float类型
        arm_joints_float = [float(j) for j in arm_joints]
        gripper_pos_float = float(gripper_pos)
        
        # 发送命令
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        msg.position = arm_joints_float + [gripper_pos_float]
        
        self.joint_cmd_pub.publish(msg)
        return True
    
    def _check_joint_limits(self, joints: List[float]) -> List[float]:
        """检查并限制关节范围"""
        return [max(min_limit, min(max_limit, joint_val)) 
                for joint_val, (min_limit, max_limit) in zip(joints, self.joints_limits)]
    
    def _wait_for_joints(self, target_joints: List[float], timeout: float, tolerance: float) -> bool:
        """等待到达目标位置"""
        if timeout <= 0:
            return True
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            current = self.get_joints()
            if current and len(current) == len(target_joints):
                if all(abs(c - t) < tolerance for c, t in zip(current, target_joints)):
                    return True
            time.sleep(0.01)
        return False
    
    # ==================== 基础接口 ====================
    
    def get_joints(self) -> Optional[List[float]]:
        """获取机械臂关节角度"""
        with self.lock:
            return self.current_arm_joints.copy() if self.current_arm_joints else None
    
    def set_joints(self, joints: List[float], timeout: float = None, tolerance: float = None) -> bool:
        """设置机械臂关节位置
        
        Args:
            joints: 目标关节位置 (5个关节)
            timeout: 超时时间，>0时等待到位，<=0时不等待，None使用默认值
            tolerance: 位置容差，None使用默认值
            
        Returns:
            成功返回True，失败或超时返回False
        """
        if len(joints) != len(self.cfg["joint_indices"]):
            self.get_logger().error(f"关节数量错误: 期望{len(self.cfg['joint_indices'])}, 实际{len(joints)}")
            return False
        
        # 使用默认值
        if timeout is None:
            timeout = self.cfg["default_timeout"]
        if tolerance is None:
            tolerance = self.cfg["position_tolerance"]
        
        # 限制关节范围并发送命令
        limited_joints = self._check_joint_limits(joints)
        success = self._send_arm_command(limited_joints)
        
        if not success:
            return False
        
        # 根据timeout决定是否等待
        return self._wait_for_joints(limited_joints, timeout, tolerance)
    
    # ==================== 预设位置 ====================
    
    def move_home(self, timeout: float = None) -> bool:
        """移动到home位置"""
        self.get_logger().info("移动到home位置")
        return self.set_joints(self.home_joints, timeout=timeout)
    
    def move_up(self, timeout: float = None) -> bool:
        """移动到up位置"""
        self.get_logger().info("移动到up位置")
        return self.set_joints(self.up_joints, timeout=timeout)
    
    def move_zero(self, timeout: float = None) -> bool:
        """移动到zero位置"""
        self.get_logger().info("移动到zero位置")
        return self.set_joints(self.zero_joints, timeout=timeout)
    
    # ==================== 运动学计算 ====================
    
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
        """计算正运动学"""
        if self.robot_model is None:
            self.get_logger().error("机器人模型未加载")
            return None
        
        end_link = "tcp" if use_tool else "Fixed_Jaw"
        
        try:
            fk_pose_matrix = self.robot_model.fkine(np.array(joint_angles), end=end_link)
            
            position = fk_pose_matrix.t
            quaternion_xyzw = Rotation.from_matrix(fk_pose_matrix.R).as_quat()
            
            pose_list = list(position) + list(quaternion_xyzw)
            return pose_list
            
        except Exception as e:
            self.get_logger().error(f"正运动学计算失败: {e}")
            return None
    
    def _inverse_kinematics(self, target_pose_list, initial_joint_guess=None, mask=None, use_tool=False):
        """计算逆运动学"""
        if self.robot_model is None:
            self.get_logger().error("机器人模型未加载")
            return None
        
        end_link = "tcp" if use_tool else "Fixed_Jaw"
        
        try:
            target_pose_matrix = self._create_pose_matrix(target_pose_list)
            
            if initial_joint_guess is None:
                current_joints = self.get_joints()
                if current_joints is None:
                    self.get_logger().error("无法获取当前关节角度")
                    return None
                q_guess = np.array(current_joints)
            else:
                q_guess = np.array(initial_joint_guess)
            
            if mask is None:
                mask = self.cfg["kinematics"]["default_mask"]
            
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
                    self.get_logger().debug(f"逆解成功: {format_to_2dp(q_solution.tolist())}")
                    return q_solution.tolist()
                else:
                    self.get_logger().warning(f"逆解失败，目标位姿: {target_pose_list}")
                    return None
            else:
                self.get_logger().error(f"ik_LM返回值格式异常: {result}")
                return None
                
        except Exception as e:
            self.get_logger().error(f"逆解计算失败: {e}")
            return None
    
    # ==================== 位姿控制接口 ====================
    
    def get_tcp_pose(self) -> Optional[List[float]]:
        """获取TCP位姿"""
        joints = self.get_joints()
        if joints is None:
            return None
        tcp_pose = self._forward_kinematics(joints, use_tool=True)
        return format_to_2dp(tcp_pose)
    
    def get_flange_pose(self) -> Optional[List[float]]:
        """获取法兰位姿"""
        joints = self.get_joints()
        if joints is None:
            return None
        flange_pose = self._forward_kinematics(joints, use_tool=False)
        return format_to_2dp(flange_pose)
    
    def tcp_move_to_pose(self, target_pose_list, initial_joint_guess=None, mask=None, 
                        timeout: float = None, tolerance: float = None):
        """TCP移动到目标位姿"""
        if timeout is None:
            timeout = self.cfg["default_timeout"]
        if tolerance is None:
            tolerance = self.cfg["position_tolerance"]
        
        if len(target_pose_list) == 3:
            current_tcp = self.get_tcp_pose()
            if current_tcp is None:
                self.get_logger().error("无法获取当前TCP姿态")
                return False
            
            full_target_pose = list(target_pose_list) + current_tcp[3:]
            self.get_logger().info(f"移动到位置: {format_to_2dp(target_pose_list)}, 保持当前姿态")
            
        elif len(target_pose_list) == 7:
            full_target_pose = list(target_pose_list)
            self.get_logger().debug(f"移动到位姿: 位置{format_to_2dp(target_pose_list[:3])}, 姿态{format_to_2dp(target_pose_list[3:])}")
            
        else:
            self.get_logger().error(f"目标位姿参数长度错误: {len(target_pose_list)}, 应为3或7")
            return False
        
        joint_solution = self._inverse_kinematics(full_target_pose, initial_joint_guess, mask, use_tool=True)
        
        if joint_solution is None:
            return False
        
        return self.set_joints(joint_solution, timeout, tolerance)
    
    def flange_move_to_pose(self, target_pose_list, initial_joint_guess=None, mask=None, 
                           timeout: float = None, tolerance: float = None):
        """法兰移动到目标位姿"""
        if timeout is None:
            timeout = self.cfg["default_timeout"]
        if tolerance is None:
            tolerance = self.cfg["position_tolerance"]
        
        if len(target_pose_list) == 3:
            current_flange = self.get_flange_pose()
            if current_flange is None:
                self.get_logger().error("无法获取当前法兰姿态")
                return False
            
            full_target_pose = list(target_pose_list) + current_flange[3:]
            self.get_logger().info(f"移动法兰到位置: {format_to_2dp(target_pose_list)}, 保持当前姿态")
            
        elif len(target_pose_list) == 7:
            full_target_pose = list(target_pose_list)
            self.get_logger().info(f"移动法兰到完整位姿: 位置{format_to_2dp(target_pose_list[:3])}, 姿态{format_to_2dp(target_pose_list[3:])}")
            
        else:
            self.get_logger().error(f"目标位姿参数长度错误: {len(target_pose_list)}, 应为3或7")
            return False
        
        joint_solution = self._inverse_kinematics(full_target_pose, initial_joint_guess, mask, use_tool=False)
        
        if joint_solution is None:
            return False
        
        return self.set_joints(joint_solution, timeout, tolerance)


def main():
    """测试函数"""
    rclpy.init()
    arm = None
    
    try:
        arm = Arm()
        print(f"当前关节: {arm.get_joints()}")
        
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
        #     arm.set_joints([joint, 0.0, 0.0, 0.0, 0.0], timeout=0)
        #     print(f"关节位置: {joint:.3f} {'↑' if going_up else '↓'}")
            
        #     if going_up:
        #         joint += step
        #         if joint >= joint_max:
        #             going_up = False
        #     else:
        #         joint -= step
        #         if joint <= joint_min:
        #             going_up = True

        # # 3. 测试笛卡尔运动
        # position = [-0.1, -0.2, 0.1]
        # arm.tcp_move_to_pose(position)
        # current_pose = arm.get_tcp_pose()
        # print(f"移动到位置: {current_pose}")
        # time.sleep(2)
        # position2 = [0.1, -0.2, 0.1]
        # arm.tcp_move_to_pose(position2)
        # current_pose = arm.get_tcp_pose()
        # print(f"移动到位置: {current_pose}")
        # time.sleep(2)


    except KeyboardInterrupt:
        print("用户中断")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if arm:
            arm.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
