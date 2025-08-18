#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-100机械臂控制器 - 直接硬件控制版本"""

import math
import json
import time
import os
import numpy as np
from typing import List, Optional
from scipy.spatial.transform import Rotation
from roboticstoolbox import ERobot
from loguru import logger
from .so100_driver import So100Driver

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

class Arm:
    """机械臂控制类"""
    
    def __init__(self, config_file=None):
        # 直接加载配置文件
        config_path = config_file or os.path.join(os.path.dirname(__file__), "config", "arm_config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)
        logger.info(f"成功加载配置文件: {config_path}")

        # 初始化硬件驱动
        self.so100_driver = So100Driver()
        if not hasattr(self.so100_driver, 'initialized') or not self.so100_driver.initialized:
            logger.error("硬件驱动初始化失败")
            raise RuntimeError("Failed to initialize hardware driver")
        
        # 状态变量
        self.current_arm_joints = None
        
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
        
        logger.info("Arm __init__ success")
    
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
    
    def _arm_write_joints(self, arm_joints: List[float]) -> bool:
        """发送机械臂命令"""
        if len(arm_joints) != 5:
            logger.error(f"send arm command failed: actual joints length({len(arm_joints)}) != expected joints length(5)")
            return False
        
        # 确保所有值都是float类型
        arm_joints_float = [float(j) for j in arm_joints]
        so100_target_joints = arm_joints_float + [None]

        try:
            # 直接调用硬件驱动
            success = self.so100_driver.write_joints(so100_target_joints)
            if not success:
                logger.error("hardware write_joints failed")
                return False
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
        # 上一次误差记录：每个关节保存上一次的误差值
        last_errors = [float('inf')] * len(target_joints)
        adjusted_targets = target_joints.copy()

        current = self.get_joints()
        if current is None:
            logger.error("get_joints failed in _wait_for_joints")
            return False
            
        last_errors = [abs(current[i] - target_joints[i]) for i in range(len(current))]
        error_direction = 1

        # 等待关节到达目标位置
        while time.time() - start_time < timeout:
            current = self.get_joints()
            if current is None:
                logger.error("get_joints failed during waiting")
                return False

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
                    if abs(current_errors[i]) >= abs(last_errors[i]):
                        print("joint i: ", i, " stop moving !")
                        # 误差没有减小，调整目标值
                        error_direction = 1 if current_errors[i] > 0 else -1
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
                self._arm_write_joints(adjusted_targets)

            time.sleep(0.1)
        
        # 超时，打印当前关节和目标关节
        current = self.get_joints()
        if current:
            joint_diff = [current[i] - target_joints[i] for i in range(len(target_joints))]
            for i, diff in enumerate(joint_diff):
                if abs(diff) > tolerance:
                    logger.error(f"timeout({timeout}), current_joints({current}), target_joints({target_joints}), errors({joint_diff})")
        
        return False
    
    def _create_pose_matrix(self, pose_components):
        """将位姿列表转换为4x4齐次变换矩阵
        
        Args:
            pose_components: [x, y, z, roll, pitch, yaw] 欧拉角格式
        """
        if len(pose_components) != 6:
            logger.error(f"pose_components length({len(pose_components)}) != 6")
            return None

        position = pose_components[:3]
        euler_angles = pose_components[3:]  # [roll, pitch, yaw]
        
        rotation_matrix = Rotation.from_euler('xyz', euler_angles).as_matrix()
        
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = position
        return pose_matrix
    
    def _forward_kinematics(self, joint_angles, use_tool=True):
        """计算正运动学
        
        Returns:
            位姿列表 [x, y, z, roll, pitch, yaw] 欧拉角格式
        """
        if self.robot_model is None:
            logger.error("机器人模型未加载")
            return None
        
        end_link = "tcp" if use_tool else "Fixed_Jaw"
        
        try:
            fk_pose_matrix = self.robot_model.fkine(np.array(joint_angles), end=end_link)
            
            position = fk_pose_matrix.t
            rotation_matrix = fk_pose_matrix.R
            euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('xyz')
            
            pose_list = list(position) + list(euler_angles)
            return pose_list
            
        except Exception as e:
            logger.error(f"正运动学计算失败: {e}")
            return None
    
    def _inverse_kinematics(self, target_pose_list, ik_init_pose, mask, use_tool=True):
        """计算逆运动学
        
        Args:
            target_pose_list: [x, y, z, roll, pitch, yaw] 欧拉角格式
        """
        if self.robot_model is None:
            logger.error("机器人模型未加载")
            return None
        
        end_link = "tcp" if use_tool else "Fixed_Jaw"
        
        try:
            target_pose_matrix = self._create_pose_matrix(target_pose_list)
            if target_pose_matrix is None:
                logger.error(f"create pose_matrix from target_pose_list failed !")
                return None
            
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
        """计算逆运动学关节解
        
        Args:
            target_pose_list: [x, y, z, roll, pitch, yaw] 欧拉角格式
        """
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
        try:
            joints = self.so100_driver.read_joints()
            if joints and len(joints) >= 5:
                return joints[:5]  # 只返回前5个关节
            else:
                logger.error("get_joints failed: invalid joints data")
                return None
        except Exception as e:
            logger.error(f"get_joints failed: {e}")
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
        
        return pose

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

        if len(joints) != 5:
            logger.error(f"actual joints length({len(joints)}) != expected joints length(5) !")
            return False
        
        # 检查关节范围
        if self._out_joint_limits(joints):
            logger.error("joints out of limits !")
            return False

        if not self._arm_write_joints(joints):
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
            target_position = target_pose_list[:3]
            target_rotation_euler = target_pose_list[3:]  # [roll, pitch, yaw]
            
        else:
            logger.error(f"target pose length({len(target_pose_list)}) != 3 or 6 !")
            return False
        
        # 组合完整的6元素位姿
        target_pose_euler = list(target_position) + list(target_rotation_euler)
        target_pose_euler_reachable = self._tuning_rotation_reachable(target_pose_euler)

        ik_joints = self.cacul_ik_joints_within_limits_from_pose(target_pose_euler_reachable, use_tool=tcp)
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
        """安全销毁"""
        if hasattr(self, 'so100_driver'):
            self.so100_driver.stop()

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

        diff = abs(theta_position - theta_rotation)
        if diff > 0.01:
            logger.error(f"rotation not reachable: theta_position({theta_position}), theta_rotation({theta_rotation}), diff({diff})")
            return False
        
        return True

    def _tuning_rotation_reachable(self, pose):
        """调整位姿使旋转可达
        
        Args:
            pose: [x, y, z, roll, pitch, yaw] 欧拉角格式
        
        Returns:
            调整后的位姿或原位姿
        """
        print(f"start _tuning_rotation_reachable")
        if len(pose) != 6:
            logger.warning(f"pose length({len(pose)}) != 6, skip rotation tuning")
            return pose
        
        try:
            position = pose[:3]
            rotation = pose[3:6]
            
            print(f"pose: {pose}")
            
            # 检查是否需要调整
            if self._is_rotation_reachable(pose):
                print("is_rotation_reachable: True")
                return pose  # 已经可达，不需要调整
            
            # 计算调整角度
            theta_position = self._get_theta_from_position(position)
            theta_rotation = self._get_theta_from_rotation(rotation)
            theta_diff = theta_position - theta_rotation
            print(f"theta_diff: {theta_diff}")
            print(f"theta_position: {theta_position}")
            print(f"theta_rotation: {theta_rotation}")

            # 调整旋转
            new_rotation = self._roll_rotation(rotation, theta_diff)
            print(f"new_rotation: {new_rotation}")

            new_pose = list(position) + new_rotation
            print(f"new_pose: {new_pose}")
            
            # 验证调整结果
            if self._is_rotation_reachable(new_pose):
                logger.info(f"Rotation tuned: {np.degrees(theta_diff):.1f}°")
                return new_pose
            else:
                logger.warning("Rotation tuning failed, using original pose")
                return pose
                
        except Exception as e:
            logger.error(f"Rotation tuning error: {e}")
            return pose

    def move_to_direction_abs(self, vector, timeout: float = 100, tolerance: float = 0.004, tcp: bool = True) -> bool:
        """沿绝对方向移动"""
        current_pose = self.get_pose(tcp=tcp)
        if current_pose is None:
            logger.error("get_pose failed!")
            return False
        
        # 计算目标位置 = 当前位置 + 移动向量
        target_position = [current_pose[i] + vector[i] for i in range(3)]
        
        return self.move_to_pose(target_position, timeout, tolerance, tcp)
    
    def move_to_direction_relative(self, vector, timeout: float = 100, tolerance: float = 0.004, tcp: bool = True) -> bool:
        """沿相对方向移动（相对于当前姿态）"""
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
    arm = None
    
    try:
        arm = Arm()
        # print(f"机械臂初始化成功")
        # print(f"当前关节: {arm.get_joints()}")
        # print(f"当前TCP位姿: {arm.get_pose()}")
        
        # # 保持运行状态
        # print("机械臂就绪，按 Ctrl+C 退出")
        # while True:
        #     time.sleep(1.0)


# pose: [-0.1, -0.3, 0.2, 1.7640112210628955, 0.08231736228386288, 0.10782470670965782]
# ERROR | _is_rotation_reachable:569 - rotation not reachable: theta_position(4.3906384259880475), theta_rotation(2.0764620132341225), diff(2.314176412753925)
# theta_diff: 2.314176412753925
# theta_position: 4.3906384259880475
# theta_rotation: 2.0764620132341225
# new_rotation: [1.7640112210628953, 0.0823173622838631, 2.422001119463583]
# new_pose: [-0.1, -0.3, 0.2, 1.7640112210628953, 0.0823173622838631, 2.422001119463583]


        pose = [-0.1, -0.3, 0.2, 1.7640112210628955, 0.08231736228386288, 0.10782470670965782]
        flag = arm._is_rotation_reachable(pose)
        print(f"flag: {flag}")
        pose2 = arm._tuning_rotation_reachable(pose)
        flag2 = arm._is_rotation_reachable(pose2)
        print(f"flag2: {flag2}")
        print(f"pose2: {pose2}")

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
                logger.error(f"销毁错误: {e}")

if __name__ == '__main__':
    logger.remove()
    logger.add(
        os.sys.stdout,
        format="<level>{level: <4}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    main()




