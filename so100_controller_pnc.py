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
import pinocchio as pin
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
                # 使用Pinocchio加载URDF, 并指定固定基座
                self.model = pin.buildModelFromUrdf(urdf_file, root_joint=pin.JointModelFixed())
                self.data = self.model.createData()
                
                # 打印模型信息用于调试
                self._robot.get_logger().info(f"Pinocchio模型关节数 (nq): {self.model.nq}, (nv): {self.model.nv}")
                self._robot.get_logger().info(f"Pinocchio模型frame数: {self.model.nframes}")
                
                # 打印所有关节名称和ID (有助于调试)
                # for i, name in enumerate(self.model.names):
                #     if i > 0: # Skip universe
                #         self._robot.get_logger().info(f"Joint {i}: {name}, ID in q/v: {self.model.getJointId(name)}")

                # 打印所有frame名称
                # for frame in self.model.frames:
                #     self._robot.get_logger().info(f"Frame: {frame.name}, ID: {self.model.getFrameId(frame.name)}, Parent Joint ID: {frame.parentJoint}")

                # 设置TCP偏移
                self._setup_tcp_frame()
                
                self._robot.get_logger().info(f"机器人Pinocchio模型加载成功, TCP设置为: {self.offset_tcp_2_end}")
            else:
                self.model = None
                self.data = None
                self._robot.get_logger().warn(f"URDF文件不存在: {urdf_file}")
        except Exception as e:
            self.model = None
            self.data = None
            self._robot.get_logger().error(f"加载Pinocchio模型失败: {e}")
            import traceback
            self._robot.get_logger().error(traceback.format_exc())

    def _setup_tcp_frame(self):
        """设置TCP坐标系"""
        if self.model is None:
            return
            
        try:
            # The flange is on the last link of the arm. From your URDF, this is "Fixed_Jaw".
            flange_link_name = "Fixed_Jaw" 
            
            if self.model.existFrame(flange_link_name):
                 self.flange_frame_id = self.model.getFrameId(flange_link_name)
                 self._robot.get_logger().info(f"Flange frame (on link '{flange_link_name}') ID: {self.flange_frame_id}")
            else:
                self._robot.get_logger().error(f"Flange link frame '{flange_link_name}' not found in Pinocchio model.")
                # Fallback: try to guess the last operational frame if the specific name isn't found
                op_frames = [f for f in self.model.frames if f.type == pin.FrameType.OP_FRAME and f.parentJoint != 0]
                if op_frames:
                    last_op_frame = op_frames[-1]
                    self.flange_frame_id = self.model.getFrameId(last_op_frame.name)
                    self._robot.get_logger().warn(f"Using last operational frame '{last_op_frame.name}' (ID: {self.flange_frame_id}) as flange.")
                else: # Absolute fallback
                     self.flange_frame_id = self.model.nframes - 1 # Last frame in the model
                     self._robot.get_logger().warn(f"No operational frames found. Using last model frame (ID: {self.flange_frame_id}, Name: {self.model.frames[self.flange_frame_id].name}) as flange.")

            tcp_frame_name = "tcp_link" 
            
            if not self.model.existFrame(tcp_frame_name):
                # 1. Identify the joint to which the flange link ("Fixed_Jaw") is attached.
                #    This is the 'Wrist_Roll' joint.
                #    The frame for "Fixed_Jaw" (self.flange_frame_id) has a parentJoint.
                parent_joint_of_flange_link_id = self.model.frames[self.flange_frame_id].parentJoint
                if parent_joint_of_flange_link_id == 0 and self.model.frames[self.flange_frame_id].name == self.model.names[0]: # Check if flange is base_link itself on a fixed model
                     parent_joint_of_flange_link_id = self.model.getJointId(self.model.names[1]) if len(self.model.names) > 1 else 1 # A bit hacky for fixed base root

                self._robot.get_logger().info(f"Parent joint of flange link ('{self.model.frames[self.flange_frame_id].name}') is joint ID {parent_joint_of_flange_link_id} ('{self.model.names[parent_joint_of_flange_link_id]}').")

                # 2. Get the placement of the flange frame w.r.t. its parent joint.
                flange_frame_placement_wrt_its_parent_joint = self.model.frames[self.flange_frame_id].placement
                
                # 3. Define the TCP's placement relative to the flange frame.
                tcp_placement_on_flange_frame = pin.SE3(np.eye(3), np.array(self.offset_tcp_2_end))
                
                # 4. Calculate TCP's placement relative to the flange's parent joint.
                #    This is how the new TCP frame will be defined.
                tcp_placement_wrt_flange_parent_joint = flange_frame_placement_wrt_its_parent_joint * tcp_placement_on_flange_frame
                
                self.tcp_frame_id = self.model.addFrame(
                    pin.Frame(tcp_frame_name,
                              parent_joint_of_flange_link_id,      # Joint the TCP is rigidly attached to
                              tcp_placement_wrt_flange_parent_joint, # TCP's placement w.r.t this joint
                              pin.FrameType.OP_FRAME)
                )
                
                # After adding a frame, the model changes. Re-create data and update.
                self.data = self.model.createData() 
                pin.forwardKinematics(self.model, self.data, pin.neutral(self.model)) 
                pin.updateFramePlacements(self.model, self.data)
                self._robot.get_logger().info(f"TCP frame '{tcp_frame_name}' added with ID: {self.tcp_frame_id}. Total frames now: {self.model.nframes}")
            else:
                self.tcp_frame_id = self.model.getFrameId(tcp_frame_name)
                self._robot.get_logger().info(f"TCP frame '{tcp_frame_name}' already exists with ID: {self.tcp_frame_id}")

        except Exception as e:
            self._robot.get_logger().error(f"设置TCP frame失败: {e}")
            import traceback
            self._robot.get_logger().error(traceback.format_exc())
            # Fallback: if TCP setup fails, use flange as TCP.
            if hasattr(self, 'flange_frame_id'):
                self.tcp_frame_id = self.flange_frame_id
            else: 
                 self.flange_frame_id = self.model.nframes -1 
                 self.tcp_frame_id = self.flange_frame_id
            self._robot.get_logger().warn(f"TCP frame setup failed, using flange frame (ID: {self.tcp_frame_id}) as TCP.")

    def _forward_kinematics(self, joint_angles, use_tool=True):
        """
        使用Pinocchio计算正运动学
        """
        if self.model is None or self.data is None:
            self._robot.get_logger().error("Pinocchio模型未加载，无法进行正运动学计算")
            return None
            
        try:
            q = np.array(joint_angles, dtype=float)
            
            if len(q) != self.model.nq:
                 # This case should ideally not happen if nq is correctly 5 for a 5-DOF arm
                self._robot.get_logger().warn(f"FK: Mismatch in joint_angles length ({len(q)}) and model.nq ({self.model.nq}). Padding/truncating.")
                q_full = pin.neutral(self.model)
                n_min = min(len(q), self.model.nq)
                q_full[:n_min] = q[:n_min]
                q = q_full
            
            pin.forwardKinematics(self.model, self.data, q)
            # Explicitly update frame placements AFTER forward kinematics for all frames
            pin.updateFramePlacements(self.model, self.data)
            
            target_frame_id_to_use = -1
            if use_tool and hasattr(self, 'tcp_frame_id'):
                target_frame_id_to_use = self.tcp_frame_id
            elif hasattr(self, 'flange_frame_id'): # Default to flange if not use_tool or tcp_frame_id missing
                target_frame_id_to_use = self.flange_frame_id
            else:
                 self._robot.get_logger().error("Flange or TCP frame ID not set for FK.")
                 return None

            if not (0 <= target_frame_id_to_use < len(self.data.oMf)):
                self._robot.get_logger().error(f"Target frame ID {target_frame_id_to_use} is out of bounds for oMf (size {len(self.data.oMf)}). Ensure _setup_tcp_frame ran correctly and model.data is current.")
                # This can happen if addFrame changed model.nframes but data wasn't fully updated for oMf size.
                # Forcing an update one more time, though this indicates an issue in setup.
                self.data = self.model.createData() # Recreate data just in case
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacements(self.model, self.data)
                if not (0 <= target_frame_id_to_use < len(self.data.oMf)):
                     self._robot.get_logger().error(f"Still out of bounds. Frame ID: {target_frame_id_to_use}. Model nframes: {self.model.nframes}")
                     return None
            
            pose_se3 = self.data.oMf[target_frame_id_to_use]
            position = pose_se3.translation
            rotation_matrix = pose_se3.rotation

            # Check for invalid rotation matrix
            if np.linalg.det(rotation_matrix) < 1e-6 : # Check if determinant is close to zero or negative
                self._robot.get_logger().warn(f"Invalid rotation matrix for frame ID {target_frame_id_to_use} (det={np.linalg.det(rotation_matrix)}). Matrix:\n{rotation_matrix}\nUsing identity rotation as fallback.")
                # This might happen at singular configurations or if q is all zeros and it's a singularity.
                # For a basic FK test, if q=[0,0,0,0,0], the pose should be the home/zero pose.
                # If it results in zero matrix, something is still off in model or q interpretation.
                # Fallback to identity rotation if problematic, though this hides issues.
                # A better approach for singularity might be to return the problematic matrix and let caller handle.
                # For now, to avoid crash in Rotation.from_matrix:
                # If it's truly all zeros, any SE3 operation would be problematic.
                # quaternion_xyzw = np.array([0.,0.,0.,1.]) # Default to identity quaternion
                # Check if it's all zeros before trying from_matrix
                if np.allclose(rotation_matrix, 0):
                    self._robot.get_logger().error("Rotation matrix is all zeros. Cannot compute pose.")
                    return None # Or raise an error
                # If not all zero but bad determinant, try to proceed but warn
                # Scipy's Rotation.from_matrix will raise ValueError for non-positive determinant.

            quaternion_xyzw = Rotation.from_matrix(rotation_matrix).as_quat() # [x,y,z,w]
            pose_list = list(position) + list(quaternion_xyzw)
            return pose_list
            
        except ValueError as ve: # Catch specific error from Rotation.from_matrix
            self._robot.get_logger().error(f"Scipy ValueError during FK (likely bad rotation matrix): {ve}")
            self._robot.get_logger().error(f"Problematic Rotation Matrix for frame {target_frame_id_to_use}:\n{self.data.oMf[target_frame_id_to_use].rotation if target_frame_id_to_use < len(self.data.oMf) else 'Error accessing oMf'}")
            import traceback
            self._robot.get_logger().error(traceback.format_exc())
            return None
        except Exception as e:
            self._robot.get_logger().error(f"Pinocchio正运动学计算出错: {e}")
            import traceback
            self._robot.get_logger().error(traceback.format_exc())
            return None

    def _inverse_kinematics(self, target_pose_list, initial_joint_guess=None, mask=None):
        """
        计算逆运动学解 (目标位姿为TCP位姿)
        """
        if self.model is None or self.data is None:
            self._robot.get_logger().error("Pinocchio模型未加载，无法进行逆运动学计算")
            return None
        if not hasattr(self, 'tcp_frame_id'): # Ensure TCP frame is defined
            self._robot.get_logger().error("TCP frame ID not set for IK. Cannot perform IK.")
            return None
        if target_pose_list is None: # Check if the input pose is None
            self._robot.get_logger().error("IK: Target pose list is None. Cannot perform IK.")
            return None

        try:
            pos = np.array(target_pose_list[:3])
            quat_xyzw = np.array(target_pose_list[3:]) 
            
            target_rotation = Rotation.from_quat(quat_xyzw).as_matrix()
            oMdes = pin.SE3(target_rotation, pos) 

            eps = 1e-4 
            IT_MAX = 1000 
            DT = 1e-1  # Step size for integration
            damp = 1e-6 # Damping for pseudo-inverse J.T * J + damp * I

            # Initial guess for joint configuration
            q = pin.neutral(self.model) # Initialize q for the full model.nq
            if initial_joint_guess is None:
                current_joints_arm = self.get_joints() 
                if current_joints_arm is None:
                    self._robot.get_logger().warn("IK: Could not get current joints, using model neutral for initial guess.")
                    # q is already pin.neutral(self.model)
                else:
                    # Fill in the actuated joint values into the full q vector
                    # This assumes self.joint_index correctly maps to the model's q indices
                    # and that model.nq matches len(current_joints_arm) if base is fixed.
                    if self.model.nq == len(current_joints_arm):
                        q[:] = np.array(current_joints_arm)
                    else: # Mismatch, fill what we can
                         self._robot.get_logger().warn(f"IK: Mismatch nq ({self.model.nq}) and arm joints ({len(current_joints_arm)}). Using partial fill.")
                         n_min = min(self.model.nq, len(current_joints_arm))
                         q[:n_min] = np.array(current_joints_arm)[:n_min]
            else:
                if self.model.nq == len(initial_joint_guess):
                    q[:] = np.array(initial_joint_guess)
                else:
                    self._robot.get_logger().warn(f"IK: Mismatch nq ({self.model.nq}) and initial_joint_guess ({len(initial_joint_guess)}). Using partial fill.")
                    n_min = min(self.model.nq, len(initial_joint_guess))
                    q[:n_min] = np.array(initial_joint_guess)[:n_min]

            self._robot.get_logger().info(f"IK: Initial guess q (model nq={self.model.nq}): {format_to_2dp(q.tolist())}")
            
            target_frame_for_ik = self.tcp_frame_id

            for i in range(IT_MAX):
                pin.forwardKinematics(self.model, self.data, q)
                # pin.updateFramePlacements(self.model, self.data) # Not strictly needed if using data.oMf[frame_id] after FK
                
                oMcurr = self.model.framePlacement(self.data, target_frame_for_ik) # More robust way to get frame placement
                
                err = pin.log6(oMdes.actInv(oMcurr)).vector # err = log( M_desired^-1 * M_current )
                
                if np.linalg.norm(err) < eps:
                    self._robot.get_logger().info(f"IK converged in {i+1} iterations.")
                    return q[:len(self.joint_index)].tolist() # Return actuated joint values

                # Jacobian is 6xnv. For a fixed base 5-DOF arm, nv should be 5.
                J = pin.computeFrameJacobian(self.model, self.data, q, target_frame_for_ik)
                
                # Standard Damped Least Squares: dq = J.T * inv(J * J.T + damp * I) * err
                # Or: (J.T * J + damp * I) * dq = J.T * err (if J is nv x N_task_dim, solving for nv velocities)
                # Pinocchio's J is N_task_dim x nv (6 x nv)
                # So we solve J * dq = -err (or err, depending on error def)
                
                # Solves J dq = -err with damping for singularity robustness
                # dq = -np.linalg.solve(J.T @ J + damp * np.eye(self.model.nv), J.T @ err) 
                # Simpler:
                dq = -np.linalg.lstsq(J, err, rcond=None)[0] # Solves J dq = err, so use -err or err based on err definition

                q = pin.integrate(self.model, q, DT * dq) 
                
                # Enforce joint limits (for the actuated joints, assuming q corresponds to model.nq)
                # This needs careful mapping if model.nq > num_actuated_joints
                # If model.nq == num_actuated_joints (e.g. 5), then self.joint_index might be just [0,1,2,3,4]
                for j_model_idx in range(min(self.model.nq, len(self.joints_limits))):
                     # Assuming self.joints_limits corresponds to the first nq joints
                     # and self.joint_index is not needed here if q is already the correct subset
                    q[j_model_idx] = np.clip(q[j_model_idx], self.joints_limits[j_model_idx][0], self.joints_limits[j_model_idx][1])
            
            self._robot.get_logger().warn(f"IK failed to converge after {IT_MAX} iterations. Final error norm: {np.linalg.norm(err)}")
            return None
        except TypeError as te: # Catch specific error from target_pose_list if it's None
            self._robot.get_logger().error(f"IK TypeError (likely target_pose_list is None): {te}")
            import traceback
            self._robot.get_logger().error(traceback.format_exc())
            return None
        except Exception as e:
            self._robot.get_logger().error(f"Pinocchio逆解计算出错: {e}")
            import traceback
            self._robot.get_logger().error(traceback.format_exc())
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
