#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SO-100 机器人仿真启动脚本
使用NVIDIA Isaac Sim平台进行5自由度SO-100机器人的仿真。
增加了ROS功能，发布机器人状态。
"""

import argparse
import numpy as np
import os
import time
from isaacsim import SimulationApp
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import TransformStamped, Pose
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Float64MultiArray
import cv_bridge

class SO100Simulation:
    """SO-100机器人仿真环境管理类"""
    
    def __init__(self, args):
        """初始化仿真环境
        
        Args:
            args: 命令行参数对象
        """
        self.args = args
        self.robot = None
        self.world = None
        self.simulation_app = SimulationApp({"headless": args.headless})
        
        # ROS相关初始化
        self.ros_initialized = False
        self.ros_node = None
        self.joint_state_publisher = None
        self.transform_broadcaster = None
        self.joint_command_subscriber = None
        self.publish_rate = 10  # Hz
        self.last_publish_time = 0
        
        # 平滑运动相关参数
        self.target_joint_positions = None
        self.is_moving = False
        self.motion_duration = 1.0  # 运动持续时间(秒)
        self.motion_start_time = 0.0
        self.motion_start_positions = None
        
        # 摄像头相关
        self.camera = None
        self.camera_publisher = None
        self.bridge = None
        self.camera_joint_index = 3  # 第四个关节索引（从0开始计数）
        self.camera_init_delay = 20  # 摄像头初始化延迟帧数
        self.camera_init_counter = 0  # 摄像头初始化计数器
        
        # 错误报告跟踪
        self._reported_errors = set()
        
        if args.ros:
            self._init_ros()
        
    def _log_once(self, error_key, message):
        """只记录一次指定类型的错误或信息
        
        Args:
            error_key: 错误类型的唯一标识符
            message: 要记录的消息
        """
        if error_key not in self._reported_errors:
            print(message)
            self._reported_errors.add(error_key)
        
    def _init_ros(self):
        """初始化ROS节点和发布者"""
        if not rclpy.ok():
            rclpy.init()
        
        self.ros_node = Node('so100_sim')
        self.joint_state_publisher = self.ros_node.create_publisher(
            JointState, 
            'so100_joint_states', 
            10
        )
        self.transform_broadcaster = TransformBroadcaster(self.ros_node)
        
        # 添加关节控制订阅器
        self.joint_command_subscriber = self.ros_node.create_subscription(
            JointState,
            'so100_joint_commands',
            self._joint_command_callback,
            10
        )
        
        # 添加简单命令订阅器（使用Float64MultiArray作为替代）
        self.simple_command_subscriber = self.ros_node.create_subscription(
            Float64MultiArray,
            'so100_position_commands',
            self._simple_command_callback,
            10
        )
        
        # 添加摄像头图像发布器
        self.camera_publisher = self.ros_node.create_publisher(
            Image,
            'so100_camera/image_raw',
            10
        )
        self.bridge = cv_bridge.CvBridge()
        
        self.ros_initialized = True
        print("ROS node initialized")
        print("控制话题已创建: /so100_joint_commands (JointState)")
        print("简单控制话题已创建: /so100_position_commands (Float64MultiArray)")
        print("摄像头话题已创建: /so100_camera/image_raw (Image)")
        
        # Process ROS callbacks initially to register topics
        rclpy.spin_once(self.ros_node, timeout_sec=0.1)
        
    def setup_world(self):
        """设置仿真世界"""
        from omni.isaac.core import World
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
    
    def load_robot(self):
        """加载SO-100机器人模型"""
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.robots import Robot
        
        # 设置USD文件路径
        robot_usd_path = self._get_robot_usd_path()
        
        # 添加机器人到场景
        prim_path = "/World/so100"
        add_reference_to_stage(usd_path=robot_usd_path, prim_path=prim_path)
        
        # 创建机器人实例
        self.robot = Robot(
            prim_path=prim_path,
            name="so100",
            position=np.array([0.0, 0.0, 0.0])
        )
    
    def _get_robot_usd_path(self):
        """获取机器人USD文件路径"""
        # 使用命令行参数中的路径，如果提供了的话
        if self.args.model_path:
            return self.args.model_path
        # 否则使用默认路径
        return "/home/wsy/ws_isaac/urdf/robots/SO_5DOF_ARM100_8j_URDF.SLDASM/urdf/SO_5DOF_ARM100_8j_URDF.SLDASM/SO_5DOF_ARM100_8j_URDF.SLDASM.usd"
    
    def initialize(self):
        """初始化仿真环境和机器人"""
        self.setup_world()
        self.load_robot()
        self.world.reset()
        self.world.play()
        self.robot.initialize()
        self.print_robot_info()
        
        # 摄像头初始化需要在机器人加载后进行
        self._setup_camera()
    
    def _setup_camera(self):
        """设置关节摄像头"""
        from omni.isaac.core.utils.viewports import set_camera_view
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.prims import create_prim
        from pxr import UsdGeom, Gf, Sdf, UsdPhysics
        
        if not self.robot:
            print("无法设置摄像头：机器人未加载")
            return
        
        try:
            # 创建一个小盒子作为摄像头的载体
            box_path = f"/World/so100/Wrist_Pitch_Roll/CameraMount"
            # box_path = f"/World/CameraMount"
            
            # 使用位置参数直接创建立方体
            from omni.isaac.core.utils.stage import get_current_stage
            stage = get_current_stage()
            
            # 首先检查盒子是否已存在
            if not stage.GetPrimAtPath(box_path).IsValid():
                # 调整盒子的位置和大小，使其更合理
                box = create_prim(
                    prim_path=box_path,
                    prim_type="Cube",
                    attributes={
                        "size": 0.02,  # 更小的盒子
                        "extent": [(-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)]
                    }
                    # 不在这里设置位置
                )
                
                # 使用USD的变换操作显式设置相对位置
                xform = UsdGeom.Xformable(stage.GetPrimAtPath(box_path))
                
                # 清除任何现有的变换
                xform.ClearXformOpOrder()
                
                # 添加平移操作（相对于父对象）
                translate_op = xform.AddTranslateOp()
                translate_op.Set(Gf.Vec3d(0.0, -0.02, 0.06))  # 相对于关节的位置，稍微前移和向上一点
                
                # 设置盒子颜色为黑色
                box_material = create_prim(
                    prim_path=f"{box_path}/Material",
                    prim_type="Material"
                )
            else:
                print(f"盒子已存在于路径: {box_path}")
                
            # 创建摄像头并附加到盒子上
            camera_path = f"{box_path}/Camera"
            print(f"正在将摄像头附加到路径: {camera_path}")
            
            # 首先检查摄像头是否已存在
            if not stage.GetPrimAtPath(camera_path).IsValid():
                self.camera = create_prim(
                    prim_path=camera_path,
                    prim_type="Camera",
                    attributes={
                        "focalLength": 10.0,  # 减小焦距以创建广角效果
                        "horizontalAperture": 24.0,  # 增加水平光圈，扩大视野
                        "verticalAperture": 18.0,  # 增加垂直光圈，保持合适的宽高比
                        "clippingRange": (0.01, 10000.0),  # 保持原来的裁剪范围
                    }
                    # 移除位置参数
                )
                
                # 使用USD变换操作设置相机的相对位置
                camera_xform = UsdGeom.Xformable(stage.GetPrimAtPath(camera_path))
                
                # 清除任何现有的变换
                camera_xform.ClearXformOpOrder()
                
                # 添加平移操作（相对于摄像头挂载）
                camera_translate_op = camera_xform.AddTranslateOp()
                camera_translate_op.Set(Gf.Vec3d(0.0, -0.01, 0.0))  # 位于摄像头挂载的中心位置
                
                # 添加旋转操作（如果需要调整相机朝向）
                rotate_op = camera_xform.AddRotateXYZOp()
                rotate_op.Set(Gf.Vec3f(-75, 0, 0))  # 重置为默认朝向，然后在运行时观察
            else:
                print(f"摄像头已存在于路径: {camera_path}")
                self.camera = stage.GetPrimAtPath(camera_path)
            
            # 设置摄像头视图参数
            from omni.isaac.sensor import Camera
            try:
                # 确保启用视觉传感器扩展
                from omni.isaac.core.utils import extensions
                extensions.enable_extension("omni.isaac.sensor")
                
                # 使用更小的分辨率以提高性能
                resolution = (640, 480)  # 将分辨率从320x240提高到640x480
                
                # 检查是否已经初始化了摄像头传感器
                if not hasattr(self, 'camera_sensor') or self.camera_sensor is None:
                    # 修改Camera初始化参数
                    self.camera_sensor = Camera(
                        prim_path=camera_path,
                        name="joint_camera",
                        resolution=resolution
                    )
                    self.camera_sensor.initialize()
                    print(f"摄像头初始化完成，分辨率: {resolution}")
                
                # 重置初始化计数器
                self.camera_init_counter = 0
                
            except Exception as e:
                print(f"摄像头传感器初始化失败: {e}")
                import traceback
                traceback.print_exc()
                self.camera_sensor = None
                
        except Exception as e:
            print(f"设置摄像头时出错: {e}")
            import traceback
            traceback.print_exc()
            self.camera_sensor = None
    
    def print_robot_info(self):
        """打印机器人关节信息"""
        if not self.robot:
            return
            
        num_joints = len(self.robot.get_joint_positions())
        joint_names = self.robot.dof_names
        print("\n=== Robot Information ===")
        print(f"Number of joints: {num_joints}")
        print("Joint names:", joint_names)
        print("Current joint positions:", self.robot.get_joint_positions())
    
    def publish_robot_state(self):
        """发布机器人关节状态到ROS"""
        if not self.ros_initialized or not self.robot:
            return
            
        # 获取当前时间
        current_time = time.time()
        
        # 控制发布频率
        if current_time - self.last_publish_time < 1.0/self.publish_rate:
            return
            
        self.last_publish_time = current_time
        
        # 创建关节状态消息
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
        joint_state_msg.name = self.robot.dof_names
        joint_state_msg.position = self.robot.get_joint_positions().tolist()
        
        # 如果可用，添加速度和力/扭矩信息
        try:
            joint_state_msg.velocity = self.robot.get_joint_velocities().tolist()
        except:
            joint_state_msg.velocity = [0.0] * len(joint_state_msg.name)
            
        try:
            joint_state_msg.effort = self.robot.get_applied_joint_efforts().tolist()
        except:
            joint_state_msg.effort = [0.0] * len(joint_state_msg.name)
        
        # 发布关节状态
        self.joint_state_publisher.publish(joint_state_msg)
        
        # 发布机器人基座位置变换
        self._publish_base_transform()
        
        # 处理ROS事件
        rclpy.spin_once(self.ros_node, timeout_sec=0)
    
    def _publish_base_transform(self):
        """发布机器人基座的tf变换"""
        if not self.transform_broadcaster or not self.robot:
            return
            
        try:
            # 获取机器人位置
            position = self.robot.get_world_pose()[0]
            
            # 创建并发布变换
            t = TransformStamped()
            t.header.stamp = self.ros_node.get_clock().now().to_msg()
            t.header.frame_id = 'world'
            t.child_frame_id = 'so100_base'
            
            # 设置位置 - 确保转换为标准Python浮点数
            t.transform.translation.x = float(position[0])
            t.transform.translation.y = float(position[1])
            t.transform.translation.z = float(position[2])
            
            # 设置旋转（简化为无旋转）
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            
            # 发布变换
            self.transform_broadcaster.sendTransform(t)
        except Exception as e:
            print(f"Error publishing transform: {e}")
    
    def _joint_command_callback(self, msg):
        """处理关节命令回调"""
        if not self.robot:
            return
            
        try:
            # 获取命令中的关节位置
            joint_positions = np.array(msg.position)
            self._handle_joint_command(joint_positions)
        except Exception as e:
            self._log_once("joint_cmd_error", f"设置关节位置时出错: {e}")
    
    def _simple_command_callback(self, msg):
        """处理简单控制命令回调 (Float64MultiArray)"""
        if not self.robot:
            return
            
        try:
            # 获取命令中的关节位置
            joint_positions = np.array(msg.data)
            self._handle_joint_command(joint_positions)
        except Exception as e:
            self._log_once("simple_cmd_error", f"设置关节位置时出错: {e}")
    
    def _handle_joint_command(self, joint_positions):
        """处理关节位置命令
        
        Args:
            joint_positions: 目标关节位置数组
        """
        # 检查命令与机器人关节数量是否匹配
        if len(joint_positions) != len(self.robot.dof_names):
            self._log_once("joint_count_mismatch", 
                f"警告: 接收到的关节命令数量 ({len(joint_positions)}) 与机器人关节数量 ({len(self.robot.dof_names)}) 不匹配")
            return
            
        # 设置目标关节位置
        self._start_smooth_motion(joint_positions)
        print(f"目标关节位置设置为: {joint_positions}")
    
    def _start_smooth_motion(self, target_positions):
        """启动平滑运动
        
        Args:
            target_positions: 目标关节位置
        """
        self.target_joint_positions = target_positions
        self.motion_start_positions = self.robot.get_joint_positions()
        self.motion_start_time = time.time()
        self.is_moving = True
        
    def _update_smooth_motion(self):
        """更新平滑运动"""
        if not self.is_moving or self.target_joint_positions is None:
            return
            
        current_time = time.time()
        elapsed_time = current_time - self.motion_start_time
        
        # 计算插值比例 (0.0 到 1.0)
        if elapsed_time >= self.motion_duration:
            # 运动完成
            self.robot.set_joint_positions(self.target_joint_positions)
            self.is_moving = False
            return
            
        # 线性插值
        t = elapsed_time / self.motion_duration
        current_positions = self.motion_start_positions + t * (self.target_joint_positions - self.motion_start_positions)
        
        # 设置插值后的关节位置
        self.robot.set_joint_positions(current_positions)
    
    def publish_camera_image(self):
        """发布摄像头图像到ROS"""
        if not self.ros_initialized or not hasattr(self, 'camera_sensor') or self.camera_sensor is None or not self.camera_publisher:
            return
            
        # 等待几个周期让摄像头初始化
        if self.camera_init_counter < self.camera_init_delay:
            self.camera_init_counter += 1
            if self.camera_init_counter % 5 == 0:  # 每5帧打印一次
                print(f"等待摄像头初始化... {self.camera_init_counter}/{self.camera_init_delay}")
            return
            
        try:
            camera_data = self._get_camera_data()
            if camera_data is None:
                return
                
            rgb_data = self._process_camera_data(camera_data)
            if rgb_data is None:
                return
                
            # 转换为ROS图像消息并发布
            self._publish_image_message(rgb_data)
            
        except Exception as e:
            self._log_once("camera_publish_error", f"发布摄像头图像时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_camera_data(self):
        """获取摄像头数据"""
        try:
            camera_data = self.camera_sensor.get_rgba()
            if camera_data is None or (hasattr(camera_data, 'size') and camera_data.size == 0):
                self._log_once("empty_camera_data", "摄像头返回空数据，跳过此帧")
                return None
                
            # 检查数据形状并打印信息（仅一次）
            self._log_once("camera_data_info", f"摄像头数据: 类型={type(camera_data)}, 形状={camera_data.shape}")
            return camera_data
            
        except Exception as cam_error:
            self._log_once("camera_get_error", f"获取摄像头数据失败: {cam_error}")
            return None
    
    def _process_camera_data(self, camera_data):
        """处理摄像头数据为RGB格式
        
        Args:
            camera_data: 从摄像头获取的原始数据
            
        Returns:
            处理后的RGB图像数据，如果处理失败则返回None
        """
        if len(camera_data.shape) == 1:
            return self._process_1d_camera_data(camera_data)
        elif len(camera_data.shape) == 3:
            # 已经是三维形状
            if camera_data.shape[2] >= 3:
                return camera_data[:, :, :3]
            else:
                self._log_once("unsupported_channels", f"不支持的通道数: {camera_data.shape[2]}")
                return None
        else:
            self._log_once("unsupported_shape", f"不支持的数据形状: {camera_data.shape}")
            return None
    
    def _process_1d_camera_data(self, camera_data):
        """处理一维摄像头数据
        
        Args:
            camera_data: 一维摄像头数据
            
        Returns:
            处理后的RGB图像数据，如果处理失败则返回None
        """
        if camera_data.size == 0:
            return None
                
        try:
            # 推断分辨率
            width, height = getattr(self.camera_sensor, "resolution", (320, 240))
                
            # 计算通道数
            if camera_data.size % (width * height) == 0:
                channels = camera_data.size // (width * height)
            else:
                self._log_once("resolution_mismatch", 
                    f"无法确定通道数: 数据大小 {camera_data.size} 不是分辨率 {width}x{height} 的倍数")
                return None
                
            # 重塑数组    
            reshaped_data = camera_data.reshape(height, width, channels)
            
            # 提取RGB通道
            if channels >= 3:
                return reshaped_data[:, :, :3]
            else:
                self._log_once("insufficient_channels", f"不支持的通道数: {channels}")
                return None
                
        except Exception as reshape_error:
            self._log_once("reshape_error", 
                f"重塑图像数据失败: {reshape_error}，数据形状: {camera_data.shape}，大小: {camera_data.size}")
            return None
    
    def _publish_image_message(self, rgb_data):
        """发布RGB图像数据到ROS
        
        Args:
            rgb_data: 要发布的RGB图像数据
        """
        # 确认我们有有效数据
        if rgb_data is None or rgb_data.size == 0:
            return
            
        # 转换为ROS图像消息
        img_msg = self.bridge.cv2_to_imgmsg(rgb_data, encoding="rgb8")
        img_msg.header.stamp = self.ros_node.get_clock().now().to_msg()
        img_msg.header.frame_id = f"joint_{self.camera_joint_index}_camera"
        
        # 发布图像
        self.camera_publisher.publish(img_msg)
    
    def run(self):
        """运行主仿真循环"""
        if not self.world or not self.robot:
            print("Error: World or robot not initialized")
            return
            
        step_count = 0
        dwell_time = 0
        
        # Add debug message for ROS status
        if self.args.ros:
            print(f"ROS publishing enabled at {self.publish_rate} Hz")
        else:
            print("ROS publishing disabled, use --ros flag to enable")
            
        # 给摄像头一些时间完全初始化
        print("等待摄像头初始化...")
        for _ in range(10):
            self.world.step(render=True)
            
        while self.simulation_app.is_running():
            # 处理ROS消息
            if self.args.ros:
                rclpy.spin_once(self.ros_node, timeout_sec=0)
            
            # 更新平滑运动
            self._update_smooth_motion()
            
            # 执行仿真步骤
            self.world.step(render=True)
            
            # 发布机器人状态到ROS
            if self.args.ros:
                # 发布摄像头图像 - 与仿真同步
                self.publish_camera_image()
                
                # 发布其他状态信息（受频率控制）
                self.publish_robot_state()
            
            dwell_time += 1
            step_count += 1
            
    
    def close(self):
        """关闭仿真环境"""
        # 关闭ROS节点
        if self.ros_initialized:
            if self.ros_node is not None:
                self.ros_node.destroy_node()
            rclpy.shutdown()
            print("ROS node shutdown")
            
        # 关闭仿真应用
        if self.simulation_app:
            self.simulation_app.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="SO-100 Robot Simulation")
    parser.add_argument("--headless", default=False, action="store_true", help="Run in headless mode")
    parser.add_argument("--model-path", type=str, help="Path to robot USD model", default="")
    parser.add_argument("--ros", default=True, action="store_true", help="Enable ROS integration")
    return parser.parse_known_args()[0]


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 创建并运行仿真
    sim = SO100Simulation(args)
    
    # Print startup message about ROS status
    if args.ros:
        print("Starting simulation with ROS integration enabled")
    else:
        print("Starting simulation without ROS integration (use --ros flag to enable)")
    
    try:
        sim.initialize()
        sim.run()
    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
