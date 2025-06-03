#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-100机器人硬件控制节点 - 接收控制命令并控制真实机械臂"""

import rclpy
import numpy as np
import time
import threading
import json
import os
from typing import List, Optional, Dict, Tuple
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from ft_servo_wrapper import FTServoWrapper
import logging


class So100Driver(Node):
    """SO-100机器人硬件控制节点"""
    
    def __init__(self):
        # 配置Python logging
        # logging.basicConfig(
        #     level=logging.INFO,
        #     format='[%(levelname)s] [%(name)s] [%(asctime)s] [%(filename)s:%(lineno)d] %(message)s',
        #     datefmt='%Y-%m-%d %H:%M:%S'
        # )
        logging.basicConfig(
            level=logging.INFO,
            format='[%(levelname)s][%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger('so100_hardware_controller')
        
        # 先初始化ROS节点
        super().__init__('so100_hardware_controller')
        
        # 1. 舵机相关
        self.port = '/dev/ttyACM0'
        self.baudrate = 1000000
        self.servo_ids = [1, 2, 3, 4, 5, 6]
        # self.speed = 200 # 舵机运动的速度
        # self.acceleration = 100 # 舵机运动的加速度
        self.servo_controller = FTServoWrapper(self.port, self.baudrate)
        self.initialize_hardware()

        # 2. so100 相关
        self.calibration_file = 'so100_calibration.json'
        self.zero_positions = []
        self.joint_limits = []
        self.position_tolerance = 10
        self.current_joints = [0] * len(self.servo_ids)  # 当前位置
        # self.target_positions = [0] * len(self.servo_ids)   # 目标位置

        # 3. ros 相关
        self.publish_rate = 100

        self.joint_state_publisher = self.create_publisher(
            JointState, 'so100_joint_states', int(self.publish_rate) )
        self.command_subscriber = self.create_subscription(
            Float64MultiArray, 'so100_position_commands', 
            self.command_callback, 10)
        
        self.timer = self.create_timer(
            1.0 / self.publish_rate, self.publish_joint_states)
        
        # 4. 其它
        self.angle_to_pulse_ratio = 651.74 # 关节角度转脉冲的转换系数,4096脉冲/2pi ≈ 651.74脉冲/rad
        self.position_lock = threading.Lock()
        self.logger.info('SO-100硬件控制节点已启动')

        if not self.load_calibration_data():
            self.initialized = False
            self.logger.error('加载标定数据失败')
            return  False# 直接返回，不继续初始化
        
        # 标记初始化成功
        self.initialized = True

    def _check_joint_limits(self, servo_id: int, joint_rad: int):
        """检查关节是否超出限制"""
        if joint_rad < self.joint_limits[servo_id-1][0]:
            self.logger.warning(f'关节{servo_id}({joint_rad})超出最小限制({self.joint_limits[servo_id-1][0]})')
            joint_rad = self.joint_limits[servo_id-1][0]
        elif joint_rad > self.joint_limits[servo_id-1][1]:
            self.logger.warning(f'关节{servo_id}({joint_rad})超出最大限制({self.joint_limits[servo_id-1][1]})')
            joint_rad = self.joint_limits[servo_id-1][1]
        return joint_rad


    def initialize_hardware(self):
        """
        初始化硬件连接
        1. 连接舵机串口
        2. 检查所需舵机是否在线
        """
        try:
            # 连接舵机控制器
            if not self.servo_controller.connect():
                self.logger.error('舵机控制器连接失败')
                return False
            
            self.logger.info(f'舵机控制器连接成功: {self.port}')
            
            # 检查所需舵机是否在线
            for servo_id in self.servo_ids:
                if self.servo_controller.ping(servo_id) == False:
                    self.logger.warning(f'舵机 {servo_id} 不在线')
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f'硬件初始化失败: {e}')
            return False
    
    def read_joint(self, servo_id: int):
        """读取当前舵机位置"""
        try:
            pulse = self.servo_controller.read_position(servo_id)
            return pulse / self.angle_to_pulse_ratio
        except Exception as e:
            self.logger.error(f'读取舵机位置出错: {e}')

    def read_joints(self):
        """读取当前舵机位置"""
        try:
            with self.position_lock:
                joints = self.servo_controller.sync_read_positions(self.servo_ids)

                if len(joints) != len(self.servo_ids):
                    self.logger.error(f'读取舵机位置失败，舵机数量错误: {len(joints)} != {len(self.servo_ids)}')
                
                if None not in joints:
                    # 转换脉冲值到角度  
                    for i, joint in enumerate(joints):
                        self.current_joints[i] = joint / self.angle_to_pulse_ratio
                    return self.current_joints
                else:
                    # 找出读取失败的舵机ID
                    failed_servos = []
                    for i, joint in enumerate(joints):
                        if joint is None:
                            failed_servos.append(self.servo_ids[i])
                    self.logger.error(f'读取舵机位置失败，失败的舵机ID: {failed_servos}')

        except Exception as e:
            self.logger.error(f'读取舵机位置出错: {e}')
    

    def write_joint(self, servo_id: int, joint_rad: int, speed: int = 200, acceleration: int = 100):
        """一次写入so100一个关节的位置"""
        try:
            joint_rad = self._check_joint_limits(servo_id, joint_rad)
            pulse = int(joint_rad * self.angle_to_pulse_ratio)
            self.servo_controller.write_position(servo_id, pulse, speed, acceleration)
        except Exception as e:
            self.logger.error(f'写入舵机位置出错: {e}')

    def write_joints(self, joints_rad: List[int], speed: int = 200, acceleration: int = 100):
        """一次写入so100所有关节的位置"""
        try:
            if len(joints_rad) != len(self.servo_ids):
                self.logger.error(f'关节数量错误: 收到{len(joints_rad)}个，需要{len(self.servo_ids)}个')
                return
            
            print("write joints_rad: ", joints_rad)
            # 准备同步写入数据
            servo_data = []
            for i, servo_id in enumerate(self.servo_ids):
                joint_rad = joints_rad[i]
                joint_rad = self._check_joint_limits(servo_id, joint_rad)

                pulse = int(joint_rad * self.angle_to_pulse_ratio)
                servo_data.append((servo_id, pulse, speed, acceleration))
                # print("1servo_data: ", servo_data)
            
            # print("2servo_data: ", servo_data)
            time.sleep(1)
            success = self.servo_controller.sync_write_positions(servo_data)
            
            if not success:
                self.logger.error('同步写入舵机位置失败')
                return False    
            return True
        except Exception as e:
            self.logger.error(f'写舵机数据出错: {e}')
            return False

    def publish_joint_states(self):
        """发布关节状态"""
        try:
            # 读取当前位置
            joints = self.read_joints()
            # print("current_joints: ", self.current_joints)
            
            # 创建关节状态消息
            joint_state = JointState()
            joint_state.header.stamp = self.get_clock().now().to_msg()
            joint_state.header.frame_id = 'base_link'
            joint_state.name = [f'joint_{i+1}' for i in range(len(self.servo_ids))]
            
            joint_state.position = joints
            
            # 发布关节状态
            self.joint_state_publisher.publish(joint_state)
            
        except Exception as e:
            self.logger.error(f'发布关节状态出错: {e}')

    def command_callback(self, msg: Float64MultiArray):
        """处理位置命令"""
        try:
            if len(msg.data) != len(self.servo_ids):
                self.logger.error(
                    f'命令关节数量错误: 收到{len(msg.data)}个，需要{len(self.servo_ids)}个')
                return
            
            # 转换为普通列表
            target_rad = list(msg.data)  # 或者 [float(x) for x in msg.data]
            
            print("收到目标位置: ", target_rad)
            self.write_joints(target_rad)
            
            self.logger.debug(f'发送位置命令: {[f"{a:.3f}" for a in target_rad]}')
            
        except Exception as e:
            self.logger.error(f'处理位置命令出错: {e}')
    
    def stop(self):
        """关闭节点"""
        try:
            self.logger.info('正在关闭硬件控制节点...')
            
            # 断开舵机控制器连接
            if hasattr(self, 'servo_controller'):
                self.servo_controller.disconnect()
            
            self.logger.info('硬件控制节点已关闭')
            
        except Exception as e:
            self.logger.error(f'关闭节点出错: {e}')
    
    def load_calibration_data(self):
        """从配置文件加载标定数据"""
        try:
            with open(self.calibration_file, 'r', encoding='utf-8') as f:
                calibration_data = json.load(f)
            
            # 加载关节限制
            joint_limits_dict = calibration_data.get('joint_limits', {})
            for id in self.servo_ids:  
                jointi_limit = joint_limits_dict.get(str(id))
                if jointi_limit is None:
                    self.logger.error(f'关节{id}的限制配置未找到')
                    return False
                self.joint_limits.append([jointi_limit["min"], jointi_limit["max"]])
            
            self.init_pose_joints = calibration_data["poses_joints"]["init_pose_joints"]
            self.home_pose_joints = calibration_data["poses_joints"]["home_pose_joints"]
            self.up_pose_joints = calibration_data["poses_joints"]["up_pose_joints"]

            self.logger.info(f'成功加载标定数据: {self.calibration_file}')
            self.logger.info(f'关节限制: {self.joint_limits}')
            self.logger.info(f'init_pose关节位置: {self.init_pose_joints}')
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

    def move_to_preset_pose(self, pose_name: str):
        """移动到预设位置"""
        if pose_name == 'init_pose':
            self.write_joints(self.init_pose_joints)
        elif pose_name == 'home_pose':
            self.write_joints(self.home_pose_joints)
        elif pose_name == 'up_pose':
            self.write_joints(self.up_pose_joints)

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    
    try:
        # 创建硬件控制节点
        hardware_controller = So100Driver()
        
        # 检查初始化是否成功
        if not hasattr(hardware_controller, 'initialized') or not hardware_controller.initialized:
            hardware_controller.logger.error('硬件控制节点初始化失败，程序退出')
            return
        
        # 运行节点
        rclpy.spin(hardware_controller)

    except KeyboardInterrupt:
        print('用户中断')
    except Exception as e:
        print(f'节点运行出错: {e}')
    finally:
        # 清理资源
        if 'hardware_controller' in locals():
            hardware_controller.stop()
            hardware_controller.destroy_node()
        
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
