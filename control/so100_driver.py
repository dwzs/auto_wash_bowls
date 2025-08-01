#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-100机器人硬件控制节点 - 接收控制命令并控制真实机械臂"""

import time
import threading
import json
import os
from typing import List, Optional, Dict, Tuple
from .FT_servo.ft_servo_wrapper import FTServoWrapper

class So100Driver:
    """SO-100机器人硬件控制节点"""
    
    def __init__(self):
        # 直接加载配置文件
        config_path = os.path.join(os.path.dirname(__file__), "config", "so100_config.json")
        with open(config_path, "r") as f:
            self.cfg = json.load(f)

        # 硬件配置
        self.angle_to_pulse_ratio = 651.74
        self.port = self.cfg["robot"]["port"]
        self.baudrate = self.cfg["robot"]["baudrate"]
        self.servo_ids = self.cfg["robot"]["servo_ids"]
        self.servo_dir = self.cfg["robot"]["servo_dir"]
        self.speed = self.cfg["robot"]["speed"]
        self.acceleration = self.cfg["robot"]["acceleration"]
        self.calibration_file = self.cfg["calibration"]["calibration_file"]
        
        # 初始化舵机控制器
        self.servo_controller = FTServoWrapper(self.port, self.baudrate)
        self.initialize_hardware()

        # SO-100相关数据
        self.zero_pose_joints = []
        self.joint_limits_offseted = []
        self.current_joints = [0] * len(self.servo_ids)  # 当前位置
        
        # 线程锁
        self.position_lock = threading.Lock()

        if not self.load_calibration_data():
            self.initialized = False
            print('ERROR: 加载标定数据失败')
            return
        
        # 标记初始化成功
        self.initialized = True

    def _check_and_limit_joint(self, servo_id: int, joint_rad: float):
        """检查关节是否超出限制"""
        if joint_rad < self.joint_limits_offseted[servo_id-1][0]:
            print(f'WARNING: 关节{servo_id}({joint_rad})超出最小限制({self.joint_limits_offseted[servo_id-1][0]})')
            joint_rad = self.joint_limits_offseted[servo_id-1][0]
        elif joint_rad > self.joint_limits_offseted[servo_id-1][1]:
            print(f'WARNING: 关节{servo_id}({joint_rad})超出最大限制({self.joint_limits_offseted[servo_id-1][1]})')
            joint_rad = self.joint_limits_offseted[servo_id-1][1]
        return joint_rad

    def _rad_offseted_to_pulse(self, joint_rad: float, servo_id: int):
        """将关节角度(弧度)转换为脉冲值"""
        return int((joint_rad + self.zero_pose_joints[servo_id-1]) * self.angle_to_pulse_ratio)
    
    def _rads_offseted_to_pulses(self, joints_rad: List[float]):
        """将关节角度(弧度)转换为脉冲值"""
        pulses = []
        for i, joint_rad in enumerate(joints_rad):
            pulses.append(self._rad_offseted_to_pulse(joint_rad, i+1))
        return pulses

    def _pulse_to_rad_offseted(self, pulse: int, servo_id: int):
        """将脉冲值转换为关节角度(弧度)"""
        return (pulse / self.angle_to_pulse_ratio) - self.zero_pose_joints[servo_id-1]

    def _pulses_to_rads_offseted(self, pulses: List[int]):
        """将脉冲值转换为关节角度(弧度)"""
        joints_rad = []
        for i, pulse in enumerate(pulses):
            joints_rad.append(self._pulse_to_rad_offseted(pulse, i+1) * self.servo_dir[i])
        return joints_rad

    def initialize_hardware(self):
        """初始化硬件连接"""
        try:
            # 连接舵机控制器
            if not self.servo_controller.connect():
                print('ERROR: 舵机控制器连接失败')
                return False
            
            print(f'INFO: 舵机控制器连接成功: {self.port}')
            
            # 检查所需舵机是否在线
            for servo_id in self.servo_ids:
                if self.servo_controller.ping(servo_id) == False:
                    print(f'WARNING: 舵机 {servo_id} 不在线')
                    return False
            
            print(f'INFO: success connect all servos: {self.servo_ids}')
            return True
            
        except Exception as e:
            print(f'ERROR: 硬件初始化失败: {e}')
            return False
    
    def read_joint_pulse(self, servo_id: int):
        """读取当前舵机位置(脉冲值)"""
        try:
            pulse = self.servo_controller.read_position(servo_id)
            return pulse
        except Exception as e:
            print(f'ERROR: 读取舵机位置出错: {e}')

    def read_joint(self, servo_id: int):
        """读取当前舵机经过零位偏移后的位置(弧度)"""
        joint_pulse = self.read_joint_pulse(servo_id)
        joint_pulse = joint_pulse * self.servo_dir[servo_id-1]
        return self._pulse_to_rad_offseted(joint_pulse, servo_id)

    def read_joints_pulse(self):
        """读取当前舵机位置(脉冲值)"""
        try:
            with self.position_lock:
                joints = self.servo_controller.sync_read_positions(self.servo_ids)

                if len(joints) != len(self.servo_ids):
                    print(f'ERROR: 读取舵机位置失败，舵机数量错误: {len(joints)} != {len(self.servo_ids)}')
                
                if None not in joints:
                    return joints
                else:
                    # 找出读取失败的舵机ID
                    failed_servos = []
                    for i, joint in enumerate(joints):
                        if joint is None:
                            failed_servos.append(self.servo_ids[i])
                    print(f'ERROR: 读取舵机位置失败，失败的舵机ID: {failed_servos}')

        except Exception as e:
            print(f'ERROR: 读取舵机位置出错: {e}')
    
    def read_joints(self):
        """读取当前舵机零位偏移后的位置(弧度)"""
        joints_pulse = self.read_joints_pulse()
        return self._pulses_to_rads_offseted(joints_pulse)

    def write_joint(self, servo_id: int, joint_rad: float, speed: int = None, acceleration: int = None):
        """写入单个关节零位偏移后的位置(弧度)"""
        try:
            # 使用配置的默认值
            if speed is None:
                speed = self.speed
            if acceleration is None:
                acceleration = self.acceleration
                
            joint_rad_offseted = self._check_and_limit_joint(servo_id, joint_rad)
            joint_rad_offseted = joint_rad_offseted * self.servo_dir[servo_id-1]
            raw_pulse = self._rad_offseted_to_pulse(joint_rad_offseted, servo_id)
            self.servo_controller.write_position(servo_id, raw_pulse, speed, acceleration)
            return True
        except Exception as e:
            print(f'ERROR: 写入舵机位置出错: {e}')
            return False

    def write_joints(self, joints_rad: List[float], speed: int = None, acceleration: int = None):
        """写入所有关节零位偏移后的位置(弧度)"""
        try:
            # 使用配置的默认值
            if speed is None:
                speed = self.speed
            if acceleration is None:
                acceleration = self.acceleration
                
            if len(joints_rad) != len(self.servo_ids):
                print(f'ERROR: 关节数量错误: 收到{len(joints_rad)}个，需要{len(self.servo_ids)}个')
                return False
            
            print("write joints_rad: ", joints_rad)
            # 准备同步写入数据
            raw_pulses = []
            for i, servo_id in enumerate(self.servo_ids):
                joint_rad_offseted = joints_rad[i]
                if joint_rad_offseted is not None:
                    joint_rad_offseted = self._check_and_limit_joint(servo_id, joint_rad_offseted)
                    joint_rad_offseted = joint_rad_offseted * self.servo_dir[servo_id-1]
                    raw_pulse = self._rad_offseted_to_pulse(joint_rad_offseted, servo_id)
                    raw_pulses.append((servo_id, raw_pulse, speed, acceleration))
            
            print("raw_pulse: ", raw_pulses)
            success = self.servo_controller.sync_write_positions(raw_pulses)
            
            if not success:
                print('ERROR: 同步写入舵机位置失败')
                return False    
            return True
        except Exception as e:
            print(f'ERROR: 写舵机数据出错: {e}')
            return False

    def stop(self):
        """关闭节点"""
        try:
            print('INFO: 正在关闭硬件控制节点...')
            
            # 断开舵机控制器连接
            if hasattr(self, 'servo_controller'):
                self.servo_controller.disconnect()
            
            print('INFO: 硬件控制节点已关闭')
            
        except Exception as e:
            print(f'ERROR: 关闭节点出错: {e}')
    
    def load_calibration_data(self):
        """从配置文件加载标定数据"""
        try:
            with open(self.calibration_file, 'r', encoding='utf-8') as f:
                cal_data = json.load(f)
            
            # 加载关节限制
            joint_limits_dict = cal_data["joint_limits_offseted"]
            for servo_id in self.servo_ids:  
                joint_limit = joint_limits_dict[str(servo_id)]
                self.joint_limits_offseted.append([joint_limit["min"], joint_limit["max"]])
            
            # 加载预设位置
            poses = cal_data["poses_joints"]
            self.zero_pose_joints = poses["zero_pose_joints"]
            self.home_pose_joints = poses["home_pose_joints"]
            self.up_pose_joints = poses["up_pose_joints"]

            poses_offseted = cal_data["poses_joints_offseted"]
            self.zero_pose_joints_offseted = poses_offseted["zero_pose_joints"]
            self.home_pose_joints_offseted = poses_offseted["home_pose_joints"]
            self.up_pose_joints_offseted = poses_offseted["up_pose_joints"]

            print(f'INFO: 成功加载标定数据: {self.calibration_file}')
            print(f'INFO: 关节限制: {self.joint_limits_offseted}')
            print(f'INFO: zero_pose关节位置: {self.zero_pose_joints}')
            print(f'INFO: home_pose关节位置: {self.home_pose_joints}')
            print(f'INFO: up_pose关节位置: {self.up_pose_joints}')
            print(f'INFO: zero_pose关节位置(offseted): {self.zero_pose_joints_offseted}')
            print(f'INFO: home_pose关节位置(offseted): {self.home_pose_joints_offseted}')
            print(f'INFO: up_pose关节位置(offseted): {self.up_pose_joints_offseted}')
            return True
        
        except FileNotFoundError:
            print(f'ERROR: 标定文件未找到: {self.calibration_file}')
            return False
        except json.JSONDecodeError as e:
            print(f'ERROR: 标定文件格式错误: {e}')
            return False
        except Exception as e:
            print(f'ERROR: 加载标定数据失败: {e}')
            return False

    def move_to_preset_pose(self, pose_name: str):
        """移动到预设位置"""
        if pose_name == 'zero_pose':
            self.write_joints(self.zero_pose_joints_offseted)
        elif pose_name == 'fold_pose':
            self.write_joints(self.home_pose_joints_offseted)
        elif pose_name == 'up_pose':
            self.write_joints(self.up_pose_joints_offseted)

def main():
    try:
        hardware_driver = So100Driver()
        if not hasattr(hardware_driver, 'initialized') or not hardware_driver.initialized:
            print('ERROR: 硬件控制节点初始化失败，程序退出')
            return

        # 示例：移动到home位
        hardware_driver.move_to_preset_pose("zero_pose")
        time.sleep(2)
        # 读取关节
        joints_offseted = hardware_driver.read_joints()
        print(f'当前关节角度(弧度): {joints_offseted}')

    except KeyboardInterrupt:
        print('INFO: 用户中断')
    except Exception as e:
        print(f'ERROR: 脚本运行出错: {e}')
    finally:
        if 'hardware_driver' in locals():
            hardware_driver.stop()

if __name__ == '__main__':
    main()
