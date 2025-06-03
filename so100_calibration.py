#!/usr/bin/env python

import os
import json
import sys
import math
import time

sys.path.append("..")
from scservo_sdk import *     


class SO100Calibrator:
    """SO100机械臂标定类"""
    
    def __init__(self, device_name='/dev/ttyACM0', baudrate=1000000, servo_ids=[1, 2, 3, 4, 5, 6]):
        """
        初始化标定器
        
        Args:
            device_name (str): 串口设备路径
            baudrate (int): 波特率
            servo_ids (list): 舵机ID列表
        """
        self.device_name = device_name
        self.baudrate = baudrate
        self.servo_ids = servo_ids
        self.position_to_radian_ratio = 2 * math.pi / 4096  # 舵机分辨率为4096
        
        self.port_handler = None
        self.servo = None
        
    def position_to_radian(self, position):
        """将舵机位置值转换为弧度"""
        return position * self.position_to_radian_ratio

    def radian_to_position(self, radian):
        """将弧度转换为舵机位置值"""
        return int(radian / self.position_to_radian_ratio)

    def __enter__(self):
        """上下文管理器入口 - 自动初始化连接"""
        if self.initialize_port():
            return self
        raise ConnectionError(f"无法打开串口 {self.device_name}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口 - 自动关闭连接"""
        self.close_port()

    def initialize_port(self):
        """初始化并打开串口通信"""
        self.port_handler = PortHandler(self.device_name)
        
        try:
            self.port_handler.openPort()
            self.port_handler.setBaudRate(self.baudrate)
            self.servo = sms_sts(self.port_handler)
            print(f"串口 {self.device_name} 已成功打开，波特率: {self.baudrate}")
            return True
        except:
            print(f"无法打开串口 {self.device_name} 或设置波特率")
            return False

    def close_port(self):
        """关闭串口连接"""
        if self.port_handler:
            self.port_handler.closePort()
            print("串口连接已关闭")

    def read_servo_position(self, servo_id):
        """读取单个舵机位置"""
        position, result, error = self.servo.ReadPos(servo_id)
        if result == COMM_SUCCESS:
            return self.position_to_radian(position)
        else:
            print(f"无法读取舵机 ID: {servo_id}, 错误: {error}")
            return None

    def read_all_positions(self):
        """读取所有舵机位置"""
        positions = {}
        for servo_id in self.servo_ids:
            pos = self.read_servo_position(servo_id)
            if pos is not None:
                positions[servo_id] = pos
        return positions

    def set_torque(self, enable=True):
        """设置所有舵机扭矩状态"""
        value = 1 if enable else 0
        action = "启用" if enable else "禁用"
        
        for servo_id in self.servo_ids:
            result, error = self.servo.write1ByteTxRx(servo_id, SMS_STS_TORQUE_ENABLE, value)
            if result != COMM_SUCCESS:
                print(f"警告: 无法{action}舵机 ID: {servo_id} 的扭矩")

    def get_user_position(self, servo_id, description):
        """获取用户手动设置的位置"""
        print(f"\n[舵机 ID: {servo_id}] {description}，然后按回车键...")
        input()
        return self.read_servo_position(servo_id)

    def calibrate_joint_limits(self):
        """标定关节限位"""
        joint_limits = {}
        
        for servo_id in self.servo_ids:
            print(f"\n=== 标定舵机 ID: {servo_id} 的限位 ===")
            
            pos1 = self.get_user_position(servo_id, "请移动到第一个限位位置")
            pos2 = self.get_user_position(servo_id, "请移动到第二个限位位置")
            
            if pos1 is not None and pos2 is not None:
                joint_limits[servo_id] = {
                    "min": min(pos1, pos2),
                    "max": max(pos1, pos2)
                }
                print(f"舵机 ID: {servo_id} 限位: {joint_limits[servo_id]['min']:.4f} ~ {joint_limits[servo_id]['max']:.4f}")
        
        return joint_limits

    def calibrate_pose(self, pose_name, description):
        """标定单个位姿"""
        print(f"\n=== 标定 {pose_name} ===")
        print(f"描述: {description}")
        print(f"请手动移动机械臂到 {pose_name} 位置，然后按回车键...")
        input()
        
        positions = self.read_all_positions()
        if len(positions) == len(self.servo_ids):
            print(f"{pose_name} 标定成功！")
            # 将字典转换为按servo_ids顺序排列的列表
            position_list = []
            for servo_id in self.servo_ids:
                if servo_id in positions:
                    position_list.append(positions[servo_id])
                    print(f"舵机 ID: {servo_id}, 位置: {positions[servo_id]:.4f} 弧度")
                else:
                    print(f"警告: 舵机 ID: {servo_id} 位置读取失败")
                    return None
            return position_list
        else:
            print(f"{pose_name} 标定失败！")
            return None

    def calibrate_poses(self):
        """标定所有关键位姿"""
        pose_configs = {
            "home_pose": "机械臂的零位姿态",
            "up_pose": "机械臂的竖直向上位姿", 
            "init_pose": "机械臂的初始化位姿"
        }
        
        poses = {}
        for pose_name, description in pose_configs.items():
            poses[pose_name] = self.calibrate_pose(pose_name, description)
        
        return poses

    def save_calibration_data(self, calibration_data, filepath='so100_calibration.json'):
        """保存标定数据"""
        try:
            with open(filepath, 'w') as f:
                json.dump(calibration_data, f, indent=4)
            print(f"\n标定数据已保存到 {filepath}")
            return True
        except Exception as e:
            print(f"保存失败: {e}")
            return False

    def load_calibration_data(self, filepath='so100_calibration.json'):
        """加载标定数据"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载失败: {e}")
            return None

    def move_to_pose(self, pose_name, config_file='so100_calibration.json'):
        """移动到指定位姿"""
        with self:  # 使用上下文管理器自动管理连接
            calibration_data = self.load_calibration_data(config_file)
            if not calibration_data or "poses" not in calibration_data:
                print("无法加载标定数据")
                return False
            
            if pose_name not in calibration_data["poses"]:
                print(f"位姿 {pose_name} 不存在")
                return False
            
            pose_data = calibration_data["poses"][pose_name]
            if not pose_data:
                print(f"位姿 {pose_name} 数据无效")
                return False
            
            print(f"正在移动到 {pose_name}...")
            self.set_torque(True)  # 启用扭矩
            
            # 添加调试信息
            print(f"位姿数据: {pose_data}")
            
            success = True
            for i, servo_id in enumerate(self.servo_ids):
                # 修复：pose_data是列表，不是字典
                if i < len(pose_data):
                    target_radian = pose_data[i]
                    target_position = self.radian_to_position(target_radian)
                    print(f"舵机 ID: {servo_id}, 目标弧度: {target_radian:.4f}, 目标位置: {target_position}")
                    
                    # 读取当前位置进行对比
                    current_pos = self.read_servo_position(servo_id)
                    if current_pos is not None:
                        print(f"舵机 ID: {servo_id}, 当前位置: {current_pos:.4f} 弧度")
                    
                    result, error = self.servo.WritePosEx(servo_id, target_position, 1000, 50)
                    if result != COMM_SUCCESS:
                        print(f"舵机 ID: {servo_id} 移动失败，错误码: {error}")
                        success = False
                    else:
                        print(f"舵机 ID: {servo_id} 指令发送成功")
                else:
                    print(f"警告: 舵机 ID: {servo_id} 没有对应的位姿数据")
                    success = False
            
            # 等待运动完成
            print("等待运动完成...")
            time.sleep(2)
            
            # 验证是否到达目标位置
            print("验证位置...")
            for i, servo_id in enumerate(self.servo_ids):
                if i < len(pose_data):
                    current_pos = self.read_servo_position(servo_id)
                    target_pos = pose_data[i]
                    if current_pos is not None:
                        error = abs(current_pos - target_pos)
                        print(f"舵机 ID: {servo_id}, 位置误差: {error:.4f} 弧度")
            
            print(f"{'成功' if success else '失败'}移动到 {pose_name}")
            return success

    def run_calibration(self, config_file='so100_calibration.json'):
        """运行完整标定流程"""
        with self:  # 使用上下文管理器
            print("=== SO100 机械臂标定程序 ===")
            
            self.set_torque(False)  # 禁用扭矩，允许手动移动
            
            # 标定限位和位姿
            calibration_data = {
                "joint_limits": self.calibrate_joint_limits(),
                "poses": self.calibrate_poses()
            }
            
            # 保存数据
            if self.save_calibration_data(calibration_data, config_file):
                print("\n=== 标定完成 ===")
                return True
            return False

def main():
    """主函数 - 保持向后兼容性"""
    # 使用默认参数创建标定器
    calibrator = SO100Calibrator()
    # calibrator.run_calibration()
    calibrator.move_to_pose("home_pose")
    # calibrator.move_to_pose("up_pose")
    # calibrator.move_to_pose("init_pose")


if __name__ == "__main__":
    main()
