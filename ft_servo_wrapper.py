#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import time
from typing import Optional, List, Tuple, Dict

# 添加上级目录到路径以导入 scservo_sdk
sys.path.append(".")
from scservo_sdk import *

class FTServoWrapper:
    """飞特舵机控制器 - 基于 scservo_sdk 的封装"""
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 1000000):
        """
        初始化飞特舵机控制器
        
        Args:
            port: 串口端口号，如 '/dev/ttyUSB0' 或 'COM1'
            baudrate: 波特率，默认1000000
        """
        self.port = port
        self.baudrate = baudrate
        
        # 初始化端口处理器
        self.port_handler = PortHandler(port)
        
        # 初始化数据包处理器 (使用SMS/STS协议)
        self.packet_handler = sms_sts(self.port_handler)
        
        # 同步读写对象
        self.group_sync_read = None
        self.group_sync_write = None
        
    def connect(self) -> bool:
        """
        连接串口
        
        Returns:
            bool: 连接是否成功
        """
        # 打开端口
        if not self.port_handler.openPort():
            print("Failed to open the port")
            return False
        print("Succeeded to open the port")
        
        # 设置波特率
        if not self.port_handler.setBaudRate(self.baudrate):
            print("Failed to change the baudrate")
            self.port_handler.closePort()
            return False
        print("Succeeded to change the baudrate")
        
        return True
    
    def disconnect(self):
        """断开串口连接"""
        if self.port_handler:
            self.port_handler.closePort()
            print("Port closed")
    
    def ping(self, servo_id: int) -> Tuple[bool, Optional[int]]:
        """
        检测舵机是否在线并获取型号
        
        Args:
            servo_id: 舵机ID
            
        Returns:
            Tuple[bool, Optional[int]]: (是否在线, 舵机型号)
        """
        model_number, comm_result, error = self.packet_handler.ping(servo_id)
        
        if comm_result != COMM_SUCCESS:
            print(f"Ping failed: {self.packet_handler.getTxRxResult(comm_result)}")
            return False, None
        
        if error != 0:
            print(f"Ping error: {self.packet_handler.getRxPacketError(error)}")
            return False, None
            
        print(f"[ID:{servo_id:03d}] ping Succeeded. Model number: {model_number}")
        return True, model_number
    
    def scan_servos(self, start_id: int = 1, end_id: int = 253) -> List[int]:
        """
        扫描在线的舵机
        
        Args:
            start_id: 起始ID
            end_id: 结束ID
            
        Returns:
            List[int]: 在线舵机的ID列表
        """
        online_servos = []
        for servo_id in range(start_id, end_id + 1):
            is_online, _ = self.ping(servo_id)
            if is_online:
                online_servos.append(servo_id)
            time.sleep(0.01)  # 避免通信过快
        return online_servos

    def read_position(self, servo_id: int) -> Optional[int]:
        """
        读取舵机当前位置
        
        Args:
            servo_id: 舵机ID    
        Returns:
            Optional[int]: 当前位置, None表示读取失败
        """
        position, comm_result, error = self.packet_handler.ReadPos(servo_id)    
        
        if comm_result != COMM_SUCCESS:
            print(f"Read position failed: {self.packet_handler.getTxRxResult(comm_result)}")
            return None
        
        if error != 0:
            print(f"Read position error: {self.packet_handler.getRxPacketError(error)}")
            return None 
        
        return position
    
    def read_speed(self, servo_id: int) -> Optional[int]:
        """
        读取舵机当前速度
        
        Args:
            servo_id: 舵机ID        
        Returns:
            Optional[int]: 当前速度, None表示读取失败
        """
        speed, comm_result, error = self.packet_handler.ReadSpeed(servo_id)
        
        if comm_result != COMM_SUCCESS:
            print(f"Read speed failed: {self.packet_handler.getTxRxResult(comm_result)}")
            return None
        
        if error != 0:
            print(f"Read speed error: {self.packet_handler.getRxPacketError(error)}")
            return None     

    def read_position_speed(self, servo_id: int) -> Tuple[Optional[int], Optional[int]]:
        """
        读取舵机当前位置和速度
        
        Args:
            servo_id: 舵机ID
            
        Returns:
            Tuple[Optional[int], Optional[int]]: (当前位置, 当前速度)
        """
        position, speed, comm_result, error = self.packet_handler.ReadPosSpeed(servo_id)
        
        if comm_result != COMM_SUCCESS:
            print(f"Read position/speed failed: {self.packet_handler.getTxRxResult(comm_result)}")
            return None, None
        
        if error != 0:
            print(f"Read position/speed error: {self.packet_handler.getRxPacketError(error)}")
            return None, None
            
        return position, speed
    
    def write_position(self, servo_id: int, position: int, speed: int = 60, acceleration: int = 50) -> bool:
        """
        写入舵机位置 (带速度和加速度控制)
        
        Args:
            servo_id: 舵机ID
            position: 目标位置 (0-4095)
            speed: 最大速度 (速度*0.732=rpm)
            acceleration: 加速度 (加速度*8.7=deg/s²)
            
        Returns:
            bool: 写入是否成功
        """
        comm_result, error = self.packet_handler.WritePosEx(servo_id, position, speed, acceleration)
        
        if comm_result != COMM_SUCCESS:
            print(f"Write position failed: {self.packet_handler.getTxRxResult(comm_result)}")
            return False
        
        if error != 0:
            print(f"Write position error: {self.packet_handler.getRxPacketError(error)}")
            return False
            
        return True

    def reg_write_position(self, servo_id: int, position: int, speed: int = 60, acceleration: int = 50) -> bool:
        """
        寄存器写入位置 (需要调用reg_action执行)
        
        Args:
            servo_id: 舵机ID
            position: 目标位置
            speed: 速度
            acceleration: 加速度
            
        Returns:
            bool: 写入是否成功
        """
        comm_result, error = self.packet_handler.RegWritePosEx(servo_id, position, speed, acceleration)
        
        if comm_result != COMM_SUCCESS:
            print(f"Reg write position failed: {self.packet_handler.getTxRxResult(comm_result)}")
            return False
        
        if error != 0:
            print(f"Reg write position error: {self.packet_handler.getRxPacketError(error)}")
            return False
            
        return True
    
    def reg_action(self) -> bool:
        """
        执行寄存器写入的动作
        
        Returns:
            bool: 执行是否成功
        """
        return self.packet_handler.RegAction()
    
    def sync_write_positions(self, servo_data: List[Tuple[int, int, int, int]]) -> bool:
        """
        同步写入多个舵机位置
        
        Args:
            servo_data: [(servo_id, position, speed, acceleration), ...]
            
        Returns:
            bool: 同步写入是否成功
        """
        # 添加参数到同步写入
        for servo_id, position, speed, acceleration in servo_data:
            result = self.packet_handler.SyncWritePosEx(servo_id, position, speed, acceleration)
            if not result:
                print(f"[ID:{servo_id:03d}] groupSyncWrite addparam failed")
                return False
        
        # 发送同步写入数据包
        comm_result = self.packet_handler.groupSyncWrite.txPacket()
        if comm_result != COMM_SUCCESS:
            print(f"Sync write failed: {self.packet_handler.getTxRxResult(comm_result)}")
            self.packet_handler.groupSyncWrite.clearParam()
            return False
        
        # 清除参数
        self.packet_handler.groupSyncWrite.clearParam()
        return True
    
    def sync_read_positions(self, servo_ids: List[int]) -> list[int]:
        """
        同步读取多个舵机位置和速度
        
        Args:
            servo_ids: 舵机ID列表
            
        Returns:
            Dict[int, Tuple[Optional[int], Optional[int]]]: {servo_id: (position, speed)}
        """
        # 创建同步读取对象
        group_sync_read = GroupSyncRead(self.packet_handler, SMS_STS_PRESENT_POSITION_L, 4)
        
        # 添加参数
        for servo_id in servo_ids:
            result = group_sync_read.addParam(servo_id)
            if not result:
                print(f"[ID:{servo_id:03d}] groupSyncRead addparam failed")
        
        # 发送同步读取数据包
        comm_result = group_sync_read.txRxPacket()
        if comm_result != COMM_SUCCESS:
            print(f"Sync read failed: {self.packet_handler.getTxRxResult(comm_result)}")
            return {}
        
        # 读取数据
        results = []
        for servo_id in servo_ids:
            data_result, error = group_sync_read.isAvailable(servo_id, SMS_STS_PRESENT_POSITION_L, 4)
            if data_result:
                position = group_sync_read.getData(servo_id, SMS_STS_PRESENT_POSITION_L, 2)
                # speed = group_sync_read.getData(servo_id, SMS_STS_PRESENT_SPEED_L, 2)
                # 转换速度为有符号数
                # speed = self.packet_handler.scs_tohost(speed, 15)
                results.append(position)
            else:
                print(f"[ID:{servo_id:03d}] groupSyncRead getdata failed")
                results.append(None)
            
            if error != 0:
                print(f"[ID:{servo_id:03d}] error: {self.packet_handler.getRxPacketError(error)}")
        
        # 清除参数
        group_sync_read.clearParam()
        return results

    def read_servo_errors(self, servo_id: int) -> Optional[Dict[str, bool]]:
        """
        读取舵机状态和错误信息
        
        Args:
            servo_id: 舵机ID
            
        Returns:
            Optional[Dict[str, bool]]: 舵机状态字典，包含各种错误标志位
                {
                    'voltage_error': bool,      # 电压错误
                    'angle_error': bool,        # 角度传感器错误  
                    'overheat_error': bool,     # 过热错误
                    'overele_error': bool,      # 过电流错误
                    'overload_error': bool      # 过载错误
                }
                None表示读取失败
        """
        # 发送ping命令获取错误状态
        _, comm_result, error = self.packet_handler.ping(servo_id)
        
        if comm_result != COMM_SUCCESS:
            print(f"Read servo status failed: {self.packet_handler.getTxRxResult(comm_result)}")
            return None
        
        # 解析错误位
        status = {
            'voltage_error': bool(error & 1),      # ERRBIT_VOLTAGE = 1
            'angle_error': bool(error & 2),        # ERRBIT_ANGLE = 2  
            'overheat_error': bool(error & 4),     # ERRBIT_OVERHEAT = 4
            'overele_error': bool(error & 8),      # ERRBIT_OVERELE = 8
            'overload_error': bool(error & 32)     # ERRBIT_OVERLOAD = 32
        }
        
        return status
    
    def get_servo_error_description(self, servo_id: int) -> Optional[str]:
        """
        获取舵机错误的文字描述
        
        Args:
            servo_id: 舵机ID
            
        Returns:
            Optional[str]: 错误描述，None表示读取失败
        """
        # 发送ping命令获取错误状态
        _, comm_result, error = self.packet_handler.ping(servo_id)
        
        if comm_result != COMM_SUCCESS:
            print(f"Read servo error failed: {self.packet_handler.getTxRxResult(comm_result)}")
            return None
        
        # 使用协议处理器的错误描述方法
        error_description = self.packet_handler.getRxPacketError(error)
        
        if error_description:
            return error_description
        else:
            return "No error"

    def set_torque_enable(self, servo_id: int, enable: bool = True) -> bool:
        """
        设置单个舵机的扭矩使能状态
        
        Args:
            servo_id: 舵机ID
            enable: True为启用扭矩(锁定)，False为禁用扭矩(可手动转动)
            
        Returns:
            bool: 设置是否成功
        """
        value = 1 if enable else 0
        comm_result, error = self.packet_handler.write1ByteTxRx(servo_id, SMS_STS_TORQUE_ENABLE, value)
        
        if comm_result != COMM_SUCCESS:
            print(f"Set torque enable failed: {self.packet_handler.getTxRxResult(comm_result)}")
            return False
        
        if error != 0:
            print(f"Set torque enable error: {self.packet_handler.getRxPacketError(error)}")
            return False
            
        action = "启用" if enable else "禁用"
        print(f"[ID:{servo_id:03d}] 扭矩{action}成功")
        return True
    
    def set_all_torque_enable(self, servo_ids: List[int], enable: bool = True) -> bool:
        """
        设置多个舵机的扭矩使能状态
        
        Args:
            servo_ids: 舵机ID列表
            enable: True为启用扭矩(锁定)，False为禁用扭矩(可手动转动)
            
        Returns:
            bool: 所有舵机设置是否成功
        """
        success = True
        action = "启用" if enable else "禁用"
        
        for servo_id in servo_ids:
            if not self.set_torque_enable(servo_id, enable):
                print(f"警告: 舵机 ID: {servo_id} 扭矩{action}失败")
                success = False
        
        if success:
            print(f"所有舵机扭矩{action}成功")
        
        return success
    
    def read_torque_enable(self, servo_id: int) -> Optional[bool]:
        """
        读取舵机的扭矩使能状态
        
        Args:
            servo_id: 舵机ID
            
        Returns:
            Optional[bool]: True为扭矩启用，False为扭矩禁用，None为读取失败
        """
        value, comm_result, error = self.packet_handler.read1ByteTxRx(servo_id, SMS_STS_TORQUE_ENABLE)
        
        if comm_result != COMM_SUCCESS:
            print(f"Read torque enable failed: {self.packet_handler.getTxRxResult(comm_result)}")
            return None
        
        if error != 0:
            print(f"Read torque enable error: {self.packet_handler.getRxPacketError(error)}")
            return None
            
        return bool(value)

# 使用示例
if __name__ == "__main__":
    ids = [1, 2, 3, 4, 5, 6]
    controller = FTServoWrapper('/dev/ttyACM0')
    
    # 重新连接
    if not controller.connect():
        print("连接失败")
        exit(1)
    
    # zero_joints = [2039, 1916, 2086, 2034, 100, 2180]

    # 写入位置
    # controller.sync_write_positions([(1, 2048, 200, 100), (2, 1916, 200, 100), (3, 2086, 200, 100), (4, 2034, 200, 100), (5, 100, 200, 100), (6, 2180, 200, 100)])
    # controller.sync_write_positions([(1, 2039, 200, 100)])
    while True:
        positions = controller.sync_read_positions(ids)
        print(positions)
        time.sleep(0.1)



    # 错误码读取
    # controller.write_position(1, 2048, 200, 100)
    # while True:
    #     positions = controller.read_position(4)
    
    #     # 检查舵机状态
    #     status = controller.read_servo_errors(4)
    #     error_desc = controller.get_servo_error_description(4)
        
    #     print("position: ", positions)
    #     print("error_id: ", status)
    #     print("error_desc: ", error_desc)
    #     time.sleep(0.1)

    # # 使用完毕后断开连接
    controller.disconnect()
