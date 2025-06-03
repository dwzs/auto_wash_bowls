#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import math
from typing import List

# 导入飞特舵机控制器
from ft_servo_wrapper import FTServoWrapper

class ServoAngleReader:
    """舵机角度读取器"""
    
    def __init__(self, port: str = '/dev/ttyACM0', baudrate: int = 1000000):
        self.controller = FTServoWrapper(port, baudrate)
        self.servo_ids = []
        
    def connect(self) -> bool:
        """连接舵机控制器"""
        return self.controller.connect()
    
    def disconnect(self):
        """断开连接"""
        self.controller.disconnect()
    
    def scan_servos(self, start_id: int = 1, end_id: int = 20) -> List[int]:
        """扫描连接的舵机"""
        print(f"扫描舵机 (ID: {start_id}-{end_id})...")
        self.servo_ids = self.controller.scan_servos(start_id, end_id)
        print(f"发现 {len(self.servo_ids)} 个舵机: {self.servo_ids}")
        return self.servo_ids
    
    def position_to_radians(self, position: int) -> float:
        """将位置值转换为弧度"""
        angle_rad = position / 4096 * 2 * math.pi
        return angle_rad
    
    def real_time_monitor(self, update_interval: float = 0.1):
        """实时监控舵机角度"""
        if not self.servo_ids:
            print("请先扫描舵机")
            return
        
        print(f"\n实时监控 {len(self.servo_ids)} 个舵机角度 (按Ctrl+C停止)")
        print("=" * 40)
        
        try:
            while True:
                # 同步读取所有舵机位置
                positions = self.controller.sync_read_positions(self.servo_ids)
                
                # 显示时间和角度
                current_time = time.strftime('%H:%M:%S')
                print(f"\r{current_time}", end="")
                
                for i, servo_id in enumerate(self.servo_ids):
                    if i < len(positions) and positions[i] is not None:
                        radians = self.position_to_radians(positions[i])
                        print(f" | {servo_id}:{radians:+6.3f}", end="")
                    else:
                        print(f" | {servo_id}:ERROR", end="")
                
                print("", end="\r")
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n\n监控停止")

def main():
    """主函数"""
    reader = ServoAngleReader(port='/dev/ttyACM0', baudrate=1000000)
    
    try:
        # 连接
        if not reader.connect():
            print("连接失败")
            return
        
        # 扫描舵机
        servos = reader.scan_servos(start_id=1, end_id=20)
        
        if servos:
            # 禁用所有舵机的扭矩，使其可以手动转动
            print("禁用舵机扭矩，使其可以手动转动...")
            success = reader.controller.set_all_torque_enable(servos, False)
            print("所有舵机现在可以手动转动")
            
            # 开始实时监控
            reader.real_time_monitor(update_interval=0.1)
        else:
            print("未发现舵机")
    
    except Exception as e:
        print(f"错误: {e}")
    
    finally:
        reader.disconnect()

if __name__ == "__main__":
    main()
