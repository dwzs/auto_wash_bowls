#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO-100夹爪控制器 - 直接硬件控制版本"""

import json
import time
import os
from typing import Optional
from loguru import logger
from .so100_driver import So100Driver

class GripperController:
    """夹爪控制器 - 直接硬件控制版本"""
    
    def __init__(self):
        # 加载配置
        config_path = os.path.join(os.path.dirname(__file__), "config", "gripper_config.json")
        self.config = self._load_config(config_path)
        
        # 夹爪参数
        self.joint_index = self.config.get("joint_index", 5)  # 夹爪在关节数组中的索引
        self.open_position = self.config.get("open_position", 1.2)    # 打开位置
        self.close_position = self.config.get("close_position", -0.15)  # 关闭位置
        self.tolerance = self.config.get("tolerance", 0.05)  # 位置容差
        
        # 初始化硬件驱动（复用机械臂的驱动）
        self.so100_driver = So100Driver()
        if not hasattr(self.so100_driver, 'initialized') or not self.so100_driver.initialized:
            logger.error("硬件驱动初始化失败")
            raise RuntimeError("Failed to initialize hardware driver")
        
        logger.info(f"夹爪控制器初始化完成 - 关节索引:{self.joint_index}, 开:{self.open_position}, 关:{self.close_position}")
    
    def _load_config(self, config_file: str) -> dict:
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件未找到 {config_file}, 使用默认配置")
            return {}
        except json.JSONDecodeError as e:
            logger.warning(f"配置文件格式错误 {e}, 使用默认配置")
            return {}
    
    def _send_gripper_command(self, position: float) -> bool:
        """发送夹爪命令 - 其他关节用None跳过"""
        try:
            # 准备关节位置数组，只设置夹爪关节
            joint_positions = [None] * 6  # 6个关节
            joint_positions[self.joint_index] = position  # 设置夹爪位置
            
            success = self.so100_driver.write_joints(joint_positions)
            if not success:
                logger.error("发送夹爪命令失败")
                return False
            return True
        except Exception as e:
            logger.error(f"发送夹爪命令出错: {e}")
            return False
    
    def _wait_for_position(self, target_pos: float, timeout: float) -> bool:
        """等待到达目标位置"""
        if timeout <= 0:
            return True
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            current_pos = self.get_joint()
            if current_pos is not None and abs(current_pos - target_pos) < self.tolerance:
                return True
            time.sleep(0.05)  # 50ms检查一次
        
        logger.warning(f"等待夹爪到达位置超时: 目标{target_pos:.3f}, 当前{self.get_joint()}")
        return False
    
    # ==================== 对外接口 ====================
    
    def open(self, timeout: float = 5.0) -> bool:
        """打开夹爪
        
        Args:
            timeout: 超时时间，>0时等待到位，<=0时不等待
            
        Returns:
            成功返回True，失败或超时返回False
        """
        if not self._send_gripper_command(self.open_position):
            return False
        
        logger.info("打开夹爪")
        return self._wait_for_position(self.open_position, timeout)
    
    def close(self, timeout: float = 5.0) -> bool:
        """关闭夹爪
        
        Args:
            timeout: 超时时间，>0时等待到位，<=0时不等待
            
        Returns:
            成功返回True，失败或超时返回False
        """
        if not self._send_gripper_command(self.close_position):
            return False
        
        logger.info("关闭夹爪")
        return self._wait_for_position(self.close_position, timeout)
    
    def set_joint(self, position: float, timeout: float = 5.0) -> bool:
        """设置夹爪关节位置
        
        Args:
            position: 目标位置
            timeout: 超时时间，>0时等待到位，<=0时不等待
            
        Returns:
            成功返回True，失败或超时返回False
        """
        # 限制位置范围
        min_pos = min(self.open_position, self.close_position)
        max_pos = max(self.open_position, self.close_position)
        position = max(min_pos, min(max_pos, position))
        
        if not self._send_gripper_command(position):
            return False
        
        logger.info(f"设置夹爪位置: {position:.3f}")
        return self._wait_for_position(position, timeout)
    
    def get_joint(self) -> Optional[float]:
        """获取夹爪关节位置
        
        Returns:
            当前夹爪位置，如果无法获取则返回None
        """
        try:
            joints = self.so100_driver.read_joints()
            if not joints or len(joints) <= self.joint_index:
                return None
            return joints[self.joint_index]
        except Exception as e:
            logger.error(f"获取夹爪位置失败: {e}")
            return None
    
    def set_opening_percentage(self, percent: float, timeout: float = 5.0) -> bool:
        """按百分比设置夹爪开度
        
        Args:
            percent: 开度百分比 (0-100)，0为完全关闭，100为完全打开
            timeout: 超时时间，>0时等待到位，<=0时不等待
            
        Returns:
            成功返回True，失败或超时返回False
        """
        # 限制百分比范围
        percent = max(0.0, min(100.0, percent))
        
        # 计算目标位置
        range_total = self.open_position - self.close_position
        target_pos = self.close_position + (percent / 100.0) * range_total
        
        if not self._send_gripper_command(target_pos):
            return False
        
        logger.info(f"设置夹爪开度: {percent:.1f}%")
        return self._wait_for_position(target_pos, timeout)
    
    def get_opening_percentage(self) -> Optional[float]:
        """获取夹爪开度百分比
        
        Returns:
            开度百分比 (0-100)，如果无法获取则返回None
        """
        position = self.get_joint()
        if position is None:
            return None
        
        # 计算百分比
        range_total = self.open_position - self.close_position
        if abs(range_total) < 1e-6:  # 避免除零
            return 0.0
        
        percentage = (position - self.close_position) / range_total * 100.0
        return max(0.0, min(100.0, percentage))
    
    def destroy_node(self):
        """安全销毁"""
        if hasattr(self, 'so100_driver'):
            self.so100_driver.stop()

def test_gripper_basic(gripper):
    """基础夹爪测试"""
    print("=== 基础夹爪测试 ===")
    
    print("当前夹爪位置:")
    current_pos = gripper.get_joint()
    current_percent = gripper.get_opening_percentage()
    print(f"  位置: {current_pos}")
    print(f"  开度: {current_percent:.1f}%")
    
    print("\n打开夹爪...")
    success = gripper.open(timeout=3)
    print(f"打开结果: {success}")
    print(f"当前位置: {gripper.get_joint()}")
    
    input("按回车继续...")
    
    print("\n关闭夹爪...")
    success = gripper.close(timeout=3)
    print(f"关闭结果: {success}")
    print(f"当前位置: {gripper.get_joint()}")

def test_gripper_percentage(gripper):
    """百分比开度测试"""
    print("=== 百分比开度测试 ===")
    
    percentages = [0, 25, 50, 75, 100]
    
    for percent in percentages:
        print(f"\n设置开度为 {percent}%...")
        success = gripper.set_opening_percentage(percent, timeout=2)
        current_percent = gripper.get_opening_percentage()
        print(f"设置结果: {success}")
        print(f"当前开度: {current_percent:.1f}%")
        input("按回车继续...")

def test_gripper_oscillation(gripper):
    """夹爪来回摆动测试"""
    print("=== 夹爪摆动测试 ===")
    print("按Ctrl+C停止")
    
    joint_max = gripper.open_position
    joint_min = gripper.close_position
    step = 0.01
    going_up = True
    joint = joint_min
    
    try:
        while True:
            gripper.set_joint(joint, timeout=0)
            print(f"关节位置: {joint:.3f} {'↑' if going_up else '↓'}")
            
            if going_up:
                joint += step
                if joint >= joint_max:
                    going_up = False
            else:
                joint -= step
                if joint <= joint_min:
                    going_up = True
            
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n摆动测试结束")

def main():
    """测试函数"""
    gripper = None
    
    try:
        gripper = GripperController()
        
        # 测试列表
        tests = {
            '1': ('基础夹爪测试', test_gripper_basic),
            '2': ('百分比开度测试', test_gripper_percentage),
            '3': ('夹爪摆动测试', test_gripper_oscillation),
        }
        
        print("可用测试：")
        for key, (name, _) in tests.items():
            print(f"  {key}: {name}")
        
        choice = input(f"\n请选择测试 (1-{len(tests)}, 回车跳过): ").strip()
        
        if choice in tests:
            test_name, test_func = tests[choice]
            print(f"\n开始测试: {test_name}")
            test_func(gripper)
        else:
            print("跳过测试，程序正常结束")

    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if gripper:
            try:
                gripper.destroy_node()
            except Exception as e:
                logger.error(f"销毁节点错误: {e}")

if __name__ == '__main__':
    logger.remove()
    logger.add(
        os.sys.stdout,
        format="<level>{level: <4}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    main()
